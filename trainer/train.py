import os
import argparse
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config
from datasets.dataset import ShinganDataset
from models import build_model
from models.losses import CharbonnierLoss, SobelEdgeLoss, HybridLoss, FocalFrequencyLoss
from evaluation.evaluator import Evaluator
from utils.logger import ExperimentLogger

try:
    from thop import profile
except ImportError:
    profile = None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--subset', type=int, default=None, help='Train on a subset of data (for smoke testing)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset & DataLoader
    norm_config = getattr(cfg.data, 'normalization', {"method": "none"})
    if hasattr(norm_config, "__dict__"):
        # Convert Config object to dict if it was parsed as such
        norm_dict = {k: v for k, v in norm_config.__dict__.items()}
    else:
        norm_dict = norm_config
    
    train_split = getattr(cfg.data, 'train_split', None)
    val_split = getattr(cfg.data, 'val_split', None)
    
    train_dataset = ShinganDataset(cfg.data.train_dir, norm_config=norm_dict, split_file=train_split)
    val_dataset = ShinganDataset(cfg.data.val_dir, norm_config=norm_dict, split_file=val_split)
    
    if args.subset:
        train_dataset.image_filenames = train_dataset.image_filenames[:args.subset]
        val_dataset.image_filenames = val_dataset.image_filenames[:args.subset]
        print(f"Running in subset mode: {args.subset} samples")

    train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)

    # Dynamically infer upscale factor
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        gt_shape = sample["GT"].shape # (C, H, W)
        lr_shape = sample["NoisyLR"].shape # (C, H, W)
        
        scale_h = gt_shape[1] / lr_shape[1]
        scale_w = gt_shape[2] / lr_shape[2]
        
        if scale_h != scale_w or not scale_h.is_integer():
            raise ValueError(f"Inconsistent or non-integer upscale factor detected: H_scale={scale_h}, W_scale={scale_w}")
            
        cfg.model.upscale_factor = int(scale_h)

    # Initialize Logger
    logger = ExperimentLogger(cfg)
    
    # Save config
    with open(os.path.join(logger.config_dir, "experiment.json"), "w") as f:
        json.dump({
            "dataset_scale": getattr(cfg.model, 'upscale_factor', 2),
            "input_resolution": list(lr_shape) if len(train_dataset) > 0 else "unknown",
            "target_resolution": list(gt_shape) if len(train_dataset) > 0 else "unknown"
        }, f, indent=4)
        
    with open(os.path.join(logger.config_dir, "config.yaml"), "w") as f:
        # Save a basic yaml copy
        f.write(str(cfg.__dict__))

    # Model
    model = build_model(cfg).to(device)

    # Loss and Optimizer
    loss_cfg = getattr(cfg.training, 'loss', None)
    is_hybrid = False
    
    if loss_cfg and getattr(loss_cfg, 'type', 'charbonnier') == 'hybrid':
        is_hybrid = True
        components_cfg = getattr(loss_cfg, 'components', None)
        loss_dict = {}
        loss_weights = {}
        
        # Build components
        if hasattr(components_cfg, '__dict__'):
            comp_dict = components_cfg.__dict__
        else:
            comp_dict = components_cfg if isinstance(components_cfg, dict) else {}
            
        if 'charbonnier' in comp_dict:
            loss_dict['charbonnier'] = CharbonnierLoss()
            loss_weights['charbonnier'] = getattr(comp_dict['charbonnier'], 'weight', 1.0) if hasattr(comp_dict['charbonnier'], 'weight') else comp_dict['charbonnier'].get('weight', 1.0)
            
        if 'sobel' in comp_dict:
            loss_dict['sobel'] = SobelEdgeLoss()
            loss_weights['sobel'] = getattr(comp_dict['sobel'], 'weight', 0.1) if hasattr(comp_dict['sobel'], 'weight') else comp_dict['sobel'].get('weight', 0.1)
            
        if 'focal_frequency' in comp_dict:
            loss_dict['focal_frequency'] = FocalFrequencyLoss()
            loss_weights['focal_frequency'] = getattr(comp_dict['focal_frequency'], 'weight', 0.1) if hasattr(comp_dict['focal_frequency'], 'weight') else comp_dict['focal_frequency'].get('weight', 0.1)
            
        criterion = HybridLoss(loss_dict, loss_weights).to(device)
    else:
        criterion = CharbonnierLoss().to(device)
        
    if getattr(cfg.training, 'optimizer', 'adam') == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)

    evaluator = Evaluator(device=device, metrics_list=cfg.evaluation.metrics)

    start_epoch = 1
    best_psnr = 0.0

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming training from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
            if 'best_psnr' in checkpoint:
                best_psnr = checkpoint['best_psnr']
            print(f"Resumed from epoch {start_epoch - 1} with best PSNR {best_psnr:.4f}")
        else:
            # Fallback to legacy checkpoints which just contain model weights
            model.load_state_dict(checkpoint)
            print("Loaded legacy model weights.")

    # FLOPs and Params
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    flops = 0
    if profile and len(train_dataset) > 0:
        try:
            dummy_input = torch.randn(1, *lr_shape).to(device)
            flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
        except Exception as e:
            print(f"Warning: FLOPs profiling failed: {e}")

    # Sanity Report
    method = norm_dict.get("method", "none")
    opt_name = getattr(cfg.training, 'optimizer', 'adam').upper()
    print("======================================")
    print(f"Experiment {logger.exp_name} Sanity Report")
    print(f"Dataset Images   : {len(train_dataset)}")
    if len(train_dataset) > 0:
        print(f"Input Shape      : {tuple(lr_shape)}")
        print(f"Target Shape     : {tuple(gt_shape)}")
    print(f"Scale Factor     : {cfg.model.upscale_factor}")
    print(f"Normalization    : {method}")
    print(f"Batch Size       : {cfg.data.batch_size}")
    print(f"Optimizer        : {opt_name}")
    print(f"Loss             : {'Hybrid (' + ', '.join(loss_dict.keys()) + ')' if is_hybrid else 'Charbonnier'}")
    model_type = getattr(cfg.model, 'type', 'BaselineCNN')
    print(f"Model            : {model_type}")
    print(f"Parameters       : {num_params:,}")
    print(f"FLOPs (G)        : {flops / 1e9:.2f}" if flops > 0 else "FLOPs (G)        : N/A")
    print(f"Device           : {device}")
    print("======================================")

    val_every = getattr(cfg.validation, 'every_n_epochs', 1)
    
    # Early stopping config
    early_stop_cfg = getattr(cfg.training, 'early_stopping', None)
    
    # Handle both Config objects and raw dicts for robustness
    if hasattr(early_stop_cfg, '__dict__'):
        es_enabled = getattr(early_stop_cfg, 'enabled', False)
        es_patience = getattr(early_stop_cfg, 'patience', 10)
    elif isinstance(early_stop_cfg, dict):
        es_enabled = early_stop_cfg.get('enabled', False)
        es_patience = early_stop_cfg.get('patience', 10)
    else:
        es_enabled = False
        es_patience = 10
        
    epochs_no_improve = 0

    run_stats = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "loss_fn": "Charbonnier",
        "params": num_params,
        "flops_g": flops / 1e9 if flops > 0 else 0,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "train_time": "N/A",
        "inference_fps": 0,
        "best_psnr": 0,
        "best_ssim": 0,
        "best_gmc": 0,
        "final_psnr": 0,
        "final_ssim": 0,
        "final_gmc": 0,
        "peak_gpu_mem_mb": 0,
        "avg_epoch_time": 0
    }

    train_start_time = time.time()
    
    print(f"Starting training for {cfg.training.epochs} epochs...")
    for epoch in range(start_epoch, cfg.training.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_comp_losses = {}
        
        epoch_start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            noisy = batch["NoisyLR"].to(device)
            gt = batch["GT"].to(device)

            optimizer.zero_grad()
            preds = model(noisy)
            
            assert preds.shape == gt.shape, f"Prediction Shape: {preds.shape} != GT Shape: {gt.shape}"
            
            loss_out = criterion(preds, gt)
            
            if isinstance(loss_out, dict):
                loss = loss_out["total"]
                comp_losses = {k: v for k, v in loss_out.items() if k != "total"}
                for k, v in comp_losses.items():
                    epoch_comp_losses[k] = epoch_comp_losses.get(k, 0.0) + v
            else:
                loss = loss_out
                
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            
            if batch_idx % cfg.training.log_interval == 0:
                if isinstance(loss_out, dict):
                    comp_str = " | ".join([f"{k}: {v:.4f}" for k, v in comp_losses.items()])
                    print(f"Epoch [{epoch}/{cfg.training.epochs}] Batch [{batch_idx}/{len(train_loader)}] Total Loss: {loss.item():.6f} | {comp_str}")
                else:
                    print(f"Epoch [{epoch}/{cfg.training.epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.6f}")

        avg_loss = epoch_loss / len(train_loader)
        avg_comp_losses = {f"train_loss_{k}": v / len(train_loader) for k, v in epoch_comp_losses.items()}
        
        epoch_time = time.time() - epoch_start_time
        
        gpu_mem = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
        
        if epoch_comp_losses:
            # We want to print without the 'train_loss_' prefix for brevity in the console
            comp_str = " | ".join([f"{k}: {v / len(train_loader):.4f}" for k, v in epoch_comp_losses.items()])
            print(f"--- Epoch {epoch} completed in {epoch_time:.2f}s. Avg Total Loss: {avg_loss:.6f} | {comp_str} ---")
        else:
            print(f"--- Epoch {epoch} completed in {epoch_time:.2f}s. Avg Loss: {avg_loss:.6f} ---")

        # Validation
        val_loss, psnr, ssim = 0.0, 0.0, 0.0
        if epoch % val_every == 0 or epoch == cfg.training.epochs:
            print("Running validation...")
            val_metrics = evaluator.evaluate(model, val_loader, criterion=criterion, epoch=epoch, logger=logger)
            print(f"Validation Metrics: {val_metrics}")
            
            
            val_loss = val_metrics.get("val_loss", 0)
            
            # Extract validation component losses to pass to logger
            val_comp_losses = {k: v for k, v in val_metrics.items() if k.startswith("val_loss_")}
            if val_comp_losses:
                # Merge into avg_comp_losses for the logger
                for k, v in val_comp_losses.items():
                    avg_comp_losses[k] = v

            psnr = val_metrics.get("psnr", 0)
            ssim = val_metrics.get("ssim", 0)
            gmc = val_metrics.get("gmc", 0)
            run_stats["inference_fps"] = val_metrics.get("throughput_fps", 0)
            
            if epoch == cfg.training.epochs:
                run_stats["final_psnr"] = psnr
                run_stats["final_ssim"] = ssim
                run_stats["final_gmc"] = gmc

            # Save best model
            if psnr > best_psnr:
                best_psnr = psnr
                run_stats["best_psnr"] = best_psnr
                run_stats["best_ssim"] = ssim
                run_stats["best_gmc"] = gmc
                
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_psnr': best_psnr
                }
                torch.save(checkpoint_dict, os.path.join(logger.ckpt_dir, "best_model.pth"))
                print(f"Saved new best model with PSNR: {best_psnr:.4f}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
        # Log to CSV
        lr = optimizer.param_groups[0]['lr']
        logger.log_epoch(epoch, avg_loss, val_loss, psnr, ssim, lr, epoch_time, gpu_mem, avg_comp_losses)
        
        run_stats["peak_gpu_mem_mb"] = max(run_stats["peak_gpu_mem_mb"], gpu_mem)
        run_stats["avg_epoch_time"] = (run_stats["avg_epoch_time"] * (epoch - 1) + epoch_time) / epoch
        
        # Save last model
        last_checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_psnr': best_psnr
        }
        torch.save(last_checkpoint_dict, os.path.join(logger.ckpt_dir, "last_model.pth"))
        
        # Early Stopping
        if es_enabled and epochs_no_improve >= es_patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    train_end_time = time.time()
    total_train_hours = (train_end_time - train_start_time) / 3600
    run_stats["train_time"] = f"{total_train_hours:.2f} hours"

    print("\nTraining Complete! Generating plots and benchmark report...")
    logger.generate_plots()
    logger.generate_benchmark_report(run_stats)
    print(f"Experiment saved to {logger.exp_dir}")

if __name__ == '__main__':
    main()
