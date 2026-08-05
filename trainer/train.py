import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config
from datasets.dataset import ShinganDataset
from models.baseline_cnn import BaselineCNN
from models.losses import CharbonnierLoss
from evaluation.evaluator import Evaluator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--subset', type=int, default=None, help='Train on a subset of data (for smoke testing)')
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
    
    train_dataset = ShinganDataset(cfg.data.train_dir, norm_config=norm_dict)
    val_dataset = ShinganDataset(cfg.data.val_dir, norm_config=norm_dict)
    
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
        print(f"Dynamically inferred upscale factor: x{cfg.model.upscale_factor}")

    # Model
    model = BaselineCNN(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        features=cfg.model.features,
        upscale_factor=cfg.model.upscale_factor
    ).to(device)

    # Loss and Optimizer
    criterion = CharbonnierLoss().to(device)
    
    if getattr(cfg.training, 'optimizer', 'adam') == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)

    evaluator = Evaluator(device=device, metrics_list=cfg.evaluation.metrics)

    # Checkpoint dir
    os.makedirs(cfg.training.save_dir, exist_ok=True)
    best_psnr = 0.0

    print(f"Starting training for {cfg.training.epochs} epochs...")
    for epoch in range(1, cfg.training.epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            noisy = batch["NoisyLR"].to(device)
            gt = batch["GT"].to(device)

            optimizer.zero_grad()
            preds = model(noisy)
            
            loss = criterion(preds, gt)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            
            if batch_idx % cfg.training.log_interval == 0:
                print(f"Epoch [{epoch}/{cfg.training.epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.6f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"--- Epoch {epoch} completed. Avg Loss: {avg_loss:.6f} ---")

        # Validation
        print("Running validation...")
        val_metrics = evaluator.evaluate(model, val_loader)
        print(f"Validation Metrics: {val_metrics}")

        # Save best model
        current_psnr = val_metrics.get('psnr', 0)
        if current_psnr > best_psnr:
            best_psnr = current_psnr
            save_path = os.path.join(cfg.training.save_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model with PSNR: {best_psnr:.4f}")

if __name__ == '__main__':
    main()
