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
    norm_min = getattr(cfg.data, 'norm_min', 0.0)
    norm_max = getattr(cfg.data, 'norm_max', 1.0)
    
    train_dataset = ShinganDataset(cfg.data.train_dir, norm_min=norm_min, norm_max=norm_max)
    val_dataset = ShinganDataset(cfg.data.val_dir, norm_min=norm_min, norm_max=norm_max)
    
    if args.subset:
        train_dataset.image_filenames = train_dataset.image_filenames[:args.subset]
        val_dataset.image_filenames = val_dataset.image_filenames[:args.subset]
        print(f"Running in subset mode: {args.subset} samples")

    train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)

    # Model
    model = BaselineCNN(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        features=cfg.model.features
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
