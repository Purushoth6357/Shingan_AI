import os
import argparse
import torch
from torch.utils.data import DataLoader

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.baseline_cnn import BaselineCNN
from datasets.dataset import ShinganDataset
from evaluation.evaluator import Evaluator
from utils.config import load_config

def main():
    parser = argparse.ArgumentParser(description="Re-evaluate models using new validation split")
    parser.add_argument('--checkpoint1', type=str, required=True, help="Path to exp001 checkpoint")
    parser.add_argument('--checkpoint2', type=str, required=True, help="Path to exp002 checkpoint")
    parser.add_argument('--config', type=str, default='configs/default.yaml', help="Config file")
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    cfg = load_config(args.config)
    
    norm_config = getattr(cfg.data, 'normalization', {"method": "none"})
    if hasattr(norm_config, "__dict__"):
        norm_dict = {k: v for k, v in norm_config.__dict__.items()}
    else:
        norm_dict = norm_config
        
    val_split = getattr(cfg.data, 'val_split', 'datasets/splits/val.txt')
    
    print(f"Loading validation dataset using split file: {val_split}")
    val_dataset = ShinganDataset(cfg.data.val_dir, norm_config=norm_dict, split_file=val_split)
    val_loader = DataLoader(val_dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)
    
    evaluator = Evaluator(device=device, metrics_list=["psnr", "ssim", "gmc"])
    
    def evaluate_model(ckpt_path, name):
        print(f"\n--- Evaluating {name} ---")
        model = BaselineCNN(
            in_channels=cfg.model.in_channels, 
            out_channels=cfg.model.out_channels, 
            features=cfg.model.features,
            upscale_factor=cfg.model.upscale_factor
        ).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        
        metrics = evaluator.evaluate(model, val_loader)
        
        print(f"Results for {name}:")
        print(f"  PSNR: {metrics.get('psnr', 0):.4f} dB")
        print(f"  SSIM: {metrics.get('ssim', 0):.4f}")
        print(f"  GMC:  {metrics.get('gmc', 0):.4f}")
        
    evaluate_model(args.checkpoint1, "Experiment 001")
    evaluate_model(args.checkpoint2, "Experiment 002")
    
if __name__ == "__main__":
    main()
