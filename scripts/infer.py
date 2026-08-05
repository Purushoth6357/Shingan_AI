import os
import argparse
import torch
import numpy as np
import cv2

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.baseline_cnn import BaselineCNN
from utils.config import load_config

def main():
    parser = argparse.ArgumentParser(description="Inference for a single image/npy file")
    parser.add_argument('--input', type=str, required=True, help="Path to input noisy image or .npy")
    parser.add_argument('--output', type=str, default="output.png", help="Path to save the restored image")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to model checkpoint")
    parser.add_argument('--config', type=str, default='configs/default.yaml', help="Path to config file")
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load config and model
    cfg = load_config(args.config)
    model = BaselineCNN(
        in_channels=cfg.model.in_channels, 
        out_channels=cfg.model.out_channels, 
        features=cfg.model.features
    ).to(device)
    
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    print(f"Loaded model from {args.checkpoint}")

    # Read image
    if args.input.lower().endswith('.npy'):
        noisy_img = np.load(args.input)
    else:
        noisy_img = cv2.imread(args.input, cv2.IMREAD_COLOR)
        if noisy_img is None:
            raise ValueError(f"Failed to load image: {args.input}")
        noisy_img = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB)
    
    # Preprocess
    noisy_img = noisy_img.astype(np.float32) / 255.0
    noisy_img = np.transpose(noisy_img, (2, 0, 1)) # HWC to CHW
    noisy_tensor = torch.from_numpy(noisy_img).unsqueeze(0).to(device) # Add batch dim

    # Forward pass
    with torch.no_grad():
        preds = model(noisy_tensor)
        preds = torch.clamp(preds, 0.0, 1.0)
        
    # Postprocess
    pred_img = preds.squeeze(0).cpu().numpy() # CHW
    pred_img = np.transpose(pred_img, (1, 2, 0)) # HWC
    pred_img = (pred_img * 255.0).clip(0, 255).astype(np.uint8)
    
    # Save (Convert back to BGR for cv2)
    if not args.output.lower().endswith('.npy'):
        pred_img_bgr = cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(args.output, pred_img_bgr)
    else:
        np.save(args.output, pred_img)
        
    print(f"Successfully saved restored output to {args.output}")

if __name__ == "__main__":
    main()
