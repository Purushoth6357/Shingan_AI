import os
import argparse
import glob
import torch
import numpy as np
import cv2
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.baseline_cnn import BaselineCNN
from utils.config import load_config

def process_image(model, img_path, device, norm_config):
    # Read image
    is_npy = img_path.lower().endswith('.npy')
    if is_npy:
        noisy_img = np.load(img_path)
    else:
        noisy_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if noisy_img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        noisy_img = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB)
    
    # Preprocess
    noisy_img = noisy_img.astype(np.float32)
    
    method = norm_config.get("method", "none")
    if method == "minmax":
        n_min = norm_config.get("min", 0.0)
        n_max = norm_config.get("max", 1.0)
        if n_max > n_min:
            noisy_img = (noisy_img - n_min) / (n_max - n_min)
    elif method == "zscore":
        mean = norm_config.get("mean", 0.0)
        std = norm_config.get("std", 1.0)
        noisy_img = (noisy_img - mean) / (std + 1e-8)
        
    if noisy_img.ndim == 2:
        noisy_img = np.expand_dims(noisy_img, axis=-1)
        
    noisy_img = np.transpose(noisy_img, (2, 0, 1)) # HWC to CHW
    noisy_tensor = torch.from_numpy(noisy_img).unsqueeze(0).to(device) # Add batch dim

    # Forward pass
    with torch.no_grad():
        preds = model(noisy_tensor)
        preds = torch.clamp(preds, 0.0, 1.0)
        
    # Postprocess (de-normalize and format)
    pred_img = preds.squeeze(0).cpu().numpy() # CHW
    pred_img = np.transpose(pred_img, (1, 2, 0)) # HWC
    
    if method == "minmax":
        n_min = norm_config.get("min", 0.0)
        n_max = norm_config.get("max", 1.0)
        if n_max > n_min:
            pred_img = pred_img * (n_max - n_min) + n_min
    elif method == "zscore":
        mean = norm_config.get("mean", 0.0)
        std = norm_config.get("std", 1.0)
        pred_img = pred_img * (std + 1e-8) + mean
        
    if pred_img.shape[-1] == 1:
        pred_img = pred_img.squeeze(-1)
        
    if not is_npy:
        pred_img = pred_img.clip(0, 255).astype(np.uint8)
    
    return pred_img, is_npy

def main():
    parser = argparse.ArgumentParser(description="Batch Inference for a folder of images/npy files")
    parser.add_argument('--input', type=str, required=True, help="Path to input folder containing noisy images or .npy files")
    parser.add_argument('--output', type=str, required=True, help="Path to output folder to save restored files")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to model checkpoint")
    parser.add_argument('--config', type=str, default='configs/default.yaml', help="Path to config file")
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load config and model
    cfg = load_config(args.config)
    num_blocks = getattr(cfg.model, 'num_blocks', 4)
    model = BaselineCNN(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        features=cfg.model.features,
        upscale_factor=cfg.model.upscale_factor,
        num_blocks=num_blocks
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    print(f"Loaded model from {args.checkpoint}")
    
    # Setup output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Get files to process
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.npy')
    files = [f for f in os.listdir(args.input) if f.lower().endswith(valid_extensions)]
    
    if not files:
        print(f"No valid image files found in {args.input}")
        return
        
    print(f"Found {len(files)} files to process in {args.input}")
    
    for filename in tqdm(files, desc="Processing files"):
        input_path = os.path.join(args.input, filename)
        output_path = os.path.join(args.output, filename)
        
        norm_config = getattr(cfg.data, 'normalization', {"method": "none"})
        if hasattr(norm_config, "__dict__"):
            norm_dict = {k: v for k, v in norm_config.__dict__.items()}
        else:
            norm_dict = norm_config
            
        try:
            pred_img, is_npy = process_image(model, input_path, device, norm_dict)
            
            # Save
            if is_npy:
                np.save(output_path, pred_img)
            else:
                pred_img_bgr = cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, pred_img_bgr)
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    print(f"Inference complete! Results saved to {args.output}")

if __name__ == "__main__":
    main()
