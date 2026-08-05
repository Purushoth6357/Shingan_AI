import os
import argparse
import cv2
import numpy as np
import torch
from dataset import ShinganDataset

def verify_integrity(root_dir):
    print(f"--- Verifying Dataset Integrity: {root_dir} ---")
    gt_dir = os.path.join(root_dir, "GT")
    noisy_dir = os.path.join(root_dir, "NoisyLR")
    
    if not os.path.exists(gt_dir) or not os.path.exists(noisy_dir):
        print("ERROR: Missing GT or NoisyLR folders.")
        return False
        
    gt_files = set(os.listdir(gt_dir))
    noisy_files = set(os.listdir(noisy_dir))
    
    missing_in_gt = noisy_files - gt_files
    missing_in_noisy = gt_files - noisy_files
    
    if missing_in_gt:
        print(f"ERROR: {len(missing_in_gt)} files in NoisyLR are missing from GT. Samples: {list(missing_in_gt)[:5]}")
        return False
    if missing_in_noisy:
        print(f"ERROR: {len(missing_in_noisy)} files in GT are missing from NoisyLR. Samples: {list(missing_in_noisy)[:5]}")
        return False
        
    print(f"[OK] Found {len(gt_files)} matching pairs.")
    
    # Check a few samples for resolution and channels
    check_samples = list(gt_files)[:5]
    for fname in check_samples:
        gt_path = os.path.join(gt_dir, fname)
        noisy_path = os.path.join(noisy_dir, fname)
        
        gt_img = cv2.imread(gt_path)
        noisy_img = cv2.imread(noisy_path)
        
        if gt_img is None or noisy_img is None:
            print(f"ERROR: Failed to read {fname}")
            return False
            
        if gt_img.shape != noisy_img.shape:
            print(f"ERROR: Resolution mismatch for {fname}: GT {gt_img.shape} vs NoisyLR {noisy_img.shape}")
            return False
            
        if len(gt_img.shape) != 3 or gt_img.shape[2] != 3:
            print(f"WARNING: Image {fname} might not be 3-channel RGB. Shape: {gt_img.shape}")
            
    print("[OK] Sample resolutions and channel counts match.")
    return True

def test_dataloader(root_dir):
    print(f"\n--- Testing DataLoader ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        dataset = ShinganDataset(root_dir=root_dir)
        print(f"Dataset initialized with {len(dataset)} samples.")
        
        if len(dataset) == 0:
            print("Dataset is empty. Cannot test loading.")
            return
            
        sample = dataset[0]
        noisy_tensor = sample["NoisyLR"]
        gt_tensor = sample["GT"]
        fname = sample["filename"]
        
        print(f"Loaded sample: {fname}")
        print(f"NoisyLR shape: {noisy_tensor.shape}, dtype: {noisy_tensor.dtype}")
        print(f"GT shape: {gt_tensor.shape}, dtype: {gt_tensor.dtype}")
        
        print(f"NoisyLR pixel range: [{noisy_tensor.min().item():.4f}, {noisy_tensor.max().item():.4f}]")
        print(f"GT pixel range: [{gt_tensor.min().item():.4f}, {gt_tensor.max().item():.4f}]")
        
        # Test moving to device
        noisy_tensor = noisy_tensor.to(device)
        gt_tensor = gt_tensor.to(device)
        print(f"[OK] Successfully moved tensors to {device}")
        
    except Exception as e:
        print(f"ERROR during dataset loading: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Dataset Integrity and Loader")
    parser.add_argument("--data_dir", type=str, default="datasets/hackathon_data", help="Path to dataset root")
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"Dataset directory '{args.data_dir}' not found. Please place dataset files or override with --data_dir.")
    else:
        is_valid = verify_integrity(args.data_dir)
        if is_valid:
            test_dataloader(args.data_dir)
