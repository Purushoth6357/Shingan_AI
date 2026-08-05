import os
import argparse
import numpy as np
from tqdm import tqdm
try:
    import cv2
except ImportError:
    pass

def verify_integrity(root_dir):
    print(f"--- Verifying Dataset Integrity: {root_dir} ---")
    gt_dir = os.path.join(root_dir, "GT")
    noisy_dir = os.path.join(root_dir, "NoisyLR")
    
    if not os.path.exists(gt_dir) or not os.path.exists(noisy_dir):
        print("ERROR: Missing GT or NoisyLR folders.")
        return False
        
    gt_files = sorted(os.listdir(gt_dir))
    noisy_files = set(os.listdir(noisy_dir))
    
    missing_in_noisy = set(gt_files) - noisy_files
    if missing_in_noisy:
        print(f"ERROR: {len(missing_in_noisy)} files in GT are missing from NoisyLR. Samples: {list(missing_in_noisy)[:5]}")
        return False
        
    shapes = set()
    channels = set()
    dtypes = set()
    
    min_vals = []
    max_vals = []
    means = []
    variances = []
    
    corrupted_files = []
    
    print(f"Scanning {len(gt_files)} images to compute global statistics...")
    
    for fname in tqdm(gt_files):
        gt_path = os.path.join(gt_dir, fname)
        
        try:
            if fname.lower().endswith('.npy'):
                img = np.load(gt_path)
            else:
                img = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise ValueError("cv2 failed to read image")
                
            shapes.add(img.shape[:2]) # H, W
            dtypes.add(str(img.dtype))
            
            if img.ndim == 2:
                channels.add(1)
            else:
                channels.add(img.shape[2])
                
            min_vals.append(img.min())
            max_vals.append(img.max())
            means.append(img.mean())
            variances.append(img.var())
                
        except Exception as e:
            corrupted_files.append(fname)
            
    print("\nDataset Summary")
    print("-" * 30)
    print(f"Images           : {len(gt_files) - len(corrupted_files)}")
    
    shape_str = ", ".join([str(s) for s in shapes]) if shapes else "N/A"
    print(f"Shape            : {shape_str}")
    
    channel_str = ", ".join([str(c) for c in channels]) if channels else "N/A"
    print(f"Channels         : {channel_str}")
    
    dtype_str = ", ".join(dtypes) if dtypes else "N/A"
    print(f"dtype            : {dtype_str}")
    
    if min_vals:
        global_min = min(min_vals)
        global_max = max(max_vals)
        global_mean = np.mean(means)
        global_std = np.sqrt(np.mean(variances))
        
        print(f"Global Min       : {global_min:.4f}")
        print(f"Global Max       : {global_max:.4f}")
        print(f"Mean             : {global_mean:.4f}")
        print(f"Std              : {global_std:.4f}")
        
    print(f"Unique Shapes    : {len(shapes)}")
    print(f"Corrupted Files  : {len(corrupted_files)}")
    
    if corrupted_files:
        print(f"List of corrupted files: {corrupted_files[:10]}...")
        
    if len(shapes) > 1:
        print("\nWARNING: Dataset contains multiple shapes. Model might require resizing or cropping.")
        
    return len(corrupted_files) == 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Dataset Integrity and Compute Global Stats")
    parser.add_argument("--data_dir", type=str, default="datasets/hackathon_data", help="Path to dataset root")
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"Dataset directory '{args.data_dir}' not found. Please place dataset files or override with --data_dir.")
    else:
        verify_integrity(args.data_dir)
