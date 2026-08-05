import os
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

class ShinganDataset(Dataset):
    """
    Dataset for Shingan AI: AI-Based Restoration of Degraded Images for Semiconductor Inspection.
    Expects a root directory with `GT` and `NoisyLR` subdirectories.
    """
    def __init__(self, root_dir, transform=None, norm_min=0.0, norm_max=1.0):
        self.root_dir = root_dir
        self.gt_dir = os.path.join(root_dir, "GT")
        self.noisy_dir = os.path.join(root_dir, "NoisyLR")
        self.transform = transform
        self.norm_min = norm_min
        self.norm_max = norm_max
        
        if not os.path.exists(self.gt_dir) or not os.path.exists(self.noisy_dir):
            raise FileNotFoundError(f"Missing GT or NoisyLR directory in {root_dir}")
            
        # We assume matching filenames. We collect them from NoisyLR.
        self.image_filenames = sorted([
            f for f in os.listdir(self.noisy_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.npy'))
        ])
        
        # Verify that all corresponding GT files exist
        for fname in self.image_filenames:
            if not os.path.exists(os.path.join(self.gt_dir, fname)):
                raise FileNotFoundError(f"Missing GT pair for {fname}")
                
    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        fname = self.image_filenames[idx]
        noisy_path = os.path.join(self.noisy_dir, fname)
        gt_path = os.path.join(self.gt_dir, fname)
        # Read images using OpenCV (BGR to RGB) or np.load for .npy files
        if fname.lower().endswith('.npy'):
            noisy_img = np.load(noisy_path)
            gt_img = np.load(gt_path)
        else:
            noisy_img = cv2.imread(noisy_path, cv2.IMREAD_COLOR)
            gt_img = cv2.imread(gt_path, cv2.IMREAD_COLOR)
            
            if noisy_img is None:
                raise ValueError(f"Failed to load image: {noisy_path}")
            if gt_img is None:
                raise ValueError(f"Failed to load image: {gt_path}")
                
            noisy_img = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB)
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
        # Apply albumentations transforms if any (requires HWC format)
        if self.transform:
            augmented = self.transform(image=noisy_img, target=gt_img)
            noisy_img = augmented['image']
            gt_img = augmented['target']
            
        if not isinstance(noisy_img, torch.Tensor):
            noisy_img = noisy_img.astype(np.float32)
            gt_img = gt_img.astype(np.float32)
            
            # Custom normalization using config parameters
            if self.norm_max > self.norm_min:
                noisy_img = (noisy_img - self.norm_min) / (self.norm_max - self.norm_min)
                gt_img = (gt_img - self.norm_min) / (self.norm_max - self.norm_min)
            
            # Handle 2D arrays by expanding to (H, W, 1) to simulate channels
            if noisy_img.ndim == 2:
                noisy_img = np.expand_dims(noisy_img, axis=-1)
                gt_img = np.expand_dims(gt_img, axis=-1)
            
            # HWC to CHW
            noisy_img = np.transpose(noisy_img, (2, 0, 1))
            gt_img = np.transpose(gt_img, (2, 0, 1))
            
            noisy_tensor = torch.from_numpy(noisy_img)
            gt_tensor = torch.from_numpy(gt_img)
        else:
            noisy_tensor = noisy_img
            gt_tensor = gt_img
            
        return {"NoisyLR": noisy_tensor, "GT": gt_tensor, "filename": fname}
