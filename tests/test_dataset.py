import os
import tempfile
import numpy as np
import torch
import pytest

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.dataset import ShinganDataset

def test_dataset_npy_loading():
    # Create a temporary directory to act as the dataset root
    with tempfile.TemporaryDirectory() as temp_dir:
        gt_dir = os.path.join(temp_dir, "GT")
        noisy_dir = os.path.join(temp_dir, "NoisyLR")
        os.makedirs(gt_dir)
        os.makedirs(noisy_dir)
        
        # Create dummy 2D .npy files with arbitrary float values
        dummy_gt = np.random.uniform(low=-0.0026, high=1.3258, size=(128, 128)).astype(np.float32)
        dummy_noisy = np.random.uniform(low=-0.0026, high=1.3258, size=(128, 128)).astype(np.float32)
        
        np.save(os.path.join(gt_dir, "test_001.npy"), dummy_gt)
        np.save(os.path.join(noisy_dir, "test_001.npy"), dummy_noisy)
        
        # Initialize dataset with normalization parameters
        dataset = ShinganDataset(
            root_dir=temp_dir, 
            norm_min=-0.0026, 
            norm_max=1.3258
        )
        
        assert len(dataset) == 1
        
        # Fetch item
        item = dataset[0]
        assert "GT" in item
        assert "NoisyLR" in item
        assert item["filename"] == "test_001.npy"
        
        # Check types and shapes
        gt_tensor = item["GT"]
        noisy_tensor = item["NoisyLR"]
        
        assert isinstance(gt_tensor, torch.Tensor)
        assert isinstance(noisy_tensor, torch.Tensor)
        
        # Expected shape after 2D -> (H,W,1) -> (1,H,W)
        assert gt_tensor.shape == (1, 128, 128)
        assert noisy_tensor.shape == (1, 128, 128)
        
        # Verify normalization to [0, 1] range
        assert gt_tensor.min() >= 0.0
        assert gt_tensor.max() <= 1.0
        assert noisy_tensor.min() >= 0.0
        assert noisy_tensor.max() <= 1.0
