import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.losses import FocalFrequencyLoss
from models.baseline_cnn import BaselineCNN
from utils.config import load_config

def verify_exp_003():
    print("=== Verification Script for Experiment 003 ===")
    
    # 1. Check Config Loading
    cfg_path = "configs/experiment_003.yaml"
    if not os.path.exists(cfg_path):
        print(f"ERROR: {cfg_path} not found.")
        return
        
    cfg = load_config(cfg_path)
    print("-> Config loaded successfully.")
    
    # 2. Check Loss Initialization
    try:
        ffl = FocalFrequencyLoss(alpha=1.0)
        print("-> FocalFrequencyLoss initialized successfully.")
    except Exception as e:
        print(f"ERROR initializing FFL: {e}")
        return
        
    # 3. Dummy Forward Pass with FFL
    print("-> Running dummy forward pass for FFL...")
    pred = torch.randn(2, 1, 128, 128)
    gt = torch.randn(2, 1, 128, 128)
    
    try:
        loss = ffl(pred, gt)
        print(f"-> FFL computed successfully. Value: {loss.item():.4f}")
    except Exception as e:
        print(f"ERROR computing FFL: {e}")
        return
        
    print("\n=== All Checks Passed for Experiment 003 ===")

if __name__ == "__main__":
    verify_exp_003()
