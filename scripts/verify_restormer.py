import os
import sys
import time
import torch
import torch.nn as nn
from thop import profile, clever_format

# Add root directory to path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_model
import yaml

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = AttrDict(value)

def main():
    print("="*60)
    print("Verification Script: Restormer (Experiment 005.1)")
    print("="*60)
    
    # Load smoke config
    config_path = "configs/experiment_0051_restormer_smoke.yaml"
    with open(config_path, 'r') as f:
        cfg = AttrDict(yaml.safe_load(f))
    
    print(f"Building model: {cfg.model.type}")
    model = build_model(cfg)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Dummy input for 128x128 x2 SR
    batch_size = 4 # Small batch for verification
    dummy_input = torch.randn(batch_size, 1, 128, 128).to(device)
    
    print(f"Device: {device}")
    
    # 1. THOP Profile (Parameters & FLOPs)
    print("\n--- Profiling with THOP ---")
    try:
        # Profile on a single instance to get standard FLOPs
        single_input = torch.randn(1, 1, 128, 128).to(device)
        flops, params = profile(model, inputs=(single_input,), verbose=False)
        flops_fmt, params_fmt = clever_format([flops, params], "%.2f")
        print(f"Exact Parameters: {params:,}")
        print(f"Exact FLOPs (1 image): {flops:,}")
        print(f"Formatted: Params={params_fmt}, FLOPs={flops_fmt}")
    except Exception as e:
        print(f"THOP Profiling failed: {e}")
        
    # 2. Forward Pass & Timing
    print("\n--- Forward Pass ---")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
    start_time = time.time()
    try:
        output = model(dummy_input)
        end_time = time.time()
        
        print(f"Input shape:  {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        
        expected_shape = (batch_size, 1, 256, 256)
        if output.shape == expected_shape:
            print("[SUCCESS] Forward pass (Output shape is correct)")
        else:
            print(f"[FAILED] Output shape mismatch. Expected {expected_shape}")
            
        print(f"Forward time ({batch_size} images): {(end_time - start_time)*1000:.2f} ms")
        
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
            print(f"Peak GPU Memory (Forward): {peak_mem:.2f} MB")
            
    except Exception as e:
        print(f"[FAILED] Forward pass failed: {e}")
        return
        
    # 3. Backward Pass
    print("\n--- Backward Pass ---")
    try:
        loss = output.mean()
        loss.backward()
        print("[SUCCESS] Backward pass")
        
        if torch.cuda.is_available():
            peak_mem_bw = torch.cuda.max_memory_allocated() / (1024**2)
            print(f"Peak GPU Memory (Forward+Backward): {peak_mem_bw:.2f} MB")
    except Exception as e:
        print(f"[FAILED] Backward pass failed: {e}")

if __name__ == "__main__":
    main()
