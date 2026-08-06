import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
from models import HybridCNNTransformer
from thop import profile

def verify():
    print("=== Hybrid CNN + Transformer Verification ===\n")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # 1. Instantiate the model
    model = HybridCNNTransformer().to(device)
    model.eval()
    
    # 2. Create a dummy input (e.g., 1x1x128x128 for grayscale images)
    dummy_input = torch.randn(1, 1, 128, 128).to(device)
    
    # 3. Output shape and FLOPs
    print("-> Calculating FLOPs and Parameters...")
    with torch.no_grad():
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        output = model(dummy_input)
        
    print(f"Input Shape:  {dummy_input.shape}")
    print(f"Output Shape: {output.shape}")
    print(f"Parameters:   {params / 1e6:.2f} M")
    print(f"FLOPs:        {flops / 1e9:.2f} G\n")
    
    # 4. Memory and Inference FPS profiling
    print("-> Profiling Inference FPS and Peak GPU Memory...")
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
            
    # Measure FPS
    start_time = time.time()
    num_iterations = 100
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)
            
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    end_time = time.time()
    fps = num_iterations / (end_time - start_time)
    print(f"Inference FPS: {fps:.2f} (Batch Size = 1)")
    
    if device.type == 'cuda':
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"Peak GPU Memory: {peak_mem:.2f} MB")
    else:
        print("Peak GPU Memory: N/A (Running on CPU)")
        
    print("\n=== Verification Complete ===")

if __name__ == "__main__":
    verify()
