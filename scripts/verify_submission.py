import os
import sys
import torch

def verify_submission():
    print("="*60)
    print("KLA Submission Readiness Verification")
    print("="*60)
    
    passed = True
    
    # 1. Check essential files
    essential_files = [
        "infer.py",
        "README.md",
        "requirements.txt",
        "configs/experiment_004.yaml",
        "scripts/infer.py"
    ]
    
    print("\n--- File Structure ---")
    for file in essential_files:
        if os.path.exists(file):
            print(f"[SUCCESS] {file} exists")
        else:
            print(f"[FAILED] {file} missing")
            passed = False
            
    # 2. Check Torch & CUDA
    print("\n--- Environment ---")
    try:
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available:  {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Device Name:     {torch.cuda.get_device_name(0)}")
        print("[SUCCESS] Environment Check")
    except Exception as e:
        print(f"[FAILED] Environment Check: {e}")
        passed = False
        
    # 3. Model Loading & Inference Simulation
    print("\n--- Model Verification ---")
    try:
        from utils.config import load_config
        from models import build_model
        
        cfg = load_config("configs/experiment_004.yaml")
        model = build_model(cfg)
        print("[SUCCESS] Config & Model instantiation")
        
        dummy_input = torch.randn(1, 1, 128, 128)
        output = model(dummy_input)
        if output.shape == (1, 1, 256, 256):
            print("[SUCCESS] Forward pass shape (128->256)")
        else:
            print(f"[FAILED] Incorrect output shape: {output.shape}")
            passed = False
            
    except Exception as e:
        print(f"[FAILED] Model Verification: {e}")
        passed = False
        
    print("\n" + "="*60)
    if passed:
        print("✅ Submission Ready")
    else:
        print("❌ Submission Failed Checks")

if __name__ == "__main__":
    verify_submission()
