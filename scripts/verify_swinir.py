import sys
import os
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.network_swinir import SwinIR
from utils.config import load_config
from thop import profile, clever_format

def main():
    print("--- SwinIR Verification ---")
    config_path = "configs/experiment_005_swinir.yaml"
    cfg = load_config(config_path)
    
    # Initialize model
    print("Instantiating SwinIR...")
    model = SwinIR(
        img_size=cfg.model.img_size,
        in_chans=cfg.model.in_chans,
        embed_dim=cfg.model.embed_dim,
        depths=cfg.model.depths,
        num_heads=cfg.model.num_heads,
        window_size=cfg.model.window_size,
        mlp_ratio=cfg.model.mlp_ratio,
        upscale=cfg.model.upscale_factor,
        upsampler=cfg.model.upsampler,
        resi_connection=cfg.model.resi_connection
    )
    
    # Dummy input
    dummy_input = torch.randn(1, 1, 128, 128)
    print(f"Input shape:  {dummy_input.shape}")
    
    # Forward pass
    try:
        output = model(dummy_input)
        print("Forward pass: SUCCESS")
        print(f"Output shape: {output.shape}")
        
        # Backward pass
        dummy_target = torch.randn_like(output)
        loss = torch.nn.functional.mse_loss(output, dummy_target)
        loss.backward()
        print("Backward pass: SUCCESS")
    except Exception as e:
        print("Forward/Backward pass: FAILED")
        print(e)
        return
    
    # Parameters and FLOPs
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    macs, params_fmt = clever_format([flops, params], "%.3f")
    
    # Since profile returns MACs, FLOPs is roughly 2 * MACs for many operations, 
    # but thop.profile is often referred to as FLOPs/MACs interchangeably. 
    # Let's print raw counts and formatted.
    print(f"Parameters:   {params:,} (Formatted: {params_fmt})")
    print(f"MACs (FLOPs): {flops:,} (Formatted: {macs})")
    
if __name__ == "__main__":
    main()
