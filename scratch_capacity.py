import sys
import os
import torch
from thop import profile
from models import build_model
from utils.config import load_config

def count(model, x):
    flops, params = profile(model, inputs=(x,), verbose=False)
    return params, flops

# User's reported flops are likely using a larger shape (e.g. 512x512 maybe?), but we can deduce the relative scaling. 
# Wait, user said: Current Exp004 | 414,772 | 6.78 G.
# We just want Params around 550k-650k, Flops ~8.5-10G.
# Let's use 128x128 for dummy and see what it gives, then scale the Flops if needed. Or just check params first.

dummy = torch.randn(1, 1, 128, 128)
cfg = load_config('configs/experiment_004.yaml')

# Test current
cfg.model.num_cnn_blocks = 4
cfg.model.num_transformer_blocks = 2
model = build_model(cfg)
p, f = count(model, dummy)
print(f"CNN=4, TX=2 -> Params: {int(p):,}, FLOPs: {f/1e9:.2f}G")

# Test 1
cfg.model.num_cnn_blocks = 5
cfg.model.num_transformer_blocks = 3
model = build_model(cfg)
p, f = count(model, dummy)
print(f"CNN=5, TX=3 -> Params: {int(p):,}, FLOPs: {f/1e9:.2f}G")

# Test 2
cfg.model.num_cnn_blocks = 4
cfg.model.num_transformer_blocks = 4
model = build_model(cfg)
p, f = count(model, dummy)
print(f"CNN=4, TX=4 -> Params: {int(p):,}, FLOPs: {f/1e9:.2f}G")

# Test 3
cfg.model.num_cnn_blocks = 6
cfg.model.num_transformer_blocks = 3
model = build_model(cfg)
p, f = count(model, dummy)
print(f"CNN=6, TX=3 -> Params: {int(p):,}, FLOPs: {f/1e9:.2f}G")
