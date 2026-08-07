import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from thop import profile
from models.hybrid_cnn_transformer import HybridCNNTransformer
from models.baseline_cnn import BaselineCNN

def count(model, x):
    flops, params = profile(model, inputs=(x,), verbose=False)
    return params, flops

dummy = torch.randn(1, 1, 128, 128)
hybrid = HybridCNNTransformer()
p_h, f_h = count(hybrid, dummy)
print(f"Hybrid: params={p_h/1e6:.4f}M, flops={f_h/1e9:.4f}G")

for f in [64, 80, 96, 128]:
    base = BaselineCNN(features=f)
    p_b, f_b = count(base, dummy)
    print(f"Baseline(f={f}): params={p_b/1e6:.4f}M, flops={f_b/1e9:.4f}G")
