import torch
from thop import profile
from models import build_model
from utils.config import load_config
dummy = torch.randn(1, 1, 128, 128)
cfg = load_config('configs/experiment_004.yaml')
for c in [4, 5, 6]:
    for t in [2, 3, 4]:
        cfg.model.num_cnn_blocks = c
        cfg.model.num_transformer_blocks = t
        model = build_model(cfg)
        flops, params = profile(model, inputs=(dummy,), verbose=False)
        print(f"CNN={c}, TX={t} -> Params: {int(params):,}, FLOPs: {flops/1e9:.2f}G")
