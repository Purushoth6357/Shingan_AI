# Init
from .hybrid_cnn_transformer import HybridCNNTransformer
from .baseline_cnn import BaselineCNN

def build_model(cfg):
    model_type = getattr(cfg.model, 'type', 'BaselineCNN')
    
    if model_type == 'BaselineCNN':
        num_blocks = getattr(cfg.model, 'num_blocks', 4)
        return BaselineCNN(
            in_channels=getattr(cfg.model, 'in_channels', 1),
            out_channels=getattr(cfg.model, 'out_channels', 1),
            features=getattr(cfg.model, 'features', 64),
            upscale_factor=getattr(cfg.model, 'upscale_factor', 2),
            num_blocks=num_blocks
        )
    elif model_type == 'HybridCNNTransformer':
        return HybridCNNTransformer(config=cfg)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
