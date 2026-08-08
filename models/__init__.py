# Init
from .hybrid_cnn_transformer import HybridCNNTransformer
from .baseline_cnn import BaselineCNN
from .network_swinir import SwinIR

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
    elif model_type == 'SwinIR':
        return SwinIR(
            img_size=getattr(cfg.model, 'img_size', 128),
            in_chans=getattr(cfg.model, 'in_chans', 1),
            embed_dim=getattr(cfg.model, 'embed_dim', 60),
            depths=getattr(cfg.model, 'depths', [4, 4, 4, 4]),
            num_heads=getattr(cfg.model, 'num_heads', [3, 3, 3, 3]),
            window_size=getattr(cfg.model, 'window_size', 8),
            mlp_ratio=getattr(cfg.model, 'mlp_ratio', 2.0),
            upscale=getattr(cfg.model, 'upscale_factor', 2),
            upsampler=getattr(cfg.model, 'upsampler', 'pixelshuffle'),
            resi_connection=getattr(cfg.model, 'resi_connection', '1conv')
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
