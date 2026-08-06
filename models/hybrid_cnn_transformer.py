import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class LayerNorm(nn.Module):
    """
    Purpose: Applies Layer Normalization across the channel dimension for 2D images.
    Input shape: (B, C, H, W)
    Output shape: (B, C, H, W)
    Computational complexity: O(B * C * H * W)
    Why this block exists for semiconductor image restoration:
        Standard LayerNorm operates on the last dimension. This custom implementation
        allows LayerNorm to operate seamlessly on spatial feature maps, providing stable
        training for Transformer blocks without reshaping overhead.
    """
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))
            self.bias = None
        else:
            self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        sigma = x.var(1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(sigma + 1e-5)
        if self.bias is not None:
            return x * self.weight + self.bias
        return x * self.weight

class MDTA(nn.Module):
    """
    Purpose: Multi-Dconv Head Transposed Attention (Restormer-inspired).
    Input shape: (B, C, H, W)
    Output shape: (B, C, H, W)
    Computational complexity: O(B * C^2 / heads * H * W)
    Why this block exists for semiconductor image restoration:
        Standard self-attention scales quadratically with image resolution (H*W)^2.
        For high-resolution semiconductor images, this is intractable. MDTA computes
        attention across the channel dimension, making it linear with respect to spatial
        resolution, capturing global interactions efficiently.
    """
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, stride=1, padding=1, groups=channels * 3, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class GDFN(nn.Module):
    """
    Purpose: Gated-Dconv Feed-Forward Network.
    Input shape: (B, C, H, W)
    Output shape: (B, C, H, W)
    Computational complexity: O(B * C * expansion_factor * H * W)
    Why this block exists for semiconductor image restoration:
        Provides non-linear feature transformation with a gating mechanism. 
        The depth-wise convolutions inject local spatial context into the 
        otherwise point-wise operations, helping refine thin structural defects.
    """
    def __init__(self, channels, expansion_factor):
        super(GDFN, self).__init__()
        hidden_channels = int(channels * expansion_factor)
        
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, stride=1, padding=1, groups=hidden_channels * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class RestormerBlock(nn.Module):
    """
    Purpose: A single Transformer block combining MDTA and GDFN.
    Input shape: (B, C, H, W)
    Output shape: (B, C, H, W)
    Computational complexity: Dominated by MDTA and GDFN computations.
    Why this block exists for semiconductor image restoration:
        Extracts both long-range dependencies (MDTA) and local gated features (GDFN) 
        without downsampling, preserving the high-frequency structural integrity of the chip image.
    """
    def __init__(self, channels, num_heads, ffn_expansion_factor, layer_norm_type='WithBias'):
        super(RestormerBlock, self).__init__()
        self.norm1 = LayerNorm(channels, layer_norm_type)
        self.attn = MDTA(channels, num_heads)
        self.norm2 = LayerNorm(channels, layer_norm_type)
        self.ffn = GDFN(channels, ffn_expansion_factor)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class ShallowFeatureExtractor(nn.Module):
    """
    Purpose: Extracts initial features from the input image.
    Input shape: (B, in_channels, H, W)
    Output shape: (B, embed_dim, H, W)
    Computational complexity: O(B * in_channels * embed_dim * 3 * 3 * H * W)
    Why this block exists for semiconductor image restoration:
        Projects the low-dimensional input (1-channel grayscale) into a richer, 
        high-dimensional embedding space necessary for the subsequent residual and transformer blocks.
    """
    def __init__(self, in_channels, embed_dim):
        super(ShallowFeatureExtractor, self).__init__()
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)

class CNNResidualBranch(nn.Module):
    """
    Purpose: Processes features using standard CNN residual blocks.
    Input shape: (B, embed_dim, H, W)
    Output shape: (B, embed_dim, H, W)
    Computational complexity: O(num_blocks * B * embed_dim^2 * 3 * 3 * H * W)
    Why this block exists for semiconductor image restoration:
        Transformers are good at global context, but CNNs excel at extracting highly localized, 
        translation-invariant patterns such as edges and specific lithography defects. 
        This branch ensures local inductive biases are preserved.
    """
    def __init__(self, embed_dim, num_blocks):
        super(CNNResidualBranch, self).__init__()
        layers = []
        for _ in range(num_blocks):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
                )
            )
        self.blocks = nn.ModuleList(layers)

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = out + block(out)
        return out

class TransformerBranch(nn.Module):
    """
    Purpose: Processes features using multiple Restormer blocks.
    Input shape: (B, embed_dim, H, W)
    Output shape: (B, embed_dim, H, W)
    Computational complexity: O(num_blocks * MDTA_FLOPs * GDFN_FLOPs)
    Why this block exists for semiconductor image restoration:
        Captures global context and long-range dependencies, which is critical when
        degradations (like global blur or illumination shifts) affect the entire wafer region.
    """
    def __init__(self, embed_dim, num_blocks, num_heads, ffn_expansion_factor, layer_norm_type):
        super(TransformerBranch, self).__init__()
        self.blocks = nn.Sequential(*[
            RestormerBlock(embed_dim, num_heads, ffn_expansion_factor, layer_norm_type)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        return self.blocks(x)

class FeatureFusion(nn.Module):
    """
    Purpose: Fuses features from the CNN branch and Transformer branch.
    Input shape: x_cnn (B, embed_dim, H, W), x_tx (B, embed_dim, H, W)
    Output shape: (B, embed_dim, H, W)
    Computational complexity: O(B * (2*embed_dim) * embed_dim * 1 * 1 * H * W)
    Why this block exists for semiconductor image restoration:
        Intelligently combines the local high-frequency details from the CNN and 
        the global structural information from the Transformer. 
        Currently implements Concat + 1x1 Conv (to be upgraded to gated fusion later).
    """
    def __init__(self, embed_dim):
        super(FeatureFusion, self).__init__()
        # Simple concat + 1x1 conv for Experiment 004 baseline.
        # Future work: Residual Gated Feature Fusion
        self.fusion_conv = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1)

    def forward(self, x_cnn, x_tx):
        fused = torch.cat([x_cnn, x_tx], dim=1)
        return self.fusion_conv(fused)

class PixelShuffleHead(nn.Module):
    """
    Purpose: Upscales the fused features to the target resolution.
    Input shape: (B, embed_dim, H, W)
    Output shape: (B, out_channels, H * upscale_factor, W * upscale_factor)
    Computational complexity: O(B * embed_dim * (out_channels * upscale_factor^2) * 3 * 3 * H * W)
    Why this block exists for semiconductor image restoration:
        Provides efficient sub-pixel convolution for super-resolution, mapping the rich
        low-resolution feature representations directly to the high-resolution output space
        without adding checkerboard artifacts.
    """
    def __init__(self, embed_dim, out_channels, upscale_factor):
        super(PixelShuffleHead, self).__init__()
        self.conv = nn.Conv2d(embed_dim, out_channels * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x

class HybridCNNTransformer(nn.Module):
    """
    Purpose: Main architecture combining CNNs and Transformers.
    Input shape: (B, in_channels, H, W)
    Output shape: (B, out_channels, H * upscale_factor, W * upscale_factor)
    
    Architecture Flow:
        Input -> [Image Quality Analyzer (Placeholder)]
              -> [Prompt Encoder (Placeholder)]
              -> ShallowFeatureExtractor
              -> [CNN Branch || Transformer Branch]
              -> FeatureFusion
              -> PixelShuffleHead -> Output
    """
    def __init__(self, config=None):
        super(HybridCNNTransformer, self).__init__()
        
        # Parse config or use defaults
        if config is not None and hasattr(config, 'model'):
            cfg = config.model
            in_channels = getattr(cfg, 'in_channels', 1)
            out_channels = getattr(cfg, 'out_channels', 1)
            embed_dim = getattr(cfg, 'embed_dim', 64)
            num_cnn_blocks = getattr(cfg, 'num_cnn_blocks', 4)
            num_transformer_blocks = getattr(cfg, 'num_transformer_blocks', 4)
            num_heads = getattr(cfg, 'num_heads', 4)
            ffn_expansion_factor = getattr(cfg, 'ffn_expansion_factor', 2.66)
            upscale_factor = getattr(cfg, 'upscale_factor', 2)
            layer_norm_type = getattr(cfg, 'layer_norm_type', 'WithBias')
        else:
            in_channels = 1
            out_channels = 1
            embed_dim = 64
            num_cnn_blocks = 4
            num_transformer_blocks = 4
            num_heads = 4
            ffn_expansion_factor = 2.66
            upscale_factor = 2
            layer_norm_type = 'WithBias'

        self.shallow_extractor = ShallowFeatureExtractor(in_channels, embed_dim)
        
        # Placeholders for future Image Quality Analyzer and Prompt Encoder interfaces
        # self.quality_analyzer = None
        # self.prompt_encoder = None
        
        self.cnn_branch = CNNResidualBranch(embed_dim, num_cnn_blocks)
        self.tx_branch = TransformerBranch(embed_dim, num_transformer_blocks, num_heads, ffn_expansion_factor, layer_norm_type)
        
        self.feature_fusion = FeatureFusion(embed_dim)
        self.reconstruction_head = PixelShuffleHead(embed_dim, out_channels, upscale_factor)

    def forward(self, x):
        # 1. Future integration point: Quality Analyzer / Prompt Encoder
        # degradation_prompt = self.prompt_encoder(self.quality_analyzer(x))
        
        # 2. Shallow extraction
        feat = self.shallow_extractor(x)
        
        # 3. Parallel Branches
        cnn_feat = self.cnn_branch(feat)
        tx_feat = self.tx_branch(feat)
        
        # 4. Fusion
        fused_feat = self.feature_fusion(cnn_feat, tx_feat)
        
        # 5. Global Residual Connection (optional, added for stability)
        fused_feat = fused_feat + feat
        
        # 6. Reconstruction
        out = self.reconstruction_head(fused_feat)
        
        return out
