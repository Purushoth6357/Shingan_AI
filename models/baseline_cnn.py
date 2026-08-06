import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        return x + res

class BaselineCNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=64, upscale_factor=2, num_blocks=4):
        super(BaselineCNN, self).__init__()
        
        # Initial feature extraction
        self.conv_in = nn.Conv2d(in_channels, features, kernel_size=3, padding=1)
        
        # Residual blocks (operating at LR spatial resolution)
        self.res_blocks = nn.Sequential(*[ResBlock(features) for _ in range(num_blocks)])
        
        # Post-residual convolution
        self.conv_post = nn.Conv2d(features, features, kernel_size=3, padding=1)
        
        # Upsampling via PixelShuffle
        self.upsample = nn.Sequential(
            nn.Conv2d(features, features * (upscale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor)
        )
        
        # Final output convolution
        self.conv_out = nn.Conv2d(features, out_channels, kernel_size=3, padding=1)
        
        # Standard bicubic upsampling for the global residual skip connection
        self.upscale_factor = upscale_factor

    def forward(self, x):
        # Initial features
        feat = self.conv_in(x)
        
        # Deep features
        res = self.res_blocks(feat)
        res = self.conv_post(res)
        
        # Global residual connection (within feature space)
        feat = feat + res
        
        # Upsampling
        out = self.upsample(feat)
        
        # Final prediction
        out = self.conv_out(out)
        
        # Upsample the input image to match target resolution and add it (Global Residual Learning)
        # This forces the network to learn only the high-frequency details (residuals)
        x_upsampled = torch.nn.functional.interpolate(x, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
        return out + x_upsampled
