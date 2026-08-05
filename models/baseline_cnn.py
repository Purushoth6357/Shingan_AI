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
    def __init__(self, in_channels=3, out_channels=3, features=64, num_blocks=4):
        super().__init__()
        # Initial feature extraction
        self.conv_in = nn.Conv2d(in_channels, features, kernel_size=3, padding=1)
        
        # Residual blocks (no downsampling/pooling to preserve resolution)
        self.res_blocks = nn.Sequential(*[ResBlock(features) for _ in range(num_blocks)])
        
        # Final reconstruction
        self.conv_out = nn.Conv2d(features, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # We learn the residual (noise/degradation) instead of the full image
        out = self.conv_in(x)
        out = self.res_blocks(out)
        out = self.conv_out(out)
        
        # Global skip connection
        return out + x
