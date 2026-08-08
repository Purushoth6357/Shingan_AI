import torch
import torch.nn as nn
import torch.nn.functional as F

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1 approximation)"""
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss

class SobelEdgeLoss(nn.Module):
    """
    Computes L1 loss on the X and Y spatial gradients of images 
    using fixed 3x3 Sobel filters.
    """
    def __init__(self):
        super(SobelEdgeLoss, self).__init__()
        
        # Define fixed Sobel kernels
        # Gx: horizontal gradient
        gx = torch.tensor([
            [-1., 0., 1.],
            [-2., 0., 2.],
            [-1., 0., 1.]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0) # (1, 1, 3, 3)
        
        # Gy: vertical gradient
        gy = torch.tensor([
            [-1., -2., -1.],
            [ 0.,  0.,  0.],
            [ 1.,  2.,  1.]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # Register as buffers so they move to the correct device with the model,
        # but are not treated as trainable parameters.
        self.register_buffer('weight_x', gx)
        self.register_buffer('weight_y', gy)
        
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, gt):
        # We assume pred and gt are (B, C, H, W).
        # Since images are 1 channel, this applies perfectly.
        # If multi-channel, we'd need to expand the weights or loop channels.
        
        pred_gx = F.conv2d(pred, self.weight_x, padding=1)
        pred_gy = F.conv2d(pred, self.weight_y, padding=1)
        
        gt_gx = F.conv2d(gt, self.weight_x, padding=1)
        gt_gy = F.conv2d(gt, self.weight_y, padding=1)
        
        # Independent L1 losses on the gradients (more stable than sqrt(gx^2 + gy^2))
        loss_x = self.l1_loss(pred_gx, gt_gx)
        loss_y = self.l1_loss(pred_gy, gt_gy)
        
        return loss_x + loss_y

class HybridLoss(nn.Module):
    """
    Generic wrapper to combine multiple loss functions dynamically based on config weights.
    """
    def __init__(self, loss_dict, loss_weights):
        """
        Args:
            loss_dict (dict): Dictionary mapping component names to instantiated nn.Module losses.
            loss_weights (dict): Dictionary mapping component names to their scalar weights.
        """
        super(HybridLoss, self).__init__()
        self.losses = nn.ModuleDict(loss_dict)
        self.weights = loss_weights
        
    def forward(self, pred, gt):
        total_loss = 0.0
        component_losses = {}
        
        for name, criterion in self.losses.items():
            weight = self.weights.get(name, 1.0)
            loss_val = criterion(pred, gt)
            
            component_losses[name] = loss_val.item()
            total_loss += weight * loss_val
            
        component_losses["total"] = total_loss
        return component_losses

class FocalFrequencyLoss(nn.Module):
    """
    Focal Frequency Loss.
    Computes loss in the frequency domain using 2D FFT, focusing on hard frequencies.
    """
    def __init__(self, alpha=1.0):
        super(FocalFrequencyLoss, self).__init__()
        self.alpha = alpha
        
    def forward(self, pred, gt):
        # Workaround for cuFFT CUFFT_INTERNAL_ERROR which can occur if tensors are not contiguous
        # or if they are float16 and AMP interacts poorly with certain cuFFT versions.
        pred = pred.contiguous().to(dtype=torch.float32)
        gt = gt.contiguous().to(dtype=torch.float32)
        
        freq_pred = torch.fft.fft2(pred, norm='ortho')
        freq_gt = torch.fft.fft2(gt, norm='ortho')
        diff = freq_pred - freq_gt
        
        # Calculate dynamic focal weight matrix
        weight_matrix = torch.abs(diff) ** self.alpha
        weight_matrix = weight_matrix / (torch.amax(weight_matrix, dim=(-2, -1), keepdim=True) + 1e-6)
        
        loss = torch.mean(weight_matrix * (torch.abs(diff) ** 2))
        return loss
