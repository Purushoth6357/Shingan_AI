import torch
from torchmetrics import Metric
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

class GradientMagnitudeCorrelation(Metric):
    """
    Computes Gradient Magnitude Correlation (GMC) between predicted and GT images.
    Used for assessing edge and texture localization.
    """
    is_differentiable = True
    higher_is_better = True
    full_state_update = False

    def __init__(self, data_range=1.0, **kwargs):
        super().__init__(**kwargs)
        self.data_range = data_range
        self.register_buffer('gx', torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('gy', torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32).view(1, 1, 3, 3))
        
        self.add_state("sum_corr", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_batches", default=torch.tensor(0.0), dist_reduce_fx="sum")
        
    def _compute_magnitude(self, img):
        # We assume grayscale (1 channel)
        grad_x = torch.nn.functional.conv2d(img, self.gx, padding=1)
        grad_y = torch.nn.functional.conv2d(img, self.gy, padding=1)
        return torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        
    def update(self, preds, targets):
        mag_preds = self._compute_magnitude(preds)
        mag_targets = self._compute_magnitude(targets)
        
        B = preds.size(0)
        p_flat = mag_preds.view(B, -1)
        t_flat = mag_targets.view(B, -1)
        
        p_mean = p_flat.mean(dim=1, keepdim=True)
        t_mean = t_flat.mean(dim=1, keepdim=True)
        
        p_zm = p_flat - p_mean
        t_zm = t_flat - t_mean
        
        cov = (p_zm * t_zm).sum(dim=1)
        p_var = (p_zm ** 2).sum(dim=1)
        t_var = (t_zm ** 2).sum(dim=1)
        
        corr = cov / torch.sqrt(p_var * t_var + 1e-8)
        batch_mean_corr = corr.mean()
        
        self.sum_corr += batch_mean_corr
        self.total_batches += 1
        
    def compute(self):
        return self.sum_corr / self.total_batches if self.total_batches > 0 else torch.tensor(0.0, device=self.sum_corr.device)

def get_metrics(device="cpu"):
    """
    Initializes and returns the evaluation metrics.
    We use torchmetrics to ensure calculation on GPU if available, providing speedups.
    """
    metrics = {
        "psnr": PeakSignalNoiseRatio(data_range=1.0).to(device),
        "ssim": StructuralSimilarityIndexMeasure(data_range=1.0).to(device),
        "gmc": GradientMagnitudeCorrelation(data_range=1.0).to(device)
    }
    return metrics

def compute_metrics(metrics_dict, preds, targets):
    """
    Computes all metrics in the dictionary for the given batch.
    preds: (B, C, H, W) in range [0, 1]
    targets: (B, C, H, W) in range [0, 1]
    
    Returns a dictionary of computed values.
    """
    results = {}
    for name, metric in metrics_dict.items():
        results[name] = metric(preds, targets).item()
    return results
