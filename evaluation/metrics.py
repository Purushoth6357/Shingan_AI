import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

def get_metrics(device="cpu"):
    """
    Initializes and returns the evaluation metrics.
    We use torchmetrics to ensure calculation on GPU if available, providing speedups.
    """
    metrics = {
        "psnr": PeakSignalNoiseRatio(data_range=1.0).to(device),
        "ssim": StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
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
