import time
import torch
from evaluation.metrics import get_metrics, compute_metrics
from tqdm import tqdm

class Evaluator:
    def __init__(self, device, metrics_list=["psnr", "ssim"]):
        self.device = device
        self.metrics_list = metrics_list
        # get_metrics returns a dict of torchmetrics objects
        all_metrics = get_metrics(device=self.device)
        self.metrics = {k: v for k, v in all_metrics.items() if k in self.metrics_list}
        
    @torch.no_grad()
    def evaluate(self, model, dataloader, criterion=None, epoch=None, logger=None):
        """
        Runs evaluation on a dataloader.
        """
        model.eval()
        
        total_metrics = {m: 0.0 for m in self.metrics}
        if criterion:
            total_metrics["val_loss"] = 0.0
            
        num_batches = len(dataloader)
        num_samples = 0
        
        start_time = time.time()
        
        # Determine if we should save qualitative samples this epoch
        milestones = [1, 10, 25, 50]
        save_samples = logger is not None and epoch is not None and (epoch in milestones or epoch == logger.cfg.training.epochs)
        
        for batch in tqdm(dataloader, desc="Evaluating"):
            noisy = batch["NoisyLR"].to(self.device)
            gt = batch["GT"].to(self.device)
            
            # Assuming model outputs restored image directly
            preds = model(noisy)
            
            # Ensure preds are clamped to [0, 1] range as expected by metrics
            preds = torch.clamp(preds, 0.0, 1.0)
            
            if criterion:
                loss = criterion(preds, gt)
                total_metrics["val_loss"] += loss.item() * noisy.size(0)
            
            batch_metrics = compute_metrics(self.metrics, preds, gt)
            
            for k in self.metrics:
                total_metrics[k] += batch_metrics[k] * noisy.size(0)
                
            # Save qualitative samples (only from first batch)
            if save_samples and num_samples == 0:
                for i in range(min(4, noisy.size(0))):
                    n_np = noisy[i].cpu().numpy().transpose(1, 2, 0).squeeze()
                    p_np = preds[i].cpu().numpy().transpose(1, 2, 0).squeeze()
                    g_np = gt[i].cpu().numpy().transpose(1, 2, 0).squeeze()
                    logger.save_sample(epoch, i, n_np, p_np, g_np)
                
            num_samples += noisy.size(0)
            
        end_time = time.time()
        
        # Calculate averages
        avg_metrics = {k: v / num_samples for k, v in total_metrics.items()}
        
        # Calculate throughput
        total_time = end_time - start_time
        fps = num_samples / total_time if total_time > 0 else 0
        
        avg_metrics["throughput_fps"] = fps
        
        # Reset torchmetrics states
        for m in self.metrics.values():
            m.reset()
            
        return avg_metrics
