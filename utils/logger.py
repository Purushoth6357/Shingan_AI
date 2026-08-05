import os
import json
import csv
import matplotlib.pyplot as plt
import numpy as np
import cv2
import subprocess

class ExperimentLogger:
    def __init__(self, cfg):
        self.cfg = cfg
        self.exp_name = getattr(cfg.tracking, 'experiment_name', 'experiment_001')
        self.base_dir = getattr(cfg.tracking, 'save_dir', 'experiments')
        self.exp_dir = os.path.join(self.base_dir, self.exp_name)
        
        # Subdirectories
        self.ckpt_dir = os.path.join(self.exp_dir, "checkpoints")
        self.plots_dir = os.path.join(self.exp_dir, "plots")
        self.samples_dir = os.path.join(self.exp_dir, "samples")
        self.logs_dir = os.path.join(self.exp_dir, "logs")
        self.config_dir = os.path.join(self.exp_dir, "config")
        
        for d in [self.ckpt_dir, self.plots_dir, self.samples_dir, self.logs_dir, self.config_dir]:
            os.makedirs(d, exist_ok=True)
            
        self.metrics_path = os.path.join(self.logs_dir, "metrics.csv")
        
        # Initialize CSV
        if not os.path.exists(self.metrics_path):
            with open(self.metrics_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_loss", "val_loss", "psnr", "ssim", "lr", "epoch_time", "gpu_memory_MB"])
                
        self.history = {"train_loss": [], "val_loss": [], "psnr": [], "ssim": [], "epochs": []}
        
        # Save Git Commit
        self._save_git_commit()
        
    def _save_git_commit(self):
        try:
            commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        except Exception:
            commit_hash = "unknown"
        with open(os.path.join(self.exp_dir, "git_commit.txt"), "w") as f:
            f.write(commit_hash)
        self.commit_hash = commit_hash

    def log_epoch(self, epoch, train_loss, val_loss, psnr, ssim, lr, epoch_time, gpu_mem):
        with open(self.metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, psnr, ssim, lr, epoch_time, gpu_mem])
            
        self.history["epochs"].append(epoch)
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["psnr"].append(psnr)
        self.history["ssim"].append(ssim)
        
    def generate_plots(self):
        epochs = self.history["epochs"]
        if not epochs: return
        
        # Loss Plot
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.history["train_loss"], label="Train Loss")
        plt.plot(epochs, self.history["val_loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(self.plots_dir, "loss_curve.png"))
        plt.close()
        
        # PSNR Plot
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.history["psnr"], label="Validation PSNR", color="green")
        plt.xlabel("Epoch")
        plt.ylabel("PSNR (dB)")
        plt.title("Validation PSNR")
        plt.legend()
        plt.savefig(os.path.join(self.plots_dir, "psnr_curve.png"))
        plt.close()

        # SSIM Plot
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.history["ssim"], label="Validation SSIM", color="orange")
        plt.xlabel("Epoch")
        plt.ylabel("SSIM")
        plt.title("Validation SSIM")
        plt.legend()
        plt.savefig(os.path.join(self.plots_dir, "ssim_curve.png"))
        plt.close()
        
    def save_sample(self, epoch, idx, noisy, pred, gt):
        """
        Saves qualitative samples as both .npy and visual .png
        noisy, pred, gt should be numpy arrays of shape (H, W) or (H, W, C)
        """
        prefix = os.path.join(self.samples_dir, f"epoch_{epoch}_sample_{idx}")
        
        # Save .npy
        np.save(f"{prefix}_noisy.npy", noisy)
        np.save(f"{prefix}_pred.npy", pred)
        np.save(f"{prefix}_gt.npy", gt)
        
        # Save .png (normalize for visualization)
        def to_img(arr):
            arr = np.clip(arr, 0, 1)
            return (arr * 255.0).astype(np.uint8)
            
        n_img = to_img(noisy)
        p_img = to_img(pred)
        g_img = to_img(gt)
        
        # Resize noisy (LR) for visual comparison
        h, w = g_img.shape[:2]
        n_img_resized = cv2.resize(n_img, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Compute Error Heatmap
        diff = np.abs(p_img.astype(np.float32) - g_img.astype(np.float32))
        diff_norm = np.clip(diff / 255.0, 0, 1)
        heatmap = cv2.applyColorMap((diff_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Ensure all are 3 channel for stacking
        if len(n_img_resized.shape) == 2: n_img_resized = cv2.cvtColor(n_img_resized, cv2.COLOR_GRAY2BGR)
        if len(p_img.shape) == 2: p_img = cv2.cvtColor(p_img, cv2.COLOR_GRAY2BGR)
        if len(g_img.shape) == 2: g_img = cv2.cvtColor(g_img, cv2.COLOR_GRAY2BGR)
        
        # Add labels
        def add_label(img, text):
            return cv2.putText(img.copy(), text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        grid = np.hstack((
            add_label(n_img_resized, "Input (Bicubic)"), 
            add_label(p_img, "Prediction"), 
            add_label(g_img, "Ground Truth"), 
            add_label(heatmap, "Error Heatmap")
        ))
        
        cv2.imwrite(f"{prefix}_visualization.png", grid)

    def generate_benchmark_report(self, run_stats):
        md_content = f"""# {self.exp_name} Benchmark Report
        
| Metric | Value |
|--------|-------|
| **Date** | {run_stats.get('date', 'N/A')} |
| **Git Commit** | {self.commit_hash} |
| **Dataset** | {self.cfg.data.train_dir} |
| **Architecture** | {self.cfg.model.architecture} |
| **Loss** | {run_stats.get('loss_fn', 'Charbonnier')} |
| **Optimizer** | {self.cfg.training.optimizer} |
| **Epochs** | {self.cfg.training.epochs} |
| **Batch Size** | {self.cfg.data.batch_size} |
| **Learning Rate** | {self.cfg.training.learning_rate} |
| **Parameters** | {run_stats.get('params', 'N/A'):,} |
| **FLOPs (G)** | {run_stats.get('flops_g', 'N/A'):.2f} |
| **GPU** | {run_stats.get('gpu_name', 'N/A')} |
| **Training Time** | {run_stats.get('train_time', 'N/A')} |
| **Inference FPS** | {run_stats.get('inference_fps', 'N/A'):.2f} |
| **Best PSNR** | {run_stats.get('best_psnr', 'N/A'):.4f} |
| **Best SSIM** | {run_stats.get('best_ssim', 'N/A'):.4f} |
| **Final PSNR** | {run_stats.get('final_psnr', 'N/A'):.4f} |
| **Final SSIM** | {run_stats.get('final_ssim', 'N/A'):.4f} |

## Observations
*(Add notes here after reviewing the run)*

## Known Limitations
*(Add notes here after reviewing the run)*

## Next Experiment
*(Outline what to test in the next iteration)*
"""
        with open(os.path.join(self.exp_dir, "benchmark.md"), "w") as f:
            f.write(md_content)
