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
            
        # Initialize CSV
        self.metrics_path = os.path.join(self.logs_dir, "metrics.csv")
        self.csv_headers = ["epoch", "train_loss", "val_loss", "psnr", "ssim", "lr", "epoch_time", "gpu_memory_MB"]
        self.history = {"train_loss": [], "val_loss": [], "psnr": [], "ssim": [], "epochs": []}
        self.has_written_header = False
        
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

    def log_epoch(self, epoch, train_loss, val_loss, psnr, ssim, lr, epoch_time, gpu_mem, comp_losses=None):
        if not self.has_written_header:
            if comp_losses:
                for k in comp_losses.keys():
                    self.csv_headers.append(k)
            with open(self.metrics_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.csv_headers)
            self.has_written_header = True
            
        row = [epoch, train_loss, val_loss, psnr, ssim, lr, epoch_time, gpu_mem]
        if comp_losses:
            for k in self.csv_headers[8:]:
                row.append(comp_losses.get(k, 0.0))
                
        with open(self.metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
            
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
        
        # Compute Edge Difference (using OpenCV Sobel for visualization)
        p_gray = p_img if len(p_img.shape) == 2 else cv2.cvtColor(p_img, cv2.COLOR_BGR2GRAY)
        g_gray = g_img if len(g_img.shape) == 2 else cv2.cvtColor(g_img, cv2.COLOR_BGR2GRAY)
        
        # Calculate Sobel X and Y for both
        sobel_p_x = cv2.Sobel(p_gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_p_y = cv2.Sobel(p_gray, cv2.CV_32F, 0, 1, ksize=3)
        sobel_g_x = cv2.Sobel(g_gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_g_y = cv2.Sobel(g_gray, cv2.CV_32F, 0, 1, ksize=3)
        
        # Calculate magnitude of differences
        edge_diff = np.abs(sobel_p_x - sobel_g_x) + np.abs(sobel_p_y - sobel_g_y)
        edge_diff_norm = np.clip(edge_diff / (np.max(edge_diff) + 1e-5), 0, 1)
        edge_diff_viz = cv2.applyColorMap((edge_diff_norm * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
        
        # Ensure all are 3 channel for stacking
        if len(n_img_resized.shape) == 2: n_img_resized = cv2.cvtColor(n_img_resized, cv2.COLOR_GRAY2BGR)
        if len(p_img.shape) == 2: p_img = cv2.cvtColor(p_img, cv2.COLOR_GRAY2BGR)
        if len(g_img.shape) == 2: g_img = cv2.cvtColor(g_img, cv2.COLOR_GRAY2BGR)
        
        # Add labels
        def add_label(img, text):
            return cv2.putText(img.copy(), text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
        grid = np.hstack((
            add_label(n_img_resized, "Input"), 
            add_label(p_img, "Prediction"), 
            add_label(g_img, "Ground Truth"), 
            add_label(heatmap, "Error Heatmap"),
            add_label(edge_diff_viz, "Edge Diff")
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
| **Peak GPU Memory (MB)** | {run_stats.get('peak_gpu_mem_mb', 0):.2f} |
| **Training Time** | {run_stats.get('train_time', 'N/A')} |
| **Avg Epoch Time (s)** | {run_stats.get('avg_epoch_time', 0):.2f} |
| **Inference FPS** | {run_stats.get('inference_fps', 'N/A'):.2f} |
| **Best PSNR** | {run_stats.get('best_psnr', 'N/A'):.4f} |
| **Best SSIM** | {run_stats.get('best_ssim', 'N/A'):.4f} |
| **Best GMC** | {run_stats.get('best_gmc', 'N/A'):.4f} |
| **Final PSNR** | {run_stats.get('final_psnr', 'N/A'):.4f} |
| **Final SSIM** | {run_stats.get('final_ssim', 'N/A'):.4f} |
| **Final GMC** | {run_stats.get('final_gmc', 'N/A'):.4f} |

## Experiment 001 (Baseline) vs Experiment 002

| Metric | Exp001 | Exp002 ({self.exp_name}) | Δ |
|--------|-------:|-------:|--:|
| **PSNR** | 26.42 | {run_stats.get('final_psnr', 0):.4f} | {run_stats.get('final_psnr', 0) - 26.42:+.4f} |
| **SSIM** | 0.656 | {run_stats.get('final_ssim', 0):.4f} | {run_stats.get('final_ssim', 0) - 0.656:+.4f} |
| **GMC** | N/A | {run_stats.get('final_gmc', 0):.4f} | N/A |
| **FPS** | 24 | {run_stats.get('inference_fps', 0):.0f} | {run_stats.get('inference_fps', 0) - 24:+.0f} |
| **Params** | ~150K | {run_stats.get('params', 0):,} | N/A |

## Observations
*(Add notes here after reviewing the run)*

## Known Limitations
*(Add notes here after reviewing the run)*

## Next Experiment
*(Outline what to test in the next iteration)*
"""
        with open(os.path.join(self.exp_dir, "benchmark.md"), "w") as f:
            f.write(md_content)
