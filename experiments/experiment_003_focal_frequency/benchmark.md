# experiment_003_focal_frequency Benchmark Report
        
| Metric | Value |
|--------|-------|
| **Date** | 2026-08-06 15:55:51 |
| **Git Commit** | 963c460a76e99108ab3db91c2fd42fa0385ae149 |
| **Dataset** | /content/drive/MyDrive/Shingan_AI/train |
| **Architecture** | cnn |
| **Loss** | Charbonnier |
| **Optimizer** | adamw |
| **Epochs** | 50 |
| **Batch Size** | 16 |
| **Learning Rate** | 0.0001 |
| **Parameters** | 480,769 |
| **FLOPs (G)** | 7.90 |
| **GPU** | Tesla T4 |
| **Training Time** | 0.51 hours |
| **Inference FPS** | 115.32 |
| **Best PSNR** | 28.4860 |
| **Best SSIM** | 0.8056 |
| **Final PSNR** | 28.4708 |
| **Final SSIM** | 0.8048 |

## Experiment 001 (Baseline) vs Experiment 002

| Metric | Exp001 | Exp002 (experiment_003_focal_frequency) | Δ |
|--------|-------:|-------:|--:|
| **PSNR** | 26.42 | 28.4708 | +2.0508 |
| **SSIM** | 0.656 | 0.8048 | +0.1488 |
| **FPS** | 24 | 115 | +91 |
| **Params** | ~150K | 480,769 | N/A |

## Observations
*(Add notes here after reviewing the run)*

## Known Limitations
*(Add notes here after reviewing the run)*

## Next Experiment
*(Outline what to test in the next iteration)*
