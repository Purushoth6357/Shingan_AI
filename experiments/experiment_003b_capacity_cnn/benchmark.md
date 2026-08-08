# experiment_003b_capacity_cnn Benchmark Report
        
| Metric | Value |
|--------|-------|
| **Date** | 2026-08-06 17:02:53 |
| **Git Commit** | 04b3aec75f43574e085bf3ccbaa5f8dd40ad184f |
| **Dataset** | /content/drive/MyDrive/Shingan_AI/train |
| **Architecture** | cnn |
| **Loss** | Charbonnier |
| **Optimizer** | adamw |
| **Epochs** | 50 |
| **Batch Size** | 16 |
| **Learning Rate** | 0.0001 |
| **Parameters** | 554,497 |
| **FLOPs (G)** | 9.11 |
| **GPU** | Tesla T4 |
| **Peak GPU Memory (MB)** | 1834.36 |
| **Training Time** | 0.72 hours |
| **Avg Epoch Time (s)** | 48.70 |
| **Inference FPS** | 132.60 |
| **Best PSNR** | 28.5500 |
| **Best SSIM** | 0.8079 |
| **Best GMC** | 0.8702 |
| **Final PSNR** | 28.5500 |
| **Final SSIM** | 0.8079 |
| **Final GMC** | 0.8702 |

## Experiment 001 (Baseline) vs Experiment 002

| Metric | Exp001 | Exp002 (experiment_003b_capacity_cnn) | Δ |
|--------|-------:|-------:|--:|
| **PSNR** | 26.42 | 28.5500 | +2.1300 |
| **SSIM** | 0.656 | 0.8079 | +0.1519 |
| **GMC** | N/A | 0.8702 | N/A |
| **FPS** | 24 | 133 | +109 |
| **Params** | ~150K | 554,497 | N/A |

## Observations
*(Add notes here after reviewing the run)*

## Known Limitations
*(Add notes here after reviewing the run)*

## Next Experiment
*(Outline what to test in the next iteration)*
