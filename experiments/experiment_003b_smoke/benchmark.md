# experiment_003b_smoke Benchmark Report
        
| Metric | Value |
|--------|-------|
| **Date** | 2026-08-06 16:58:49 |
| **Git Commit** | 04b3aec75f43574e085bf3ccbaa5f8dd40ad184f |
| **Dataset** | /content/drive/MyDrive/Shingan_AI/train |
| **Architecture** | cnn |
| **Loss** | Charbonnier |
| **Optimizer** | adamw |
| **Epochs** | 2 |
| **Batch Size** | 16 |
| **Learning Rate** | 0.0001 |
| **Parameters** | 554,497 |
| **FLOPs (G)** | 9.11 |
| **GPU** | Tesla T4 |
| **Peak GPU Memory (MB)** | 1834.68 |
| **Training Time** | 0.02 hours |
| **Avg Epoch Time (s)** | 19.02 |
| **Inference FPS** | 78.49 |
| **Best PSNR** | 24.5280 |
| **Best SSIM** | 0.5808 |
| **Best GMC** | 0.7117 |
| **Final PSNR** | 24.5280 |
| **Final SSIM** | 0.5808 |
| **Final GMC** | 0.7117 |

## Experiment 001 (Baseline) vs Experiment 002

| Metric | Exp001 | Exp002 (experiment_003b_smoke) | Δ |
|--------|-------:|-------:|--:|
| **PSNR** | 26.42 | 24.5280 | -1.8920 |
| **SSIM** | 0.656 | 0.5808 | -0.0752 |
| **GMC** | N/A | 0.7117 | N/A |
| **FPS** | 24 | 78 | +54 |
| **Params** | ~150K | 554,497 | N/A |

## Observations
*(Add notes here after reviewing the run)*

## Known Limitations
*(Add notes here after reviewing the run)*

## Next Experiment
*(Outline what to test in the next iteration)*
