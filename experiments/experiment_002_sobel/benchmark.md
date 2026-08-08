# experiment_002_sobel Benchmark Report
        
| Metric | Value |
|--------|-------|
| **Date** | 2026-08-06 13:49:23 |
| **Git Commit** | d7aa9183ad009e3ebeb8f44b36a626f2ffb67a70 |
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
| **Training Time** | 0.97 hours |
| **Inference FPS** | 153.87 |
| **Best PSNR** | 28.5041 |
| **Best SSIM** | 0.7873 |
| **Final PSNR** | 28.5041 |
| **Final SSIM** | 0.7873 |

## Experiment 001 (Baseline) vs Experiment 002

| Metric | Exp001 | Exp002 (experiment_002_sobel) | Δ |
|--------|-------:|-------:|--:|
| **PSNR** | 26.42 | 28.5041 | +2.0841 |
| **SSIM** | 0.656 | 0.7873 | +0.1313 |
| **FPS** | 24 | 154 | +130 |
| **Params** | ~150K | 480,769 | N/A |

## Observations
*(Add notes here after reviewing the run)*

## Known Limitations
*(Add notes here after reviewing the run)*

## Next Experiment
*(Outline what to test in the next iteration)*
