# experiment_003_smoke Benchmark Report
        
| Metric | Value |
|--------|-------|
| **Date** | 2026-08-06 15:45:15 |
| **Git Commit** | 963c460a76e99108ab3db91c2fd42fa0385ae149 |
| **Dataset** | /content/drive/MyDrive/Shingan_AI/train |
| **Architecture** | cnn |
| **Loss** | Charbonnier |
| **Optimizer** | adamw |
| **Epochs** | 2 |
| **Batch Size** | 16 |
| **Learning Rate** | 0.0001 |
| **Parameters** | 480,769 |
| **FLOPs (G)** | 7.90 |
| **GPU** | Tesla T4 |
| **Training Time** | 0.00 hours |
| **Inference FPS** | 71.03 |
| **Best PSNR** | 24.1285 |
| **Best SSIM** | 0.5206 |
| **Final PSNR** | 24.1285 |
| **Final SSIM** | 0.5206 |

## Experiment 001 (Baseline) vs Experiment 002

| Metric | Exp001 | Exp002 (experiment_003_smoke) | Δ |
|--------|-------:|-------:|--:|
| **PSNR** | 26.42 | 24.1285 | -2.2915 |
| **SSIM** | 0.656 | 0.5206 | -0.1354 |
| **FPS** | 24 | 71 | +47 |
| **Params** | ~150K | 480,769 | N/A |

## Observations
*(Add notes here after reviewing the run)*

## Known Limitations
*(Add notes here after reviewing the run)*

## Next Experiment
*(Outline what to test in the next iteration)*
