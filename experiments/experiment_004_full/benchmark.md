# experiment_004_full Benchmark Report
        
| Metric | Value |
|--------|-------|
| **Date** | 2026-08-07 14:20:22 |
| **Git Commit** | 27f00c28d96bec4bb7c0178c5109800e9b094926 |
| **Dataset** | /content/drive/MyDrive/Shingan_AI/train |
| **Architecture** | HybridCNNTransformer |
| **Loss** | Charbonnier |
| **Optimizer** | adamw |
| **Epochs** | 50 |
| **Batch Size** | 16 |
| **Learning Rate** | 0.0001 |
| **Parameters** | 562,484 |
| **FLOPs (G)** | 9.19 |
| **GPU** | Tesla T4 |
| **Peak GPU Memory (MB)** | 5625.58 |
| **Training Time** | 1.51 hours |
| **Avg Epoch Time (s)** | 102.95 |
| **Inference FPS** | 61.12 |
| **Best PSNR** | 28.5226 |
| **Best SSIM** | 0.8047 |
| **Best GMC** | 0.8692 |
| **Final PSNR** | 28.5226 |
| **Final SSIM** | 0.8047 |
| **Final GMC** | 0.8692 |

## Experiment 001 (Baseline) vs Experiment 002

| Metric | Exp001 | Exp002 (experiment_004_full) | Δ |
|--------|-------:|-------:|--:|
| **PSNR** | 26.42 | 28.5226 | +2.1026 |
| **SSIM** | 0.656 | 0.8047 | +0.1487 |
| **GMC** | N/A | 0.8692 | N/A |
| **FPS** | 24 | 61 | +37 |
| **Params** | ~150K | 562,484 | N/A |

## Observations
*(Add notes here after reviewing the run)*

## Known Limitations
*(Add notes here after reviewing the run)*

## Next Experiment
*(Outline what to test in the next iteration)*
