# Shingan AI - Ablation Study

This table tracks the performance progression across experiments to mathematically prove the value of each architectural component and loss function.

| Exp | Model | Loss | Params | FLOPs | PSNR | SSIM | GMC | FPS | GPU Mem (MB) | Epoch Time (s) |
|-----|-------|------|--------|-------|------|------|-----|-----|--------------|----------------|
| **001** | Baseline CNN (4 blocks) | Charbonnier | ~0.48 M | 7.90 G | 26.42 | 0.656 | - | 24 | - | - |
| **002** | Baseline CNN (4 blocks) | Charbonnier + Sobel | ~0.48 M | 7.90 G | 28.50 | 0.787 | *TBD* | 24 | - | - |
| **003** | Baseline CNN (4 blocks) | Charb + Sobel + FFL | ~0.48 M | 7.90 G | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **003b**| Capacity CNN (5 blocks) | Charb + Sobel + FFL | ~0.55 M | 9.11 G | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **004** | Hybrid CNN+Transformer | Charb + Sobel + FFL | ~0.52 M | 8.54 G | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |

## Key Comparisons

1. **Exp 001 vs Exp 002**: Proves the value of Edge-Aware Supervision (Sobel Loss).
2. **Exp 002 vs Exp 003**: Proves the value of Global Frequency Supervision (Focal Frequency Loss).
3. **Exp 003 vs Exp 003b**: Isolates the effect of raw parameter scaling (Depth vs Width).
4. **Exp 003b vs Exp 004**: Proves the architectural superiority of Self-Attention (Transformer) over pure CNN depth, given equal or lesser capacity.
