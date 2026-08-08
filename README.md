# Shingan AI – AI-Based Restoration of Degraded Images for Semiconductor Inspection

## Project Vision & Goal
Build a robust, reproducible image restoration system that restores degraded images for semiconductor inspection while generalizing to unseen images. Our focus is on restoration accuracy, generalization, throughput, and reproducibility.

## Hackathon Rules
- **Training dataset**: `GT/` (Ground Truth) and `NoisyLR/` (Degraded images).
- **Evaluation**: Unseen images, scored on combination of metrics, generalization, throughput, reproducibility.
- **Inference**: Single `infer.py` script that reads every image in an input folder and writes restored images to an output folder.

## Official Benchmark (Frozen Baseline)
Experiment 002 (Baseline CNN + PixelShuffle ×2 + Hybrid Loss [Charbonnier + Sobel]) is our official locked baseline for future comparisons.
- **PSNR**: 28.5041 dB
- **SSIM**: 0.7873
- **Validation Loss**: 0.04263

All future experiments (Focal Frequency Loss, Hybrid architectures, Prompt Encoders) must beat these metrics.

## Architecture
**Approved Architecture**: `Input -> Image Quality Analyzer -> Degradation Prompt Encoder -> Hybrid CNN + Transformer -> Feature Fusion -> Restored Image`
- *Note*: Edge preservation is handled via Charbonnier + Sobel Edge Loss + Focal Frequency Loss. A standalone edge refinement module is explicitly omitted for performance reasons.

## Design Rationale (Novelty Framing)
The core defensible claim is **domain-grounded engineering**, not necessarily a purely novel architecture. Every design choice (augmentation exclusions, normalization strategy, loss weighting) is justified specifically for semiconductor/wafer/SEM imagery.
- **Augmentations**: No resizing/downsampling (to preserve pixel-to-physical mapping), no JPEG compression, no motion blur.
- **Normalization**: Min-max [0, 1] scaling instead of ImageNet stats.
- **Loss**: Perceptual/LPIPS loss is avoided during training (to prevent VGG hallucination of natural-image textures), but **LPIPS is strictly monitored as a mandatory evaluation metric** to ensure perceptual quality aligns with KLA session feedback.

## Development Roadmap
- **Experiment 001**: Initial scaffolding and naive baseline. *[COMPLETED]*
- **Experiment 002**: Official baseline (Sobel Edge Loss). *[COMPLETED]*
- **Experiment 003**: Focal Frequency Loss integration. *[COMPLETED]*
- **Experiment 003b**: Capacity-Matched CNN baseline. *[COMPLETED]*
- **Experiment 004**: Hybrid CNN + Transformer Integration. *[COMPLETED]*
- **Experiment 005**: Degradation Prompt Encoder and Image Quality Analyzer.
- **Module 8**: Final Validation & Hackathon Submission.

## Principles
- CLI overrides for paths only.
- PyTorch (No Lightning).
- Device fallback logic automatically detects CUDA vs CPU.
- Pinned requirements for strict reproducibility.

## Experiment 005

Experiment 005 integrates the official SwinIR architecture (MIT License) into our training and evaluation framework. This serves as a benchmark against our Hybrid model.
