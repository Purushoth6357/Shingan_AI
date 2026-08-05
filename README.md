# Shingan AI – AI-Based Restoration of Degraded Images for Semiconductor Inspection

## Project Vision & Goal
Build a robust, reproducible image restoration system that restores degraded images for semiconductor inspection while generalizing to unseen images. Our focus is on restoration accuracy, generalization, throughput, and reproducibility.

## Hackathon Rules
- **Training dataset**: `GT/` (Ground Truth) and `NoisyLR/` (Degraded images).
- **Evaluation**: Unseen images, scored on combination of metrics, generalization, throughput, reproducibility.
- **Inference**: Single `infer.py` script that reads every image in an input folder and writes restored images to an output folder.

## Architecture
**Approved Architecture**: `Input -> Degradation Prompt Encoder -> Hybrid CNN + Transformer -> Restored Image`
- *Note*: Edge preservation is handled via Charbonnier + Sobel Edge Loss + Focal Frequency Loss. A standalone edge refinement module is explicitly omitted for performance reasons.

## Design Rationale (Novelty Framing)
The core defensible claim is **domain-grounded engineering**, not necessarily a purely novel architecture. Every design choice (augmentation exclusions, normalization strategy, loss weighting) is justified specifically for semiconductor/wafer/SEM imagery.
- **Augmentations**: No resizing/downsampling (to preserve pixel-to-physical mapping), no JPEG compression, no motion blur.
- **Normalization**: Min-max [0, 1] scaling instead of ImageNet stats.
- **Loss**: Perceptual/LPIPS loss is avoided during training, as VGG features hallucinate natural-image textures onto industrial images.

## Development Roadmap
- **Module 1**: Scaffolding & Configs
- **Module 2**: Evaluation System
- **Module 3**: Data Pipeline
- **Module 4**: Baseline CNN
- **Module 5**: Training Loop
- **Module 6**: Inference Engine
- **Module 7**: Hybrid Architecture
- **Module 8**: Final Validation

## Principles
- CLI overrides for paths only.
- PyTorch (No Lightning).
- Device fallback logic automatically detects CUDA vs CPU.
- Pinned requirements for strict reproducibility.
