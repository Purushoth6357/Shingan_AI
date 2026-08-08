# Shingan AI - Image Restoration & Super Resolution

Shingan AI is an advanced deep learning framework developed for the KLA Hackathon to restore images suffering from severe speckle and Gaussian noise while simultaneously performing 2× super-resolution.

Our approach leverages a hybrid architecture combining the local feature extraction capabilities of Convolutional Neural Networks (CNNs) with the global context modeling of self-attention mechanisms, supervised by a multi-objective loss function.

## Features
- **Dynamic Configurable Pipeline:** YAML-based configuration for datasets, models, metrics, and optimizers.
- **Hybrid Loss Landscape:** Uses a weighted combination of Charbonnier, Sobel (edge-aware), and Focal Frequency loss for robust structural recovery.
- **Multiple Architectures:** Easily switch between pure CNNs, Hybrid CNN+Transformers, and Restormer backbones.
- **Production-Ready Inference:** A standalone CLI inference script capable of processing entire directories of images or `.npy` files seamlessly.

---

## Folder Structure

```
Shingan_AI/
├── configs/               # YAML configuration files for each experiment
├── datasets/              # Training/Validation/Testing data splits
├── experiments/           # Logs, checkpoints, and validation samples
├── models/                # Network architectures (CNN, Hybrid, Restormer)
├── results/               # Final inference results and qualitative comparisons
│   ├── experiment004/
│   ├── experiment0051/
│   └── comparison/
├── scripts/               # Utility scripts (infer, verify, benchmark)
├── trainer/               # PyTorch training loop and evaluation logic
├── utils/                 # Metrics (LPIPS, PSNR, SSIM, GMC), logging, config parsing
├── README.md              # Project documentation (You are here)
├── requirements.txt       # Frozen Python dependencies
├── infer.py               # Root CLI inference script
└── ablation_table.md      # Detailed experimental progression metrics
```

---

## Installation

Ensure you have Python 3.9+ installed. Clone the repository and install the frozen dependencies:

```bash
git clone https://github.com/Purushoth6357/Shingan_AI.git
cd Shingan_AI
pip install -r requirements.txt
```

Verify your environment and submission readiness by running:
```bash
python scripts/verify_submission.py
```

---

## Dataset Structure

The pipeline expects a directory of noisy images and a corresponding directory of clean ground truth images, defined via split files (e.g., `train.txt`). Images can be standard formats (`.png`, `.jpg`) or numpy arrays (`.npy`).

---

## Training

To train a model, pass the configuration file to the trainer module:

```bash
python -m trainer.train --config configs/experiment_004.yaml
```

Checkpoints and tensorboard logs are automatically saved to `experiments/<experiment_name>/`.

---

## Evaluation

Evaluation metrics are computed automatically at the end of every `validation.every_n_epochs`. The framework computes:
- **PSNR** (Peak Signal-to-Noise Ratio)
- **SSIM** (Structural Similarity Index)
- **LPIPS** (Learned Perceptual Image Patch Similarity)
- **GMC** (Global Motion Compensation - custom structural metric)

---

## Inference

To restore a folder of hidden test images, use the root `infer.py` script. The script will automatically load the model, process the folder, and save the restored images.

```bash
python infer.py \
    --config configs/experiment_004.yaml \
    --weights experiments/experiment_004/best_model.pth \
    --input datasets/test_noisy \
    --output restored_images
```

*(Note: `--checkpoint` can also be used interchangeably with `--weights`)*

---

## Model Zoo

| Model                  | Params | FLOPs | Purpose  |
| ---------------------- | -----: | ----: | -------- |
| Baseline CNN           |   480K | 7.90G | Exp001   |
| Capacity CNN           |   550K | 9.11G | Exp003b  |
| Hybrid CNN+Transformer |   562K | 9.19G | Exp004   |
| Restormer              |   601K | 1.01G | Exp005.1 |

---

## Engineering Decisions

- **Why Hybrid?** CNNs are excellent at local feature extraction (edges) but struggle with global context. Transformers capture long-range dependencies but are computationally heavy. A hybrid approach provides the best of both worlds within a 560K parameter budget.
- **Why Charbonnier Loss?** A differentiable approximation of L1 loss that is more robust to outliers and less prone to blurring than L2 (MSE) loss.
- **Why Sobel Loss?** Semiconductor inspection requires nanometer-scale structural fidelity. By computing the L1 loss on fixed Sobel gradients, we force the network to explicitly reconstruct edges.
- **Why Focal Frequency Loss (FFL)?** Some high-frequency details (e.g., fine metal lines) are lost during downsampling/noise corruption. FFL computes loss in the frequency domain, dynamically weighting hard-to-recover frequencies.
- **Why AdamW?** Provides better weight decay handling than standard Adam, preventing overfitting on small capacity networks.

*(For detailed failures and debugging notes, see `engineering_log.md`)*

---

## Results & Experiments

For a complete breakdown of our ablation studies and how we arrived at our final architecture, please read [benchmark.md](benchmark.md).

*(Final qualitative visual comparisons will be populated in the `results/` folder upon completion of Experiment 005.1)*

---

## License

This project is licensed under the MIT License. Portions of the architectural code (Restormer) are derived from the respective authors under MIT License.

---

## References
- Restormer: Efficient Transformer for High-Resolution Image Restoration (CVPR 2022)
- SwinIR: Image Restoration Using Swin Transformer (ICCV 2021)
- Focal Frequency Loss for Image Reconstruction and Synthesis (ICCV 2021)
- Learned Perceptual Image Patch Similarity (LPIPS) (CVPR 2018)
