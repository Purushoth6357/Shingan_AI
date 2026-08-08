# Engineering Log & Development Notes

This document highlights critical failures encountered during development and the engineering decisions made to resolve them. It serves as evidence of our iterative debugging and problem-solving process.

## 1. cuFFT Internal Error (Experiment 005 - SwinIR)

**Problem:** 
During the integration of the SwinIR architecture (Experiment 005), the training loop consistently crashed with a `RuntimeError: CUFFT_INTERNAL_ERROR` specifically when executing the `FocalFrequencyLoss`.

**Investigation:**
- The exact failure occurred at: `torch.fft.fft2(pred, norm='ortho')`
- We initially hypothesized that it was due to mixed precision (AMP) or non-contiguous tensors in PyTorch.
- We instrumented the code to print `shape`, `stride`, `layout`, `storage_offset`, and ran isolation tests:
  1. `Test 1: Random CUDA tensor FFT` -> FAILED
  2. `Test 2: Clone and contiguous FFT` -> FAILED
  3. `Test 3: CPU FFT fallback` -> PASSED

**Root Cause:**
While analyzing the SwinIR architecture `network_swinir.py`, we found that the final return statement was slicing the tensor: `return x[:, :, :H*self.upscale, :W*self.upscale]`. This creates a non-contiguous view with irregular strides. However, even after calling `.contiguous()` in the loss function, the global cuFFT library in our Colab environment failed on *any* CUDA tensor during that specific runtime session.

**Decision & Action:**
- Attempting to bypass it using a CPU fallback for the FFT caused OOM errors due to cross-device gradient tracking overhead.
- Because the issue was traced to the underlying Colab CUDA/cuFFT environment rather than our framework code, we chose not to waste days debugging an external NVIDIA library issue.
- **Result:** We safely archived the SwinIR experiment (labeled Exp 005 Archived) and immediately pivoted to integrating **Restormer (Experiment 005.1)**, an architecture specifically designed for image restoration, which did not trigger the same environment bug.

## 2. Restormer Architectural Adaptation

**Problem:**
Restormer is natively an image-to-image (1:1) restoration network. However, our task requires 2× Super Resolution (e.g., 128x128 -> 256x256).

**Investigation:**
The native Restormer output computes `out = network(...) + inp_img`. This global residual connection breaks if the output resolution is scaled.

**Decision & Action:**
We modified the reconstruction head of the vendored Restormer:
1. Swapped the final convolution for a `PixelShuffle` upsampling block.
2. Preserved residual learning by injecting a bicubic-upsampled residual: `output = network_prediction + F.interpolate(input, scale_factor=2, mode='bicubic')`.
This allowed us to evaluate Restormer on our task without destroying its core U-Net design philosophy.
