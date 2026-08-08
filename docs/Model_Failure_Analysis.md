# Model Failure Analysis Notebook

This notebook tracks qualitative observations for different architectures during the Shingan AI model selection phase. We compare the predictions of different models against the Ground Truth, focusing specifically on edge preservation, high-frequency reconstruction, and artifacts.

## Observation Template

```markdown
### [Model Name / Experiment]

**Observations:**
- [Feature 1]
- [Feature 2]

**Performance vs Ground Truth:**
Input ↓ Prediction ↓ Ground Truth
(Insert qualitative notes here)
```

---

## Experiment 004 (Hybrid CNN + Transformer Baseline)

**Observations:**
- Good denoising
- Edges preserved
- Book text slightly blurry
- Fine textures (e.g., nanometer-scale metal lines, vias) are slightly oversmoothed

**Performance vs Ground Truth:**
Input ➔ Prediction ➔ Ground Truth
While the model removes noise and restores dominant edges well, **very fine high-frequency details** (like text on book spines or tiny semiconductor defects) are slightly oversmoothed compared to the ground truth.

---

## Experiment 005 (SwinIR Baseline)

*Awaiting training and inference on Colab.*

**Observations to look for:**
- Does it preserve Book text?
- Are Metal edges sharper?
- Are Thin structures (vias, lines) recovered better than Hybrid?
- How is the Noise removal?

**Performance vs Ground Truth:**
Input ➔ Prediction ➔ Ground Truth
(Add notes here after reviewing Exp005 inference results)

---

## Experiment 005 Phase B (Hybrid V2 - Edge Aware)

*Awaiting future development.*

**Observations to look for:**
- Did the Edge-Guided Attention or Multi-scale fusion reduce the oversmoothing of fine textures?

**Performance vs Ground Truth:**
Input ➔ Prediction ➔ Ground Truth
(Add notes here after reviewing Phase B inference results)
