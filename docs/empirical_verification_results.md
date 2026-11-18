# Controlled Empirical Mechanism Verification Results

This document summarizes the reproducible empirical mechanism verification experiments executed directly on the `host-forensics` reference implementation via [`scripts/run_poc_experiments.py`](../scripts/run_poc_experiments.py), with raw numerical outputs stored in [`paper/figures/poc_results.json`](../paper/figures/poc_results.json).

To reproduce all experiments from scratch (`~8 seconds` on CPU):

```bash
.venv/bin/python scripts/run_poc_experiments.py
```

---

## 1. Architectural Insights Uncovered During Verification

Empirical testing of `SelfReferentialHostConditioning` ([`src/host_forensics/models/shdc.py`](../src/host_forensics/models/shdc.py)) revealed two subtle mathematical bottlenecks in naive differential conditioning designs:

1. **Norm-Erasure from Per-Token `LayerNorm`:** Applying standard per-token `nn.LayerNorm(D)` at the output of a differential block divides each token $\mathbf{z}_m$ by its own feature standard deviation across channels, erasing the Mahalanobis Z-score magnitude difference between a $1\sigma$ authentic host patch and a $6\sigma$ manipulated foreground patch. Replacing per-token `LayerNorm` with a learnable channel gain $\boldsymbol{\omega} \in \mathbb{R}^D$ and smooth `10 * tanh(x / 10)` bounding preserves full anomaly amplitude.
2. **Zero-Sum Saddle Point of Purely Linear Mean-Centered Residuals:** Under uniform initial pooling weights, a linear projection of mean-centered tokens satisfies $\sum_{m=1}^M \mathbf{W}(\mathbf{z}_m - \bar{\mathbf{z}}) \equiv \mathbf{0}$ identically for both pristine and manipulated clips, stalling early gradient flow. Introducing **50%-breakdown-point coordinate-wise median gating** paired with a **two-layer `GELU` non-linear Z-residual MLP ($\psi_{\text{MLP}}$)** breaks zero-sum cancellation while retaining exact invariance to per-channel positive affine shifts ($Z'_{m,d} = a_d Z_{m,d} + c_d$, Proposition 1).

---

## 2. Experiment 1: Cross-Domain Score De-Polarization (`SHDC` + `PristineQuantileCalibrator`)

- **Setup:** Clips contain an `18.75%` localized manipulation footprint (`24 / 128` spatio-temporal tokens) trained on mild single-domain variation and evaluated zero-shot across **three unseen codec/restoration shift domains** under a strict $\delta = 5\%$ validation false-alarm budget:
  - *Domain 0 (Clean Studio):* $a_d \in [0.95, 1.05],\; c_d \in [-0.15, +0.15]$
  - *Domain 1 (Heavy Social Transcoding):* $a_d \in [0.45, 0.65],\; c_d \in [-2.80, -1.60]$
  - *Domain 2 (High-Contrast Neural Restoration):* $a_d \in [1.70, 2.30],\; c_d \in [+1.80, +3.20]$

| Conditioning & Calibration Method | Within-Domain AUROC (%) $\uparrow$ | Pooled AUROC (%) $\uparrow$ | Polarization $W_1$ $\downarrow$ | Cross-AUC (%) $\uparrow$ | Realized FAR (%) $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Unconditioned Attention Soft-Pool | `96.88` | `98.40` | `0.2734` | `70.39` | `30.67%` |
| **SHDC + Pristine Quantile Calibrator (Ours)** | **`97.40`** | `97.41` | **`0.0425`** (`6.4x` lower) | **`93.26`** (`+22.87`) | **`0.67%`** ($\le 5\%$ target) |

- **Takeaway:** Although unconditioned soft-pooling achieves `96.88%` Within-Domain AUROC, global affine shifts polarize its raw logits across domains ($W_1 = 0.2734$), collapsing `Cross-AUC` to `70.39%` and inflating false alarms to `30.67%` under a fixed validation threshold. `SHDC` cancels the cross-domain shift, boosting `Cross-AUC` to **`93.26%`** while holding `Realized FAR` to **`0.67%`**.

---

## 3. Experiment 2: Auxiliary-Loss-Free MoE Expert Load Convergence (`BiasBalancedTopKRouter`)

- **Setup:** A 4-expert, top-$2$ router ([`BiasBalancedTopKRouter`](../src/host_forensics/models/tri_stmoe.py)) initialized with strongly skewed expert preferences (`[+2.2, +1.6, -1.6, -2.2]`), comparing disabled bias updates ($\gamma = 0$) against non-differentiable load-balancing bias updates ($\gamma = 0.05$) over 80 optimization steps with **zero auxiliary loss terms**.

| Optimization Step | Unbalanced ($\gamma = 0$) Norm. Entropy $\mathcal{H}/\log E \uparrow$ | Unbalanced ($\gamma = 0$) Max–Min Load Gap $\downarrow$ | Bias-Balanced ($\gamma = 0.05$) Norm. Entropy $\mathcal{H}/\log E \uparrow$ | Bias-Balanced ($\gamma = 0.05$) Max–Min Load Gap $\downarrow$ |
| :---: | :---: | :---: | :---: | :---: |
| `1` | `0.7852` | `0.3672` | `0.7852` | `0.3672` |
| `10` | `0.8438` | `0.3242` | `0.9084` | `0.2598` |
| `20` | `0.8348` | `0.3398` | `0.9521` | `0.2051` |
| `30` | `0.8265` | `0.3340` | `0.9765` | `0.1523` |
| `40` | `0.8581` | `0.3105` | `0.9969` | `0.0547` |
| `50` | `0.8216` | `0.3496` | **`0.9996`** | **`0.0234`** |
| `60` | `0.8417` | `0.3242` | `0.9994` | `0.0254` |
| `80` | `0.8481` | `0.3164` | `0.9949` | `0.0684` |

- **Takeaway:** Without bias updates, two experts remain chronically starved (`0.3164` load gap at step 80). Auxiliary-loss-free bias updates reach **`0.9996` normalized load entropy** (`0.0234` max–min load gap) by step 50 without injecting competing gradients into the forensic objective.

---

## 4. Experiment 3: Color-Matched Object-Removal Micro-Texture Sensitivity (`MN-Stem`)

- **Setup:** Synthetic camera-captured frames containing authentic low-level sensor PRNU/ISP micro-texture where a localized foreground object mask is erased via spatial smoothing/inpainting whose RGB mean and low-frequency gradients match the surrounding background—mirroring the *ProPainter* / *VideoPainter* object-removal regime where semantic MLLMs score `0.135 mIoU`.

| Patch Anomaly Representation | Patch-Level Localization AUROC (%) $\uparrow$ |
| :--- | :---: |
| Raw RGB Semantic / Color Patch Deviation | `45.73%` (near chance) |
| **MN-Stem Zero-DC Bayar + SRM Residual Energy (Ours)** | **`96.09%` (`+50.36` points)** |

- **Takeaway:** Because erased objects leave behind color-matched background textures, raw RGB patch statistics cannot distinguish the inpainted region (`45.73%` AUROC). Constrained zero-DC `BayarConv2d` + `SRMFilterBank` residuals immediately isolate the suppressed camera noiseprint (`96.09%` AUROC).
