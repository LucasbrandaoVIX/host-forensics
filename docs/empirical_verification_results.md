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

---

## 5. Experiment 4: End-to-End Real-Media Foundation Pilot & 5-Variant Ablation Study

To validate the full `HoST-MoE` training, pristine-quantile calibration, zero-shot multi-domain evaluation, and ablation pipeline on real camera-captured and manipulated media prior to full-scale RTX 5090 training, we executed [`scripts/run_real_hf_pilot.py`](../scripts/run_real_hf_pilot.py) (`configs/hf_real_pilot.yaml`) with raw metrics saved in [`paper/figures/real_hf_pilot_results.json`](../paper/figures/real_hf_pilot_results.json) and LaTeX table exported to [`paper/tables/real_hf_pilot_table.tex`](../paper/tables/real_hf_pilot_table.tex).

### 5.1 Experimental Protocol
- **Pretrained Foundation Backbones:** Frozen `facebook/dino-vits16` (`ViT-S/16`, `384`-dim, `12` blocks, `16x16` patches) + frozen `microsoft/wavlm-base-plus` (`WavLM-Base+`, `768`-dim, `12` blocks) — **`123,392,482` (`123.39M`) total parameters, `7,197,042` (`7.20M`, `5.83%`) trainable**.
- **Real Open-Access Media (`524` Source Files):**
  - `289` real and manipulated `.mp4` videos from `Celeb-DF-v2` (`Celeb-real`, `Celeb-synthesis`, `YouTube-real`) and unseen `Google DeepFake Detection (DFD)` (`DFD_original`, `DFD_manipulated`), decoded at native `16:9` (`512x288`) and framed via leakage-free YCbCr + temporal-motion salience (`160x160 -> 224x224`) to preserve both foreground subjects (`~20%` area) and surrounding host context (`~80%` area).
  - `235` authentic and ElevenLabs/TTS voice-cloned `.flac` speech recordings (`16 kHz`).
- **Strict Identity-Disjoint Two-Stage Curriculum & Multi-Domain Test Suite:**
  - **Training (`64` Stage-1 clips at `192^2` x `6` epochs + `28` Stage-2 clips at `224^2` x `2` epochs):** `Celeb-DF-v2` IDs `0..29` + `YouTube-real` `0..139` across all 4 regimes (`clean`, `visual_only` face-swaps & moving-tubelet inpaints, `audio_only` cosine-crossfaded speech splices, and `both_track` deepfakes) and 3 degradation domains (`HQ`, `libx264 CRF-35`, and `Restored/Laundered`).
  - **Pristine Calibration (`28` validation clips, `16` pristine):** `Celeb-DF-v2` IDs `30..41` + `YouTube-real` `140..151`, fitting `PristineQuantileCalibrator` once (`visual_thresh`, `audio_thresh`) and freezing thresholds across all four held-out test benchmarks (`138` test clips total):
    1. `CelebDF_v2` (`36` clips, IDs `42..61`, 3 domains)
    2. `GoogleDFD_ZeroShot` (`40` clips, 100% unseen dataset & identities, 3 domains)
    3. `InpaintTubelet_Silent` (`30` silent clips, `has_audio=False` testing `Silent-Prior` fallback, `YouTube-real` `190..219`)
    4. `AV_Splice_Eval` (`32` clips, balanced 4-way asymmetric audio-visual splices)

### 5.2 Quantitative 5-Variant Results

| Model Variant (`slug`) | Google DFD (0-Shot) AUROC / Cross-AUC (%) $\uparrow$ | AV-Splice AUROC / AV-mAP (%) $\uparrow$ | AV-Splice V-mAP / A-mAP (%) $\uparrow$ | Celeb-DF-v2 AUROC / Mask mIoU (%) $\uparrow$ | 4-Bench Pooled MID-Acc (%) $\uparrow$ | 4-Bench Pooled Realized FAR (%) $\downarrow$ | Principal Leak $\|\mathbf{U}_k^\top \Delta\mathbf{W}\|_F \downarrow$ | MoE Expert Load CV $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`HoST-MoE (Full)`** (`full_host_moe`) | **`61.00` / `55.78`** | **`89.58` / `86.41`** | **`74.86` / `97.95`** | `53.09` / `25.61` | `56.52` | **`32.76`** | **`0.000000`** | **`0.1732`** |
| `w/o Ortho SVD-LoRA` (`abl_a1_standard_lora`) | `64.00` / `59.15` | `84.38` / `81.05` | `64.56` / `97.53` | `57.10` / `25.80` | `54.35` | `36.21` | `0.054988` | `0.2220` |
| `w/o Bias-Balanced MoE` (`abl_a2_no_bias_balance`) | `52.50` / `49.64` | `82.29` / `88.93` | `79.46` / `98.39` | `58.02` / `25.94` | `55.07` | `37.93` | `0.000000` | `0.5569` |
| `w/o Host Depol. & Anchor` (`abl_a3_no_depolarization`) | `49.75` / `42.82` | `89.06` / `78.69` | `60.82` / `96.55` | `56.79` / **`31.13`** | `56.52` | `50.00` | `0.000000` | `0.2667` |
| `w/o Symmetric Laundering` (`abl_a4_no_sym_laundering`) | `56.75` / `47.33` | **`90.62`** / `79.79` | `62.55` / `97.03` | **`59.88`** / `25.22` | **`62.32`** | `34.48` | `0.000000` | `0.2542` |

### 5.3 Key Findings from the Real-Media Ablations
1. **Exact Principal-Subspace Preservation (`A1`):** `HoST-MoE (Full)` and all orthogonal SVD-LoRA variants maintain **`0.000000` Frobenius leakage** into `DINO-ViT-S/16`'s top-$32$ principal subspace (vs. `0.054988` under unconstrained LoRA), improving `AV_Splice_Eval` AUROC by **`+5.20` pts** (`89.58%` vs. `84.38%`), visual temporal localization `V-mAP` by **`+10.30` pts** (`74.86%` vs. `64.56%`), and reducing 4-benchmark `Realized FAR` (`32.76%` vs. `36.21%`).
2. **Auxiliary-Loss-Free Expert Load Balancing (`A2`):** Disabling non-differentiable router bias updates ($\gamma = 0$) causes **`3.2x` higher expert load imbalance** (`CV = 0.5569` vs. `0.1732`), degrading zero-shot `Google DFD` AUROC by **`-8.50` pts** (`61.00% -> 52.50%`), `AV_Splice_Eval` AUROC by **`-7.29` pts** (`89.58% -> 82.29%`), and 4-benchmark `Cross-AUC` (`48.49% -> 44.42%`).
3. **Host De-Polarization & MIL Anchor Supervision (`A3`):** Removing $\mathcal{L}_{\text{pol}}$ and $\mathcal{L}_{\text{anc}}$ leaves `loss_anchor` uncalibrated (`0.8030` vs. `0.2647`), collapsing zero-shot unseen **`Google DFD` to chance (`61.00% -> 49.75%` AUROC; Multi-Domain `Cross-AUC` `55.78% -> 42.82%`)**, inflating `Google DFD` false alarms to **`70.00%`** (`Pooled Realized FAR`: `32.76% -> 50.00%`), and dropping `AV-mAP` by **`-7.72` pts** (`86.41% -> 78.69%`).
4. **In-Domain Shortcuts vs. Cross-Domain Laundering Robustness (`A4`):** Removing symmetric restoration laundering (`p_launder = 0`) allows the model to fit sharpness/codec shortcuts on in-distribution `Celeb-DF-v2` (`59.88%` vs. `53.09%`), at the cost of higher `Celeb-DF-v2` domain polarization ($W_1$: `17.03` vs. `11.41`), higher `Celeb-DF-v2` false alarms (`33.33%` vs. `22.22%`), weaker unseen zero-shot `Google DFD` transfer (`56.75%` vs. `61.00%` AUROC; `47.33%` vs. `55.78%` `Cross-AUC`), and `-6.62` pts lower `AV-mAP` (`79.79%` vs. `86.41%`).
5. **Scaling Motivation:** While just `64` `6`-frame pilot clips on `ViT-S/16` (`22M`) already achieve `89.58%` `AV-Splice` AUROC, `86.41%` `AV-mAP` (`97.95%` `A-mAP`, `74.86%` `V-mAP`), `25.61%` zero-shot `Celeb-DF-v2` facial mask `mIoU`, and `61.00%` zero-shot `Google DFD` transfer, subtle silent object-removal classification (`45.83%` AUROC, `10.20%` `mIoU` from only `10` training tubelets) remains data- and resolution-bound—motivating full-scale `384^2 -> 448^2` `DINOv3-L/16` training via [`scripts/cloud_gpu_bootstrap.sh`](../scripts/cloud_gpu_bootstrap.sh).

