# Physics-Constrained Machine Learning Surrogate for the Colebrook Friction Factor
## Colebrook Sürtünme Faktörü için Fizik Kısıtlı Makine Öğrenmesi Vekil Modeli

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tdmuftuoglu/colebrook-mgb-uq/blob/main/PC_ML_Colebrook_v3.ipynb)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-4.5.0-green.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Version](https://img.shields.io/badge/version-1.3-orange.svg)

---

## Overview / Genel Bakış

This repository provides the full reproducible pipeline for the manuscript:

> **Muftuoglu, T. D.** — *Physics-Constrained Machine-Learning Surrogates for the Colebrook Friction Factor: Monotonic Gradient Boosting, Uncertainty Quantification, and Open Benchmarking* — under review, *Theoretical and Computational Fluid Dynamics* (Scientific Reports, Nature Portfolio).

The Darcy–Weisbach friction factor `f` is fundamental to head-loss modelling in pressurised pipe systems. It is governed by the **Colebrook–White equation**, which is implicit in `f` and requires iterative solution. This repository introduces a **physics-constrained monotonic gradient boosting (MGB) surrogate** that:

- **Enforces physical monotonicity analytically** — ∂f/∂Re ≤ 0 and ∂f/∂(ε/D) ≥ 0 are guaranteed as hard constraints in the LightGBM learning algorithm, not just empirically satisfied.
- **Provides calibrated approximation-error bounds** — 95% prediction intervals via quantile gradient boosting combined with split conformal calibration, giving distribution-free coverage guarantees.
- **Benchmarks explicitly and reproducibly** — fixed data splits, common metrics, physics-violation auditing, and an open evaluation grid.

The model is **not** proposed as a replacement for Serghides' formula when only a point estimate is needed. Its value lies in applications that additionally require a reliability certificate: Monte Carlo uncertainty propagation, risk-aware hydraulic design, and surrogate-embedded optimisation workflows.

---

## Key Results (v1.3) / Temel Sonuçlar

| Metric | Value |
|---|---|
| MAPE on rough-pipe evaluation grid (ε/D > 0) | **0.168%** |
| MAPE on stratified test set | 0.251% |
| Monotonicity violations (1829 Re steps + 1800 ε/D steps) | **0** |
| Empirical coverage of nominal 95% prediction intervals | **95.5%** |
| Max absolute error (test set) | 5.75 × 10⁻³ |
| Training Re range | 4,000 – 10⁸ |
| Training grid size | 240 × 61 = 14,640 cases |

**Explicit baseline comparison on 24×7 rough-pipe evaluation grid:**

| Method | MAPE (%) | Monotonicity guarantee | UQ |
|---|---|---|---|
| Haaland | 0.236 | No | No |
| Swamee-Jain | 0.546 | No | No |
| Serghides | 0.0003 | No | No |
| **MGB (this study)** | **0.168** | **Yes** | **Yes** |

---

## What's New in v1.3 / Sürüm 1.3'teki Yenilikler

This release accompanies the **first peer-review revision** of the manuscript. All changes below were motivated by reviewer comments received from *Scientific Reports*.

### 🔧 Critical Bug Fix — Colebrook–White Solver

The Newton–Raphson Colebrook–White solver contained a substitution error in the `v1.0`–`v1.2` notebooks. With the substitution `x = 1/√f`, the denominator term was incorrectly coded as `2.51 / (Re * x)` instead of the correct `2.51 * x / Re`. This produced significant labelling errors for smooth-pipe cases (ε/D = 0) and low Reynolds numbers (up to ~63% error at Re = 4,000, ε/D = 0), while errors were negligible at high Re and high roughness.

**All training labels, evaluation metrics, and figures have been recomputed using the corrected solver.** The corrected implementation uses the mathematically exact form:

```python
# CORRECTED (v1.3)
denom = k_rel / 3.7 + 2.51 * x / Re      # x = 1/sqrt(f)

# BUGGY (v1.0 – v1.2)
# denom = k_rel / 3.7 + 2.51 / (Re * x)  ← WRONG
```

### 📊 New Figures

| Figure | Description | Reviewer request |
|---|---|---|
| **Fig 11** | Full Nikuradse (1933) experimental validation across 4 roughness values (ε/D = 0.004, 0.008, 0.016, 0.033), 30+ data points, with 95% conformal prediction bands | R1.3 |
| **Fig 12** | Computational timing benchmark: Newton–Raphson vs. Serghides vs. MGB across N = 1 to 10⁶ evaluations | R1.4 |
| **Fig 13** | Training grid density sensitivity: MAPE as a function of grid size (3,720 to 32,760 cases), showing near-plateau behaviour at the chosen 14,640-case grid | R1.7 |

All existing figures (Fig 1–10) have been **regenerated from the corrected solver** and produced at higher resolution (200 dpi, 14×10 in for multi-panel figures).

### 📐 Re Range Extended

The training and evaluation range was extended from Re ∈ [4,000 – 2.5×10⁷] (v1.x) to **Re ∈ [4,000 – 10⁸]** to cover the full turbulent regime including high-Re industrial applications.

### 🧪 Serghides Monotonicity Audit

A systematic check on a 500×200 grid (500 log-spaced Re × 200 linear ε/D values for ε/D ∈ [0.001, 0.06]) confirmed **zero monotonicity violations** for the Serghides formula across the physical domain. The manuscript now distinguishes between *empirical* monotonicity (which Serghides satisfies in practice) and the *analytical guarantee* enforced by the MGB constraints.

### 📝 Manuscript Changes

- Sections 1.1–1.3 merged into a unified "Background, motivation, and scope" section with extended literature positioning.
- UQ section expanded: explicit treatment of the two-stage design (quantile boosting + conformal calibration), discussion of limitations, and three new conformal prediction references.
- Fig 10 caption corrected: interval width growth at high Re is now described as *reduced surrogate emulation accuracy*, not physical uncertainty growth.
- Discussion substantially expanded: honest framing of MGB vs Serghides trade-off, Nikuradse/Colebrook offset explained, speed benchmark interpretation.
- Reference [2] updated from preprint (2018) to published version: Brki ́c & Praks (2019), *Mathematics*, 7(1), 34.

### 🐛 Figure Quality Fixes

- Removed `suptitle` from all figures (titles now only in LaTeX captions).
- Fixed empty ε/D = 0 panels in Fig 2 (residuals), Fig 3 (% error), and Fig 4 (parity) — previously blank due to a `if k > 0:` guard that incorrectly excluded smooth-pipe cases.
- Fixed Fig 4(a) showing only the reference line without scatter points.
- Fig 11 legend moved outside the plot area to avoid data overlap.
- All figure float specifiers changed from `[!ht]` to `[H]` in the LaTeX source (requires `\usepackage{float}`), preventing figures from splitting paragraphs.

---

## Repository Contents / Repo İçeriği

```
colebrook-mgb-uq/
├── PC_ML_Colebrook_v3.ipynb          # One-click Colab notebook (v1.3 pipeline)
├── mgb_eval_24x7.csv                 # 24×7 evaluation grid: predictions + 95% PI
├── summary.json                      # All key metrics and monotonicity check results
├── fig1.png  … fig13.png             # All manuscript figures (200 dpi)
├── colebrook_template_goalseek_ready_graphs.xlsx  # Excel template
├── requirements.txt                  # Python dependencies
├── LICENSE                           # MIT License
└── README.md                         # This file
```

### File descriptions

| File | Description |
|---|---|
| `PC_ML_Colebrook_v3.ipynb` | End-to-end pipeline: data generation → corrected Colebrook solver → MGB training → conformal calibration → all figures. Run with "Runtime → Run all". |
| `mgb_eval_24x7.csv` | Evaluation grid output. Columns: `Re`, `k_rel`, `f_colebrook`, `f_pred`, `PI_lower`, `PI_upper`, `covered`, `abs_err`, `pct_err`. |
| `summary.json` | Machine-readable metrics: RMSE, MAE, MAPE, MaxAbsError, coverage, PI widths, monotonicity violation counts for both MGB and Serghides. |
| `fig11.png` | Nikuradse validation (new in v1.3). |
| `fig12.png` | Speed benchmark (new in v1.3). |
| `fig13_density_sensitivity.png` | Training grid density sensitivity (new in v1.3). |
| `colebrook_template_goalseek_ready_graphs.xlsx` | Excel calculator: explicit formulas (Haaland, Swamee-Jain, Serghides) with Goal Seek setup for Colebrook, plus MGB lookup table. |

---

## How to Reproduce / Tekrarlanabilirlik

### Option 1 — Google Colab (Recommended / Önerilen)

1. Click the **"Open in Colab"** badge at the top of this page.
2. In Colab: **Runtime → Run all**.
3. All figures and metrics are generated automatically and packaged as `mgb_results_v3.zip` for download.

### Option 2 — Local Installation

```bash
git clone https://github.com/tdmuftuoglu/colebrook-mgb-uq.git
cd colebrook-mgb-uq
pip install -r requirements.txt
jupyter notebook PC_ML_Colebrook_v3.ipynb
```

**Requirements:** Python ≥ 3.9, LightGBM 4.5.0, scikit-learn 1.4.x, NumPy, pandas, matplotlib.

---

## Version History / Sürüm Geçmişi

| Version | Date | Summary |
|---|---|---|
| **v1.3** | 2025 | Corrected Colebrook solver; extended Re range to 10⁸; new figures (Nikuradse, speed benchmark, density sensitivity); all figures regenerated; manuscript revision R1 |
| v1.2 | 2025 | Figure size improvements; minor code refactoring |
| v1.1 | 2025 | Added Excel template and Colab badge |
| v1.0 | 2025 | Initial release accompanying manuscript submission |

---

## Mathematical Background / Matematiksel Arka Plan

The **Colebrook–White equation** governs turbulent pipe friction:

$$\frac{1}{\sqrt{f}} = -2\log_{10}\!\left(\frac{\varepsilon/D}{3.7} + \frac{2.51}{Re\sqrt{f}}\right)$$

The MGB surrogate learns the mapping $(Re, \varepsilon/D) \to f$ subject to:

$$\frac{\partial \hat{f}}{\partial (\log_{10} Re)} \leq 0, \qquad \frac{\partial \hat{f}}{\partial (\varepsilon/D)} \geq 0$$

Calibrated prediction intervals $[L, U]$ satisfy the finite-sample guarantee:

$$P(L \leq f_{\text{true}} \leq U) \geq 1 - \alpha \quad \text{(marginally, over the joint distribution)}$$

via split conformal prediction on a held-out calibration set.

---

## Limitations / Sınırlılıklar

- The surrogate was trained and validated in the **fully developed turbulent regime** (Re ≥ 4,000). It is not validated for laminar or transitional flow.
- The prediction intervals are **marginal** (averaged over the input distribution), not conditional on specific (Re, ε/D) pairs.
- In a Python/NumPy environment, the LightGBM inference is **slower** than both Newton–Raphson and Serghides due to Python call overhead. The computational advantage is realised in compiled deployment or when UQ intervals are co-evaluated at no extra cost.
- The model emulates the **Colebrook–White** equation. For uniform sand-roughness pipes (Nikuradse regime), the systematic Colebrook overestimation in the transitional regime is faithfully reproduced — this is a property of the reference equation, not of the surrogate.

---

## Citation / Atıf

If you use this repository in your research, please cite the manuscript and the software:

**Manuscript:**
```
Muftuoglu, T. D. (under review). Physics-Constrained Machine-Learning Surrogates 
for the Colebrook Friction Factor: Monotonic Gradient Boosting, Uncertainty 
Quantification, and Open Benchmarking. Theoretical and Computational Fluid Dynamics 
(Scientific Reports, Nature Portfolio).
```

**Software (Zenodo):**
```
Muftuoglu, T. D. (2025). colebrook-mgb-uq v1.3: Physics-Constrained Machine 
Learning Surrogate for the Colebrook Friction Factor. Zenodo. 
https://doi.org/10.5281/zenodo.XXXXXXX
```
*(DOI will be updated upon Zenodo deposit)*

---

## License / Lisans

This project is licensed under the **MIT License** — see [`LICENSE`](LICENSE) for full terms.

Bu proje **MIT Lisansı** altında lisanslanmıştır.

---

## Contact / İletişim

**Tevfik Denizhan Muftuoglu**, Asst. Prof. Dr.  
Civil Engineering (English) Department, Istanbul Aydin University  
📧 tmuftuoglu@aydin.edu.tr
