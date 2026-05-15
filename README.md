# Physics-Constrained Machine Learning Surrogate for the Colebrook Friction Factor

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tdmuftuoglu/colebrook-mgb-uq/blob/main/PC_ML_Colebrook_v3.ipynb)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-4.5.0-green.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## What This Is

This repository contains the complete, reproducible pipeline for the following manuscript (under peer review):

> Muftuoglu, T. D. — *Physics-Constrained Machine-Learning Surrogates for the Colebrook Friction Factor: Monotonic Gradient Boosting, Uncertainty Quantification, and Open Benchmarking

Everything in this repository — from raw data generation to every figure in the paper — can be reproduced by running a single Google Colab notebook.

---

## The Problem

The **Darcy–Weisbach friction factor** `f` is one of the most fundamental quantities in hydraulic engineering. It determines head loss in pressurised pipe systems and appears in every pipe flow calculation.

For turbulent flow, `f` is governed by the **Colebrook–White equation**:

```
1/sqrt(f) = -2 * log10( (eps/D)/3.7  +  2.51/(Re*sqrt(f)) )
```

where `Re` is the Reynolds number and `eps/D` is the relative roughness of the pipe wall.

The problem is that this equation **cannot be solved explicitly** — `f` appears on both sides. Engineers must iterate to find the answer. For a single calculation that is instant; but in large parametric studies, Monte Carlo uncertainty analyses, or optimisation loops that require millions of evaluations, iteration can become cumbersome.

**Explicit approximations** (Haaland, Swamee-Jain, Serghides) exist and are widely used. They give a closed-form answer without iteration. Serghides in particular is exceptionally accurate. But all of them share two limitations:

1. They give only a single number with no indication of how reliable that number is.
2. None carry a formal mathematical guarantee that they will always respect the physical behaviour of the friction factor — specifically that `f` must decrease as `Re` increases, and increase as `eps/D` increases.

---

## What This Repository Offers

A **machine learning surrogate** trained on synthetically generated Colebrook solutions that addresses both limitations simultaneously.

### 1. Guaranteed physical monotonicity

The model is trained using LightGBM with hard monotonicity constraints embedded directly in the learning algorithm:

- `f` is guaranteed to be non-increasing in `log10(Re)` at any fixed `eps/D`
- `f` is guaranteed to be non-decreasing in `eps/D` at any fixed `Re`

This is not an empirical observation about typical model behaviour — it is an analytical guarantee that holds for every possible input, including inputs far outside the training distribution.

### 2. Calibrated approximation-error bounds

The surrogate also outputs a **95% prediction interval** alongside every point prediction. These intervals are produced by quantile gradient boosting and calibrated using **split conformal prediction**, which provides a distribution-free finite-sample coverage guarantee.

It is important to understand what these intervals mean: they quantify *how accurately the surrogate approximates the Colebrook equation*, not physical uncertainty about pipe friction. The Colebrook equation is deterministic; the intervals are emulation-error bounds.

### 3. Honest benchmarking

The surrogate is benchmarked transparently against Serghides and the other explicit formulas on a fixed, publicly available evaluation grid.

Serghides remains the most accurate point estimator by a large margin (MAPE = 0.0003%). The MGB surrogate sits between Haaland and Serghides in point accuracy (MAPE = 0.168%) but is the only method that provides both a monotonicity certificate and calibrated reliability bounds. The intended use case is not to replace Serghides when a simple number is sufficient — it is for workflows that additionally need to know how reliable that number is.

---

## Results

| Metric | Value |
|---|---|
| MAPE — rough-pipe evaluation grid (eps/D > 0) | **0.168%** |
| MAPE — stratified test set | 0.251% |
| Empirical coverage of nominal 95% prediction intervals | **95.5%** |
| Monotonicity violations (1,829 Re checks + 1,800 eps/D checks) | **0** |
| Max absolute error — test set | 5.75e-3 |
| Training Re range | 4,000 – 1e8 |
| Training grid | 240 x 61 = 14,640 cases |

**Comparison with explicit formulas (24x7 rough-pipe evaluation grid):**

| Method | MAPE (%) | Monotonicity guarantee | Uncertainty bounds |
|---|---|---|---|
| Haaland | 0.236 | No | No |
| Swamee-Jain | 0.546 | No | No |
| Serghides | 0.0003 | No | No |
| **MGB (this work)** | **0.168** | **Yes** | **Yes** |

---

## Repository Contents

```
colebrook-mgb-uq/
│
├── PC_ML_Colebrook_v3.ipynb               ← Main notebook — run this
│
├── mgb_eval_24x7.csv                      ← Evaluation grid predictions + intervals
├── summary.json                           ← All key metrics, machine-readable
│
├── fig1.png  through  fig13.png           ← All manuscript figures
│
├── colebrook_template_goalseek_ready_graphs.xlsx   ← Excel calculator
│
├── requirements.txt                       ← Python dependencies
├── LICENSE                                ← MIT License
└── README.md                              ← This file
```

### File descriptions

**`PC_ML_Colebrook_v3.ipynb`**  
The end-to-end pipeline in a single Colab notebook. It generates the synthetic training data, solves the Colebrook equation for each grid point using a Newton–Raphson solver converged to 1e-12 relative tolerance, trains the monotonic gradient boosting model and the quantile models, calibrates the conformal prediction intervals, evaluates everything against explicit baselines, and produces all 13 figures. Running "Runtime → Run all" is all that is required. All outputs are packaged as `mgb_results_v3.zip` and downloaded automatically.

**`mgb_eval_24x7.csv`**  
The 24x7 evaluation grid output. 24 log-spaced Re values from 4,000 to 1e8, crossed with eps/D = {0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06}, giving 168 cases. Columns: `Re`, `k_rel` (= eps/D), `f_colebrook` (reference), `f_pred` (MGB point prediction), `PI_lower`, `PI_upper`, `covered` (1 if true value falls inside the interval, 0 otherwise), `abs_err`, `pct_err`.

**`summary.json`**  
Machine-readable dictionary of all reported metrics: RMSE, MAE, MAPE on test set and evaluation grid, coverage, mean and median interval widths, and monotonicity check results for both MGB and Serghides on their respective grids.

**`fig1.png` – `fig13.png`**  
All 13 manuscript figures at 200 dpi. Figs 1–6 compare the explicit formulas against Colebrook across seven roughness values. Figs 7–10 characterise the MGB surrogate — parity, residuals, coverage diagnostic, and interval width vs Re. Fig 11 shows external validation against Nikuradse (1933) experimental data for four roughness values. Fig 12 shows a computational timing comparison between Newton–Raphson, Serghides, and MGB. Fig 13 shows how model accuracy changes as the training grid is made denser.

**`colebrook_template_goalseek_ready_graphs.xlsx`**  
An Excel workbook with three components: (i) an explicit formula calculator for Haaland, Swamee-Jain, and Serghides; (ii) a Goal Seek setup that lets the user solve the full Colebrook equation iteratively inside Excel; (iii) a lookup table from the MGB surrogate for users who do not have Python.

---

## How to Run

### Google Colab (recommended)

1. Click the **Open in Colab** badge at the top of this page.
2. Select **Runtime → Run all**.
3. When the notebook finishes, it downloads `mgb_results_v3.zip` containing all figures and metrics.

No local installation is needed. The notebook installs its own dependencies inside the Colab session.

### Local installation

```bash
git clone https://github.com/tdmuftuoglu/colebrook-mgb-uq.git
cd colebrook-mgb-uq
pip install -r requirements.txt
jupyter notebook PC_ML_Colebrook_v3.ipynb
```

Requirements: Python >= 3.9, LightGBM 4.5.0, scikit-learn 1.4.x, NumPy, pandas, matplotlib.

---

## Figures

| Figure | What it shows |
|---|---|
| Fig 1 | Re vs friction factor for all 7 eps/D values — Colebrook vs the three explicit formulas |
| Fig 2 | Residuals (delta-f) of the explicit formulas relative to Colebrook |
| Fig 3 | Percentage error of the explicit formulas relative to Colebrook |
| Fig 4 | Parity plots — approximate vs Colebrook, by eps/D |
| Fig 5 | Average percentage error of each explicit formula by roughness value |
| Fig 6 | Standard deviation of percentage error by roughness value |
| Fig 7 | Parity plot: MGB predictions vs Colebrook on the test set |
| Fig 8 | MGB residuals vs Re on the test set |
| Fig 9 | Coverage diagnostic: distance of true values to the conformal prediction bounds |
| Fig 10 | Prediction interval width vs Re — shows where surrogate emulation is harder |
| Fig 11 | MGB predictions and 95% PI bands vs Nikuradse (1933) experimental data |
| Fig 12 | Wall-clock timing: Newton–Raphson vs Serghides vs MGB, N = 1 to 1e6 evaluations |
| Fig 13 | Training grid density sensitivity: MAPE vs number of training cases |

---

## Technical Notes

### How the Colebrook equation is solved

To generate training labels, the Colebrook equation is solved using Newton–Raphson iteration. The substitution `x = 1/sqrt(f)` is used, turning the implicit equation into a root-finding problem that converges smoothly. The correct update is:

```python
denom = k_rel / 3.7  +  2.51 * x / Re     # x = 1/sqrt(f)
g     = x + 2 * log10(denom)
dg    = 1 + (2/ln10) * (2.51/Re) / denom
x    -= g / dg
```

Iteration continues until the relative step size falls below 1e-12.

### How monotonicity is enforced

LightGBM's `monotone_constraints` parameter is set to `[-1, +1]` for the two input features `[log10(Re), eps/D]`. This enforces the physical constraints at the level of each individual tree split — not as a regularisation penalty, but as a hard structural constraint on every leaf of every tree.

### How the prediction intervals are calibrated

A held-out calibration set (15% of the full dataset, stratified by eps/D bin) is used to compute conformity scores:

```
s_i = max(L_i - y_i,  y_i - U_i)
```

where `L_i` and `U_i` are the raw 5th and 95th quantile predictions and `y_i` is the true Colebrook value. The calibration quantile `q_hat` is then computed as the `(1 - alpha)`-quantile of these scores. At inference, intervals are expanded to `[L - q_hat,  U + q_hat]`. This adjustment adds no model calls at inference time.

### Training configuration

```
Point model:   n_estimators=1200, learning_rate=0.05, num_leaves=64,
               min_child_samples=40, subsample=0.9, colsample_bytree=0.9,
               reg_lambda=1.0, objective="l2", monotone_constraints=[-1, +1]

Quantile models (tau=0.05 and tau=0.95):
               n_estimators=800, same hyperparameters, objective="quantile"
```

---

## Limitations

**Flow regime.** The model was trained and validated only in the fully developed turbulent regime (Re >= 4,000). It has not been tested for laminar or transitional flow.

**Coverage type.** The 95% coverage guarantee is marginal — it holds on average over the joint distribution of (Re, eps/D), not conditionally for each specific input pair. Interval widths vary across the domain and are wider at high Re where the Colebrook friction factor is nearly constant and the surrogate has less gradient information to work with.

**Speed.** In a Python/NumPy environment, LightGBM inference is slower than both the vectorised Newton–Raphson solver and Serghides due to Python call overhead. The computational advantage of the surrogate is realised when uncertainty intervals are needed at no extra cost, in compiled C deployment, or when the model is embedded in a differentiable optimisation loop that consumes both predictions and bounds.

**Reference equation.** The model emulates the Colebrook–White equation. In the hydrodynamically transitional regime, Colebrook systematically overestimates `f` compared to Nikuradse's uniform sand-roughness experimental data — because it was calibrated against commercial pipe data, which behaves differently. The surrogate faithfully replicates this offset, because that is what it was trained to do. For uniform sand-roughness pipes, the Nikuradse equation family is the appropriate reference.

---

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{muftuoglu2025colebrook,
  author  = {Muftuoglu, Tevfik Denizhan},
  title   = {Physics-Constrained Machine-Learning Surrogates for the Colebrook
             Friction Factor: Monotonic Gradient Boosting, Uncertainty
             Quantification, and Open Benchmarking},
  journal = {Scientific Reports},
  year    = {2025},
  note    = {Under review}
}

@software{muftuoglu2025colebrook_code,
  author = {Muftuoglu, Tevfik Denizhan},
  title  = {colebrook-mgb-uq: Physics-Constrained Machine Learning Surrogate
            for the Colebrook Friction Factor},
  year   = {2025},
  url    = {https://github.com/tdmuftuoglu/colebrook-mgb-uq},
  doi    = {10.5281/zenodo.XXXXXXX}
}
```

*(Zenodo DOI will be updated after deposit.)*

---

## License

MIT License — see [`LICENSE`](LICENSE) for full terms.

---

## Contact

**Tevfik Denizhan Muftuoglu**, Asst. Prof. Dr.  
Civil Engineering (English) Department, Istanbul Aydin University  
tmuftuoglu@aydin.edu.tr
