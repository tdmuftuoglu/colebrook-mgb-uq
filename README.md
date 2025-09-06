# Physics-Constrained ML Surrogate for Colebrook (Monotonic GBM + UQ)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tdmuftuoglu/colebrook-mgb-uq/blob/main/PC-ML-Colebrook.ipynb)

This repository provides a physics-constrained machine-learning surrogate for the Colebrook friction factor.  
It matches Colebrook–White accuracy, enforces monotonic trends (↓ in Re, ↑ in ε/D), and provides calibrated prediction intervals.

---

## Reproduce the paper

1. Click the **Open in Colab** badge above.  
2. In Colab: **Runtime → Run all**.  
3. The notebook will:
   - generate the (Re, ε/D) grid and compute Colebrook labels,
   - train the monotonic GBM surrogate,
   - conformally calibrate 95% prediction intervals,
   - export metrics and figures.

**It reproduces the figures and tables cited in the manuscript (Figures 43–46; Tables 1–3).**

---

## Repository contents

- `PC-ML-Colebrook.ipynb` — one-click Colab notebook (end-to-end pipeline).  
- `mgb_eval_24x7.csv` — 24×7 evaluation grid with predictions and 95% PIs.  
- `summary.json` — RMSE, MAE, MAPE, MaxAbsError, coverage, interval widths, monotonicity checks.  
- `parity_test.png`, `residual_vs_re_test.png`, `coverage_diag.png`, `width_vs_re_eval.png` — generated figures.  
- `colebrook_template_goalseek_ready_graphs.xlsx` — Excel template (explicit formulas + Colebrook via Goal Seek).  
- `requirements.txt` — optional local install.  
- `LICENSE` — MIT license.

---

## Requirements (local, optional)

Colab installs everything automatically. To run locally:

~~~bash
pip install -r requirements.txt
# or minimal:
# pip install lightgbm scikit-learn numpy pandas matplotlib
~~~

---

## How to cite

If you use this repository, please cite:

Muftuoglu, T. D. (2025). *Physics-Constrained Machine-Learning Surrogates for the Colebrook Friction Factor: Monotonic Gradient Boosting, Uncertainty Quantification, and Open Benchmarking*. GitHub. https://github.com/tdmuftuoglu/colebrook-mgb-uq

**BibTeX**
~~~bibtex
@misc{Muftuoglu2025_Colebrook_MGB_UQ,
  title   = {Physics-Constrained Machine-Learning Surrogates for the Colebrook Friction Factor: Monotonic Gradient Boosting, Uncertainty Quantification, and Open Benchmarking},
  author  = {Muftuoglu, Tevfik Denizhan},
  year    = {2025},
  howpublished = {\url{https://github.com/tdmuftuoglu/colebrook-mgb-uq}},
  note    = {Version v1.0},
  url     = {https://github.com/tdmuftuoglu/colebrook-mgb-uq}
}
~~~

---

## License

This project is licensed under the **MIT License** — see [`LICENSE`](LICENSE) for full terms.
