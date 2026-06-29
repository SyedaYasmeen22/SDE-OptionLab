# SDE-OptionLab

A configurable simulation toolkit for pricing **spread** and **better-of**
options on two correlated assets — built on the application blueprint derived
from the MPhil thesis *"A Simulation-Based Model of Stochastic Differential
Equations in Spread and Better-of Option Pricing."*

It ships as:
- `sde_optionlab/` — a standalone, dependency-light Python pricing engine
  (no Streamlit dependency; usable from any script or notebook).
- `app.py` — an interactive Streamlit web interface on top of that engine.

---

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Run the app

```bash
streamlit run app.py
```

This opens the app in your browser (default `http://localhost:8501`).

## 3. Run the validation tests

```bash
python test_engine.py
```

This checks the engine against known closed-form and cross-method
benchmarks (20 checks): Margrabe vs. Monte Carlo, better-of/worst-of vs.
their closed forms, the binomial tree vs. Margrabe and vs. Kirk's
approximation, Euler-Maruyama discretization bias shrinking as steps
increase, Greeks signs, VaR/CVaR consistency, and parameter estimation
recovery from synthetic data.

---

## Project structure

```
sde_optionlab_app/
├── app.py                   # Streamlit UI
├── requirements.txt
├── test_engine.py           # validation / regression tests
└── sde_optionlab/
    ├── __init__.py
    ├── models.py             # asset dynamics simulators
    ├── payoffs.py            # spread / margrabe / better-of / worst-of / basket payoffs
    ├── pricers.py            # analytical formulas, binomial tree, Monte Carlo driver
    ├── greeks.py              # bumped-MC Greeks, VaR/CVaR
    ├── diagnostics.py         # method comparison & convergence studies
    └── estimation.py          # CSV-driven mu/sigma/rho calibration
```

## What's implemented

**Asset dynamics** (`models.py`)
- Correlated GBM — exact terminal-value simulation (no discretization error)
- Correlated GBM — Euler–Maruyama
- Correlated GBM — Milstein
- Skew-correlated GBM — a *practical proxy* for asymmetric dependence
  (downside correlation rises relative to upside), in the spirit of
  Pasricha–He skew-correlated Brownian motion. This is **not** the closed-form
  Pasricha–He process — it's a simulation-friendly approximation useful for
  stress-testing common downside shocks. The docstring in `models.py` flags
  this explicitly.
- Mean-reverting spread (Ornstein–Uhlenbeck), modelling the spread itself
  rather than the two legs separately — appropriate once maturities run
  well past ~90 days, per the thesis's own note on this regime.
- Merton-style jump-diffusion with correlated diffusive shocks and
  independent compound-Poisson jumps per asset.

**Payoffs** (`payoffs.py`): spread call/put, Margrabe exchange (K=0 spread),
better-of, worst-of, and basket (n-asset weighted average).

**Pricing engines** (`pricers.py`)
- Analytical: Black–Scholes, Margrabe (1978) closed form, better-of/worst-of
  via the max/min decomposition, and **Kirk's (1995) approximation** for
  spread options with a *nonzero* strike — there is no exact closed form once
  K ≠ 0, so Kirk's approximation is the standard desk-level formula used in
  practice for this case (commodity- and rate-spread options).
- A correlated two-asset binomial tree (Boyle 1988 four-branch extension of
  CRR), supporting **American exercise** via backward induction. The
  moment-matching branch probabilities can occasionally fall slightly outside
  [0, 1] for large |ρ| combined with a coarse time grid; the code clips and
  renormalises in that case (accuracy recovers as the number of steps grows).
- A single, general Monte Carlo driver that combines any model with any
  payoff, with antithetic variates and a control variate (using the
  analytically-known mean of asset 1's discounted terminal value).

**Diagnostics** (`diagnostics.py`): side-by-side method/MAE/RMSE comparison
tables and convergence studies vs. path count or step count — the live,
on-demand equivalent of the thesis's static tables and figures.

**Risk** (`greeks.py`): per-asset deltas via bumped Monte Carlo with common
random numbers (so MC noise mostly cancels between the up/down bump), an
implied hedge ratio for two-asset payoffs, and historical VaR/CVaR computed
directly from the simulated payoff distribution (no extra simulation needed).

**Parameter estimation** (`estimation.py`): upload a two-column CSV of
historical prices (or EPS/PAT, FX rates, etc.) and estimate annualised
μ/σ/ρ via log returns, with an optional rolling-window re-estimator.

## Honest modeling caveats

- **Drift (μ) is hard to estimate from price history.** Volatility and
  correlation estimates stabilise reasonably from a few years of daily data;
  drift estimates remain noisy even with a *lot* of data — this is a
  well-known statistical fact about return series, not a flaw in the
  estimator. For fair-value pricing, set μ = r (and use dividend yields q if
  relevant) rather than relying on the historical drift estimate.
- **Kirk's approximation and the skew-correlated GBM model are both
  approximations**, not exact closed forms — they're flagged as such in the
  code and in the app's UI captions.
- **The binomial tree assumes lognormal (GBM-style) dynamics**; it is not
  offered for the mean-reverting or jump-diffusion models, where it would not
  be a meaningful comparison.

## Extending it

- **Basket options for >2 assets**: `payoffs.basket_payoff` and
  `models.simulate_exact_gbm`/`simulate_gbm_path` already support an
  arbitrary number of assets if you pass a full correlation matrix instead of
  a scalar ρ — the Streamlit UI currently only exposes 2-asset inputs, so a
  natural next step is an "n assets" mode with a correlation-matrix editor.
- **Live data feed**: `estimation.py` works on any DataFrame, so wiring in a
  market-data API (rather than only CSV upload) is a small addition.
- **Excel/PDF export**: the "Report export" tab already produces JSON, CSV,
  and a one-page PDF; an Excel pack (per the original blueprint's "PDF/Excel
  pack" idea) could reuse the same `export_dict`.
