"""
Validation tests for SDE-OptionLab's pricing engine: checks Monte Carlo and
binomial-tree prices against known closed-form benchmarks, and sanity-checks
Greeks, diagnostics, and the estimation module. Run with:

    python test_engine.py
"""

import numpy as np
import pandas as pd

from sde_optionlab import models, payoffs, pricers, greeks, diagnostics, estimation

PASS = []
FAIL = []


def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print(f"  [PASS] {name}")
    else:
        FAIL.append(name)
        print(f"  [FAIL] {name}  {detail}")


# ---------------------------------------------------------------------------
print("1) Margrabe exchange option: exact-GBM Monte Carlo vs closed form")
# ---------------------------------------------------------------------------
S0 = np.array([100.0, 95.0])
r, T = 0.05, 1.0
mu = np.array([r, r])   # NOTE: closed-form benchmarks (Margrabe etc.) assume
                        # risk-neutral drift mu_i = r - q_i for every tradable
                        # asset; using a real-world mu != r is fine for the
                        # engine itself but will not match these formulas,
                        # since the formulas price under no-arbitrage.
sigma = np.array([0.25, 0.20])
rho = 0.4

analytical = pricers.margrabe_price(S0[0], S0[1], sigma[0], sigma[1], rho, T)

params = {"S0": S0, "mu": mu, "sigma": sigma, "rho": rho, "r": r, "T": T}
rng = np.random.default_rng(1)
mc = pricers.monte_carlo_price("margrabe", "gbm_exact", params, n_paths=400_000, n_steps=1, rng=rng,
                                antithetic=True, control_variate=True)

print(f"  analytical = {analytical:.4f}, MC = {mc['price']:.4f} +/- {1.96*mc['std_error']:.4f}")
check("Margrabe MC matches closed form within 3 std errors",
      abs(mc["price"] - analytical) < max(3 * mc["std_error"], 0.02),
      f"diff={abs(mc['price']-analytical):.4f}, 3*se={3*mc['std_error']:.4f}")


# ---------------------------------------------------------------------------
print("\n2) Spread option == Margrabe when K = 0")
# ---------------------------------------------------------------------------
params_spread = dict(params, K=0.0, option_sub="call")
rng = np.random.default_rng(2)
mc_spread = pricers.monte_carlo_price("spread", "gbm_exact", params_spread, 200_000, 1, rng, antithetic=True)
check("Spread(K=0) price close to Margrabe price",
      abs(mc_spread["price"] - analytical) < max(3 * mc_spread["std_error"], 0.03))


# ---------------------------------------------------------------------------
print("\n3) Better-of / worst-of: Monte Carlo vs closed form, and put-call-style identity")
# ---------------------------------------------------------------------------
bo_analytical = pricers.better_of_analytical(S0[0], S0[1], sigma[0], sigma[1], rho, T)
wo_analytical = pricers.worst_of_analytical(S0[0], S0[1], sigma[0], sigma[1], rho, T)

rng = np.random.default_rng(3)
mc_bo = pricers.monte_carlo_price("better_of", "gbm_exact", params, 400_000, 1, rng, antithetic=True)
rng = np.random.default_rng(4)
mc_wo = pricers.monte_carlo_price("worst_of", "gbm_exact", params, 400_000, 1, rng, antithetic=True)

print(f"  better-of: analytical={bo_analytical:.4f} MC={mc_bo['price']:.4f}")
print(f"  worst-of : analytical={wo_analytical:.4f} MC={mc_wo['price']:.4f}")
check("Better-of MC matches closed form", abs(mc_bo["price"] - bo_analytical) < max(3 * mc_bo["std_error"], 0.05))
check("Worst-of MC matches closed form", abs(mc_wo["price"] - wo_analytical) < max(3 * mc_wo["std_error"], 0.05))

discounted_sum = mc_bo["price"] + mc_wo["price"]
fwd_sum = S0[0] * np.exp((mu[0] - r) * T) + S0[1] * np.exp((mu[1] - r) * T)  # E[disc*(S1+S2)]
check("max + min = S1 + S2 identity holds (within MC noise)",
      abs(discounted_sum - fwd_sum) < 0.1, f"diff={abs(discounted_sum - fwd_sum):.4f}")


# ---------------------------------------------------------------------------
print("\n4) Two-asset binomial tree (European) vs Margrabe closed form, and American >= European")
# ---------------------------------------------------------------------------
payoff_fn = payoffs.make_binomial_payoff("margrabe")
euro_price = pricers.two_asset_binomial(S0[0], S0[1], sigma[0], sigma[1], rho, r, T,
                                         n_steps=150, payoff_fn=payoff_fn, american=False)
amer_price = pricers.two_asset_binomial(S0[0], S0[1], sigma[0], sigma[1], rho, r, T,
                                         n_steps=150, payoff_fn=payoff_fn, american=True)
print(f"  binomial European = {euro_price:.4f}, analytical = {analytical:.4f}, binomial American = {amer_price:.4f}")
check("Binomial European converges to Margrabe closed form", abs(euro_price - analytical) < 0.15)
check("American price >= European price (early exercise option has non-negative value)",
      amer_price >= euro_price - 1e-6)


# ---------------------------------------------------------------------------
print("\n4b) Kirk's approximation (nonzero-strike spread call) vs Monte Carlo")
# ---------------------------------------------------------------------------
K_nonzero = 3.0
kirk_price = pricers.kirk_spread_approx(S0[0], S0[1], sigma[0], sigma[1], rho, K_nonzero, r, T, option="call")
params_k = dict(params, K=K_nonzero, option_sub="call")
rng = np.random.default_rng(21)
mc_k = pricers.monte_carlo_price("spread", "gbm_exact", params_k, 400_000, 1, rng, antithetic=True)
print(f"  Kirk approx = {kirk_price:.4f}, MC = {mc_k['price']:.4f} +/- {1.96*mc_k['std_error']:.4f}")
check("Kirk's approximation close to Monte Carlo for a nonzero-strike spread call",
      abs(kirk_price - mc_k["price"]) < max(5 * mc_k["std_error"], 0.1),
      f"diff={abs(kirk_price - mc_k['price']):.4f}")


# ---------------------------------------------------------------------------
print("\n4c) Two-asset binomial tree for nonzero-strike spread vs Monte Carlo")
# ---------------------------------------------------------------------------
payoff_fn_spread = payoffs.make_binomial_payoff("spread", K=K_nonzero, option_sub="call")
binom_spread = pricers.two_asset_binomial(S0[0], S0[1], sigma[0], sigma[1], rho, r, T, 150, payoff_fn_spread)
print(f"  binomial = {binom_spread:.4f}, MC = {mc_k['price']:.4f}")
check("Binomial tree matches Monte Carlo for nonzero-strike spread call",
      abs(binom_spread - mc_k["price"]) < max(5 * mc_k["std_error"], 0.1))


# ---------------------------------------------------------------------------
print("\n5) Euler-Maruyama discretization bias shrinks as n_steps grows (vs exact GBM)")
# ---------------------------------------------------------------------------
errs = []
for n_steps in [5, 20, 100]:
    rng = np.random.default_rng(7)
    mc_euler = pricers.monte_carlo_price("margrabe", "gbm_euler", params, 200_000, n_steps, rng, antithetic=True)
    errs.append(abs(mc_euler["price"] - analytical))
    print(f"  n_steps={n_steps:4d}  euler_price={mc_euler['price']:.4f}  abs_error={errs[-1]:.4f}")
check("Discretization error trends down as n_steps increases (5 -> 100)", errs[-1] <= errs[0] + 0.05)


# ---------------------------------------------------------------------------
print("\n6) Greeks: spread call delta_1 > 0, delta_2 < 0, sensible hedge ratio")
# ---------------------------------------------------------------------------
params_g = dict(params, K=5.0, option_sub="call")
g = greeks.deltas("spread", "gbm_exact", params_g, n_paths=100_000, n_steps=1, seed=99)
print(f"  deltas = {g}")
check("delta_1 > 0 for a spread call", g["delta_1"] > 0)
check("delta_2 < 0 for a spread call", g["delta_2"] < 0)


# ---------------------------------------------------------------------------
print("\n7) VaR/CVaR sanity check on simulated payoff distribution")
# ---------------------------------------------------------------------------
risk = greeks.value_at_risk(mc_spread["payoffs_discounted"], mc_spread["price"], confidence=0.95)
print(f"  {risk}")
check("CVaR >= VaR (tail loss at least as large as the threshold loss)", risk["CVaR"] >= risk["VaR"] - 1e-9)


# ---------------------------------------------------------------------------
print("\n8) Mean-reverting spread model and jump-diffusion model run without error")
# ---------------------------------------------------------------------------
mr_params = {"S0": S0, "kappa": 2.0, "theta": 5.0, "sigma_x": 8.0, "r": r, "T": T, "K": 0.0, "option_sub": "call"}
rng = np.random.default_rng(11)
mc_mr = pricers.monte_carlo_price("spread", "mean_reverting", mr_params, 50_000, 50, rng)
check("Mean-reverting spread model returns a finite non-negative price",
      np.isfinite(mc_mr["price"]) and mc_mr["price"] >= 0)

jd_params = dict(params, jump_intensity=1.0, jump_mean=-0.05, jump_std=0.1)
rng = np.random.default_rng(12)
mc_jd = pricers.monte_carlo_price("margrabe", "jump_diffusion", jd_params, 50_000, 50, rng)
check("Jump-diffusion model returns a finite non-negative price",
      np.isfinite(mc_jd["price"]) and mc_jd["price"] >= 0)

skew_params = dict(params, skew=0.4)
rng = np.random.default_rng(13)
mc_skew = pricers.monte_carlo_price("margrabe", "skew_gbm", skew_params, 50_000, 50, rng)
check("Skew-correlated GBM model returns a finite non-negative price",
      np.isfinite(mc_skew["price"]) and mc_skew["price"] >= 0)


# ---------------------------------------------------------------------------
print("\n9) Diagnostics: method comparison table and convergence study run cleanly")
# ---------------------------------------------------------------------------
table = diagnostics.method_comparison_table(
    "margrabe", ["gbm_exact", "gbm_euler", "gbm_milstein"], params, 50_000, 50, benchmark=analytical
)
print(table.to_string(index=False))
check("Comparison table has one row per model and finite errors",
      len(table) == 3 and table["abs_error"].notna().all())

conv = diagnostics.convergence_study("margrabe", "gbm_euler", params, 50_000, [5, 20, 80])
check("Convergence study returns one row per step count", len(conv) == 3)


# ---------------------------------------------------------------------------
print("\n10) Parameter estimation module recovers sensible mu/sigma/rho from synthetic data")
# ---------------------------------------------------------------------------
rng_np = np.random.default_rng(0)
n = 1000
true_sigma = np.array([0.3, 0.25])
true_rho = 0.5
true_mu = np.array([0.08, 0.06])
L = np.linalg.cholesky(np.array([[1, true_rho], [true_rho, 1]]))
z = rng_np.standard_normal((n, 2)) @ L.T
dt = 1 / 252
log_ret = (true_mu - 0.5 * true_sigma ** 2) * dt + true_sigma * np.sqrt(dt) * z
prices = 100 * np.exp(np.cumsum(log_ret, axis=0))
df = pd.DataFrame(prices, columns=["A", "B"])
est = estimation.estimate_params_from_prices(df, "A", "B")
print(f"  estimated mu={est['mu']}, sigma={est['sigma']}, rho={est['rho']:.3f} (true mu={true_mu}, sigma={true_sigma}, rho={true_rho})")
check("Estimated sigma within 15% of true sigma", np.all(np.abs(est["sigma"] - true_sigma) / true_sigma < 0.15))
check("Estimated rho within 0.1 of true rho", abs(est["rho"] - true_rho) < 0.1)


# ---------------------------------------------------------------------------
print(f"\n=== {len(PASS)} passed, {len(FAIL)} failed ===")
if FAIL:
    print("FAILED:", FAIL)
    raise SystemExit(1)
