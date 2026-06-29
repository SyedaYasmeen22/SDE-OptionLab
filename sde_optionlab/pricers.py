"""
Pricing engines for SDE-OptionLab:

  - Analytical closed forms: Black-Scholes, Margrabe exchange option, and the
    better-of/worst-of decomposition built on top of Margrabe.
  - Two-asset correlated binomial tree (Boyle 1988 / Rubinstein-style
    four-branch tree), supporting both European and American exercise.
  - A general Monte Carlo driver that plugs together any asset-dynamics
    model from models.py with any payoff from payoffs.py, with optional
    antithetic variates and a control variate.
"""

import time
import numpy as np
from scipy.stats import norm

from . import models as M
from . import payoffs as P


# ---------------------------------------------------------------------------
# Analytical benchmarks
# ---------------------------------------------------------------------------

def black_scholes_price(S, K, sigma, r, T, q=0.0, option="call"):
    """Single-asset Black-Scholes price; used as a building block / sanity check."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if option == "call" else max(K - S, 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def margrabe_price(S1, S2, sigma1, sigma2, rho, T, q1=0.0, q2=0.0):
    """Closed-form Margrabe (1978) exchange-option price: max(S1 - S2, 0).
    Valid for two correlated GBMs with continuous dividend yields q1, q2.
    Notably independent of the risk-free rate r (numeraire invariance).
    """
    sigma_m = np.sqrt(max(sigma1 ** 2 + sigma2 ** 2 - 2 * rho * sigma1 * sigma2, 0.0))
    fwd1, fwd2 = S1 * np.exp(-q1 * T), S2 * np.exp(-q2 * T)
    if sigma_m <= 1e-12 or T <= 0:
        return max(fwd1 - fwd2, 0.0)
    d1 = (np.log(fwd1 / fwd2) + 0.5 * sigma_m ** 2 * T) / (sigma_m * np.sqrt(T))
    d2 = d1 - sigma_m * np.sqrt(T)
    return fwd1 * norm.cdf(d1) - fwd2 * norm.cdf(d2)


def better_of_analytical(S1, S2, sigma1, sigma2, rho, T, q1=0.0, q2=0.0):
    """Closed-form best-of-two (no strike) price, via the identity
    max(S1, S2) = S2 + max(S1 - S2, 0): a forward on S2 plus a Margrabe option.
    """
    exch = margrabe_price(S1, S2, sigma1, sigma2, rho, T, q1, q2)
    return S2 * np.exp(-q2 * T) + exch


def worst_of_analytical(S1, S2, sigma1, sigma2, rho, T, q1=0.0, q2=0.0):
    """Closed-form worst-of-two price, via min(S1,S2) = S1 + S2 - max(S1,S2)."""
    best = better_of_analytical(S1, S2, sigma1, sigma2, rho, T, q1, q2)
    return S1 * np.exp(-q1 * T) + S2 * np.exp(-q2 * T) - best


def kirk_spread_approx(S1, S2, sigma1, sigma2, rho, K, r, T, q1=0.0, q2=0.0, option="call"):
    """Kirk's (1995) analytical approximation for a spread option with a
    nonzero strike: max(S1 - S2 - K, 0) for a call. There is no exact
    closed form for spread options once K != 0 (this is the standard
    desk-level approximation used for commodity- and interest-rate-spread
    options in practice); it is exact only in the K -> 0 limit, where it
    collapses to the Margrabe formula.
    """
    F1 = S1 * np.exp((r - q1) * T)
    F2 = S2 * np.exp((r - q2) * T)
    a = F2 / (F2 + K)
    sigma_k = np.sqrt(max(sigma1 ** 2 + (a * sigma2) ** 2 - 2 * rho * sigma1 * a * sigma2, 0.0))
    if sigma_k <= 1e-12 or T <= 0:
        call = np.exp(-r * T) * max(F1 - (F2 + K), 0.0)
    else:
        d1 = (np.log(F1 / (F2 + K)) + 0.5 * sigma_k ** 2 * T) / (sigma_k * np.sqrt(T))
        d2 = d1 - sigma_k * np.sqrt(T)
        call = np.exp(-r * T) * (F1 * norm.cdf(d1) - (F2 + K) * norm.cdf(d2))
    if option == "call":
        return call
    if option == "put":
        # put-call parity for the spread: Call - Put = disc * (F1 - F2 - K)
        return call - np.exp(-r * T) * (F1 - F2 - K)
    raise ValueError("option must be 'call' or 'put'")


# ---------------------------------------------------------------------------
# Two-asset correlated binomial tree (European & American)
# ---------------------------------------------------------------------------

def two_asset_binomial(S1_0, S2_0, sigma1, sigma2, rho, r, T, n_steps,
                        payoff_fn, american=False, q1=0.0, q2=0.0):
    """Correlated two-asset binomial tree (Boyle 1988 multi-asset extension
    of CRR). At every step there are four branches (uu, ud, du, dd) whose
    probabilities are chosen to match the first two moments and the
    correlation of the bivariate lognormal diffusion. Backward induction
    against `payoff_fn` supports American exercise.

    payoff_fn: vectorized callable(S1_grid, S2_grid) -> payoff_grid, e.g. one
               built with payoffs.make_binomial_payoff(...).
    Returns the price at t = 0.
    """
    dt = T / n_steps
    u1, dn1 = np.exp(sigma1 * np.sqrt(dt)), np.exp(-sigma1 * np.sqrt(dt))
    u2, dn2 = np.exp(sigma2 * np.sqrt(dt)), np.exp(-sigma2 * np.sqrt(dt))
    nu1 = (r - q1) - 0.5 * sigma1 ** 2
    nu2 = (r - q2) - 0.5 * sigma2 ** 2

    p_uu = 0.25 * (1 + np.sqrt(dt) * (nu1 / sigma1 + nu2 / sigma2) + rho)
    p_ud = 0.25 * (1 + np.sqrt(dt) * (nu1 / sigma1 - nu2 / sigma2) - rho)
    p_du = 0.25 * (1 - np.sqrt(dt) * (nu1 / sigma1 - nu2 / sigma2) - rho)
    p_dd = 0.25 * (1 - np.sqrt(dt) * (nu1 / sigma1 + nu2 / sigma2) + rho)

    probs = np.array([p_uu, p_ud, p_du, p_dd])
    if probs.min() < 0:
        # Moment-matching probabilities can stray slightly outside [0, 1] for
        # large |rho| combined with a large dt; clip and renormalise as a
        # practical fix -- accuracy improves as n_steps grows (dt shrinks).
        probs = np.clip(probs, 0.0, None)
        probs = probs / probs.sum()
    p_uu, p_ud, p_du, p_dd = probs
    disc = np.exp(-r * dt)

    n = n_steps
    i_idx = np.arange(n + 1)
    I, J = np.meshgrid(i_idx, i_idx, indexing="ij")
    S1_n = S1_0 * (u1 ** I) * (dn1 ** (n - I))
    S2_n = S2_0 * (u2 ** J) * (dn2 ** (n - J))
    values = payoff_fn(S1_n, S2_n)

    for step in range(n - 1, -1, -1):
        cont = disc * (
            p_uu * values[1:step + 2, 1:step + 2]
            + p_ud * values[1:step + 2, 0:step + 1]
            + p_du * values[0:step + 1, 1:step + 2]
            + p_dd * values[0:step + 1, 0:step + 1]
        )
        if american:
            idx = np.arange(step + 1)
            Is, Js = np.meshgrid(idx, idx, indexing="ij")
            S1_t = S1_0 * (u1 ** Is) * (dn1 ** (step - Is))
            S2_t = S2_0 * (u2 ** Js) * (dn2 ** (step - Js))
            values = np.maximum(cont, payoff_fn(S1_t, S2_t))
        else:
            values = cont

    return float(values[0, 0])


# ---------------------------------------------------------------------------
# General Monte Carlo driver
# ---------------------------------------------------------------------------

def monte_carlo_price(option_type, model, params, n_paths, n_steps, rng,
                       antithetic=False, control_variate=False):
    """General-purpose Monte Carlo pricer tying together any asset model
    from models.py with any payoff from payoffs.py.

    option_type : 'spread' | 'margrabe' | 'better_of' | 'worst_of' | 'basket'
    model       : 'gbm_exact' | 'gbm_euler' | 'gbm_milstein' | 'skew_gbm'
                  | 'mean_reverting' | 'jump_diffusion'
    params      : dict, see app.py for the exact fields each model/payoff needs.

    Returns a dict with: price, std_error, ci_low, ci_high,
    payoffs_discounted (array), elapsed_seconds.
    """
    t0 = time.time()
    r, T = params["r"], params["T"]
    K = params.get("K", 0.0)
    option_sub = params.get("option_sub", "call")
    S_T = None  # populated for asset-based models; used by the control variate

    if model == "mean_reverting":
        S0 = np.asarray(params["S0"], dtype=float)
        X0 = S0[0] - S0[1]
        paths = M.simulate_mean_reverting_spread(
            X0, params["kappa"], params["theta"], params["sigma_x"], T,
            n_paths, n_steps, rng, antithetic,
        )
        spread_T = paths[:, -1]
        if option_type != "spread":
            raise ValueError("The mean_reverting model only supports option_type='spread'")
        payoff = np.maximum(spread_T - K, 0.0) if option_sub == "call" else np.maximum(K - spread_T, 0.0)
    else:
        S0 = np.asarray(params["S0"], dtype=float)
        mu = np.asarray(params["mu"], dtype=float)
        sigma = np.asarray(params["sigma"], dtype=float)
        rho = params["rho"]

        if model == "gbm_exact":
            S_T = M.simulate_exact_gbm(S0, mu, sigma, rho, T, n_paths, rng, antithetic)
        elif model in ("gbm_euler", "gbm_milstein"):
            scheme = "euler" if model == "gbm_euler" else "milstein"
            path = M.simulate_gbm_path(S0, mu, sigma, rho, T, n_paths, n_steps, rng, scheme, antithetic)
            S_T = path[:, -1, :]
        elif model == "skew_gbm":
            path = M.simulate_skew_correlated_gbm(
                S0, mu, sigma, rho, params.get("skew", 0.3), T, n_paths, n_steps, rng, antithetic
            )
            S_T = path[:, -1, :]
        elif model == "jump_diffusion":
            path = M.simulate_jump_diffusion(
                S0, mu, sigma, rho, params["jump_intensity"], params["jump_mean"], params["jump_std"],
                T, n_paths, n_steps, rng, antithetic,
            )
            S_T = path[:, -1, :]
        else:
            raise ValueError(f"Unknown model: {model!r}")

        if option_type == "spread":
            payoff = P.spread_payoff(S_T, K, option_sub)
        elif option_type == "margrabe":
            payoff = P.margrabe_payoff(S_T)
        elif option_type == "better_of":
            payoff = P.better_of_payoff(S_T, K)
        elif option_type == "worst_of":
            payoff = P.worst_of_payoff(S_T, K)
        elif option_type == "basket":
            payoff = P.basket_payoff(S_T, params["weights"], K, option_sub)
        else:
            raise ValueError(f"Unknown option_type: {option_type!r}")

    discounted = np.exp(-r * T) * payoff

    if control_variate and S_T is not None:
        # Use asset 1's discounted terminal value as a control variate: its
        # expectation under the simulation drift is known in closed form, so
        # any simulated deviation from that known mean is pure noise we can
        # subtract off (Boyle's control variate technique).
        control = np.exp(-r * T) * S_T[:, 0]
        true_mean = S0[0] * np.exp((mu[0] - r) * T)
        var_c = np.var(control)
        if var_c > 1e-12:
            b = np.cov(discounted, control)[0, 1] / var_c
            discounted = discounted - b * (control - true_mean)

    price = float(np.mean(discounted))
    n = len(discounted)
    se = float(np.std(discounted, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return {
        "price": price,
        "std_error": se,
        "ci_low": price - 1.96 * se,
        "ci_high": price + 1.96 * se,
        "payoffs_discounted": discounted,
        "elapsed_seconds": time.time() - t0,
    }
