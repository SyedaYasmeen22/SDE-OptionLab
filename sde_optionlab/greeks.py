"""Greeks via bumped Monte Carlo with common random numbers, plus simple
VaR/CVaR risk metrics on the simulated payoff distribution.
"""

import copy
import numpy as np

from .pricers import monte_carlo_price


def _bumped_price(option_type, model, params, n_paths, n_steps, seed, asset_index, bump, mc_kwargs):
    p = copy.deepcopy(params)
    S0 = np.array(p["S0"], dtype=float)
    S0[asset_index] += bump
    p["S0"] = S0
    rng = np.random.default_rng(seed)  # same seed => same underlying random draws (common random numbers)
    return monte_carlo_price(option_type, model, p, n_paths, n_steps, rng, **mc_kwargs)["price"]


def deltas(option_type, model, params, n_paths, n_steps, seed=12345, bump_frac=0.01, **mc_kwargs):
    """Central finite-difference delta with respect to each asset's spot,
    re-using the same random seed for the up/down bump (common random
    numbers) so Monte Carlo noise cancels rather than swamping the bump.

    Returns a dict like {'delta_1': ..., 'delta_2': ..., 'hedge_ratio': ...}.
    hedge_ratio (only for 2-asset cases) is the number of units of asset 2
    needed per unit of asset 1 to delta-hedge the position.
    """
    S0 = np.asarray(params["S0"], dtype=float)
    out = {}
    for i in range(len(S0)):
        h = max(S0[i] * bump_frac, 1e-6)
        up = _bumped_price(option_type, model, params, n_paths, n_steps, seed, i, +h, mc_kwargs)
        down = _bumped_price(option_type, model, params, n_paths, n_steps, seed, i, -h, mc_kwargs)
        out[f"delta_{i + 1}"] = (up - down) / (2 * h)
    if len(S0) == 2 and abs(out["delta_1"]) > 1e-10:
        out["hedge_ratio"] = -out["delta_2"] / out["delta_1"]
    return out


def value_at_risk(discounted_payoffs, current_price, confidence=0.95):
    """Simple historical VaR / CVaR on the simulated discounted-payoff
    distribution, expressed as a loss relative to the current option value
    (positive numbers = a loss). This re-uses the same simulated paths
    already drawn for pricing -- no extra simulation needed.
    """
    pnl = np.asarray(discounted_payoffs) - current_price
    var_level = np.percentile(pnl, (1 - confidence) * 100)
    tail = pnl[pnl <= var_level]
    cvar = tail.mean() if tail.size > 0 else var_level
    return {
        "VaR": float(-var_level),
        "CVaR": float(-cvar),
        "confidence": confidence,
    }
