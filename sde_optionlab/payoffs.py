"""Payoff functions for spread, exchange (Margrabe), better-of/worst-of, and
basket options, used by the Monte Carlo engine in pricers.py.

All functions expect terminal asset values `S` with shape (n_paths, n_assets),
except the binomial-tree helpers at the bottom which operate on 2-D grids.
"""

import numpy as np


def spread_payoff(S, K, option="call"):
    """S: terminal values, shape (n_paths, 2). K: strike applied to S1 - S2."""
    spread = S[:, 0] - S[:, 1]
    if option == "call":
        return np.maximum(spread - K, 0.0)
    if option == "put":
        return np.maximum(K - spread, 0.0)
    raise ValueError("option must be 'call' or 'put'")


def margrabe_payoff(S):
    """Exchange-option payoff: max(S1 - S2, 0). Equivalent to spread_payoff with K=0."""
    return np.maximum(S[:, 0] - S[:, 1], 0.0)


def better_of_payoff(S, K=0.0):
    """Best-of payoff on the max across all assets, with optional strike: max(max(S) - K, 0)."""
    return np.maximum(np.max(S, axis=1) - K, 0.0)


def worst_of_payoff(S, K=0.0):
    """Worst-of payoff on the min across all assets, with optional strike: max(min(S) - K, 0)."""
    return np.maximum(np.min(S, axis=1) - K, 0.0)


def basket_payoff(S, weights, K, option="call"):
    """Weighted-average basket option payoff. weights: array-like summing to 1."""
    basket = S @ np.asarray(weights, dtype=float)
    if option == "call":
        return np.maximum(basket - K, 0.0)
    if option == "put":
        return np.maximum(K - basket, 0.0)
    raise ValueError("option must be 'call' or 'put'")


PAYOFFS = {
    "spread": spread_payoff,
    "margrabe": margrabe_payoff,
    "better_of": better_of_payoff,
    "worst_of": worst_of_payoff,
    "basket": basket_payoff,
}


def make_binomial_payoff(option_type, K=0.0, option_sub="call"):
    """Return a vectorized payoff(S1_grid, S2_grid) -> payoff_grid callable for
    use in the two-asset binomial tree (pricers.two_asset_binomial), matching
    the same option_type naming used by the Monte Carlo engine above.
    """
    if option_type == "spread":
        if option_sub == "call":
            return lambda s1, s2: np.maximum((s1 - s2) - K, 0.0)
        return lambda s1, s2: np.maximum(K - (s1 - s2), 0.0)
    if option_type == "margrabe":
        return lambda s1, s2: np.maximum(s1 - s2, 0.0)
    if option_type == "better_of":
        return lambda s1, s2: np.maximum(np.maximum(s1, s2) - K, 0.0)
    if option_type == "worst_of":
        return lambda s1, s2: np.maximum(np.minimum(s1, s2) - K, 0.0)
    raise ValueError(
        f"Binomial tree supports 'spread'/'margrabe'/'better_of'/'worst_of' only "
        f"(got {option_type!r}); basket options need >2 assets and use Monte Carlo."
    )
