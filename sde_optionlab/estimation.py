"""Parameter estimation from historical data, for the Data & Parameter module:
turns an uploaded CSV of two correlated series (stock prices, EPS, PAT, FX
rates, etc.) into the mu/sigma/rho inputs the pricing engine needs -- the
same log-return methodology used on the EPS/PAT data in the underlying thesis.
"""

import numpy as np
import pandas as pd


def estimate_params_from_prices(df, col1, col2, periods_per_year=252):
    """Estimate annualised drift (mu), volatility (sigma), and correlation
    (rho) from two columns of historical, chronologically-sorted, positive
    series using log returns.

    Returns a dict with mu (array len 2), sigma (array len 2), rho (float),
    S1_0, S2_0 (most recent observed levels), and n_obs.
    """
    log_ret = np.log(df[[col1, col2]] / df[[col1, col2]].shift(1)).dropna()
    sigma = log_ret.std().values * np.sqrt(periods_per_year)
    # log-return mean estimates (mu - 0.5*sigma^2)*dt under GBM; add back the
    # Ito correction to recover the SDE drift mu itself.
    mu = log_ret.mean().values * periods_per_year + 0.5 * sigma ** 2
    rho = float(log_ret[col1].corr(log_ret[col2]))
    return {
        "mu": mu,
        "sigma": sigma,
        "rho": rho,
        "S1_0": float(df[col1].iloc[-1]),
        "S2_0": float(df[col2].iloc[-1]),
        "n_obs": int(len(log_ret)),
    }


def rolling_window_estimates(df, col1, col2, window, periods_per_year=252):
    """Rolling-window re-estimation of mu/sigma/rho, so the model can be
    refreshed as new quarterly/annual data arrives. Returns a DataFrame with
    one row of estimates per window end-point.
    """
    log_ret = np.log(df[[col1, col2]] / df[[col1, col2]].shift(1)).dropna()
    if len(log_ret) < window:
        raise ValueError(f"Not enough observations ({len(log_ret)}) for a window of {window}")
    rows = []
    for end in range(window, len(log_ret) + 1):
        chunk = log_ret.iloc[end - window:end]
        sigma = chunk.std().values * np.sqrt(periods_per_year)
        mu = chunk.mean().values * periods_per_year + 0.5 * sigma ** 2
        rho = chunk[col1].corr(chunk[col2])
        rows.append({
            "window_end": log_ret.index[end - 1],
            "mu1": mu[0], "mu2": mu[1],
            "sigma1": sigma[0], "sigma2": sigma[1],
            "rho": rho,
        })
    return pd.DataFrame(rows)
