"""Diagnostics: side-by-side method comparison, convergence studies, and
timing benchmarks -- the live, on-demand equivalent of the thesis's static
Table 5.13 (MAE/RMSE comparison) and Figure 5.2 (convergence plot).
"""

import numpy as np
import pandas as pd

from .pricers import monte_carlo_price


def error_metrics(estimates, benchmark):
    """MAE and RMSE of a list/array of price estimates against a scalar benchmark."""
    estimates = np.asarray(estimates, dtype=float)
    errors = estimates - benchmark
    return {"MAE": float(np.mean(np.abs(errors))), "RMSE": float(np.sqrt(np.mean(errors ** 2)))}


def method_comparison_table(option_type, model_list, params, n_paths, n_steps, benchmark=None, seed=42):
    """Run several models at fixed n_paths/n_steps and return a DataFrame with
    price, standard error, 95% CI, timing, and (if a benchmark price is
    supplied) the absolute error against it.
    """
    rows = []
    for model in model_list:
        rng = np.random.default_rng(seed)
        result = monte_carlo_price(option_type, model, params, n_paths, n_steps, rng)
        row = {
            "model": model,
            "price": result["price"],
            "std_error": result["std_error"],
            "ci_low": result["ci_low"],
            "ci_high": result["ci_high"],
            "time_s": result["elapsed_seconds"],
        }
        if benchmark is not None:
            row["abs_error"] = abs(result["price"] - benchmark)
        rows.append(row)
    return pd.DataFrame(rows)


def convergence_study(option_type, model, params, n_paths, step_grid, seed=42):
    """Price across a grid of n_steps values (path-discretization
    convergence) at fixed n_paths. Returns a DataFrame: n_steps, price,
    std_error, time_s.
    """
    rows = []
    for n_steps in step_grid:
        rng = np.random.default_rng(seed)
        result = monte_carlo_price(option_type, model, params, n_paths, n_steps, rng)
        rows.append({
            "n_steps": n_steps, "price": result["price"],
            "std_error": result["std_error"], "time_s": result["elapsed_seconds"],
        })
    return pd.DataFrame(rows)


def path_count_convergence(option_type, model, params, path_grid, n_steps, seed=42):
    """Price across a grid of n_paths values (Monte Carlo sample-size
    convergence) at fixed n_steps. Returns a DataFrame: n_paths, price,
    std_error, ci_low, ci_high, time_s.
    """
    rows = []
    for n_paths in path_grid:
        rng = np.random.default_rng(seed)
        result = monte_carlo_price(option_type, model, params, n_paths, n_steps, rng)
        rows.append({
            "n_paths": n_paths, "price": result["price"], "std_error": result["std_error"],
            "ci_low": result["ci_low"], "ci_high": result["ci_high"], "time_s": result["elapsed_seconds"],
        })
    return pd.DataFrame(rows)
