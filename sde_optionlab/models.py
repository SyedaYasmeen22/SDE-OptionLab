"""
Asset dynamics simulators for SDE-OptionLab.

Each simulator advances one or more correlated stochastic differential
equations and returns either terminal values only (fast, for European-style
payoffs with no path dependency) or full paths (needed for American exercise
checks or path-dependent diagnostics).

Shapes:
    S0, mu, sigma : 1-D arrays of length n_assets
    terminal values : (n_paths, n_assets)
    full paths      : (n_paths, n_steps + 1, n_assets)
"""

import numpy as np


def _cholesky_corr(rho, n_assets=2):
    """Build the Cholesky factor of a correlation matrix.

    For n_assets == 2, `rho` is a scalar correlation in (-1, 1).
    For n_assets > 2, `rho` must already be an (n_assets, n_assets)
    correlation matrix.
    """
    if n_assets == 2 and np.isscalar(rho):
        corr = np.array([[1.0, rho], [rho, 1.0]])
    else:
        corr = np.asarray(rho, dtype=float)
        if corr.shape != (n_assets, n_assets):
            raise ValueError(
                "rho must be an (n_assets, n_assets) correlation matrix when n_assets > 2"
            )
    try:
        return np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        # Correlation matrix supplied isn't quite PSD (e.g. from noisy
        # historical estimation) -- project to the nearest PSD matrix.
        eigvals, eigvecs = np.linalg.eigh(corr)
        eigvals = np.clip(eigvals, 1e-10, None)
        fixed = (eigvecs * eigvals) @ eigvecs.T
        d = np.sqrt(np.diag(fixed))
        fixed = fixed / d[:, None] / d[None, :]
        return np.linalg.cholesky(fixed)


def _draw_normals(n_paths, n_assets, rng, antithetic, half_cache=None):
    """Draw a (n_paths, n_assets) block of iid standard normals, optionally
    using antithetic pairing (mirrored sign) to halve Monte Carlo variance
    for a given number of random draws.
    """
    if antithetic:
        half = (n_paths + 1) // 2
        z = rng.standard_normal((half, n_assets))
        return np.vstack([z, -z])[:n_paths]
    return rng.standard_normal((n_paths, n_assets))


def correlated_normals(n_paths, n_assets, rho, rng, antithetic=False):
    """Draw correlated standard normal shocks, shape (n_paths, n_assets)."""
    L = _cholesky_corr(rho, n_assets)
    z = _draw_normals(n_paths, n_assets, rng, antithetic)
    return z @ L.T


def simulate_exact_gbm(S0, mu, sigma, rho, T, n_paths, rng, antithetic=False):
    """Exact terminal-value simulation of correlated GBM -- samples directly
    from the (multivariate) lognormal terminal distribution, so there is
    zero discretization error. Fastest option; only valid for European-style
    (no path-dependency) payoffs.
    """
    S0, mu, sigma = np.asarray(S0, float), np.asarray(mu, float), np.asarray(sigma, float)
    n_assets = len(S0)
    z = correlated_normals(n_paths, n_assets, rho, rng, antithetic)
    drift = (mu - 0.5 * sigma ** 2) * T
    diffusion = sigma * np.sqrt(T) * z
    return S0 * np.exp(drift + diffusion)


def simulate_gbm_path(S0, mu, sigma, rho, T, n_paths, n_steps, rng, scheme="euler", antithetic=False):
    """Discretized correlated GBM path simulation.

    scheme: 'euler' (Euler-Maruyama applied to the SDE directly, so it carries
            genuine discretization bias vs. the exact lognormal) or
            'milstein' (adds the second-order correction term).
    Returns full paths, shape (n_paths, n_steps + 1, n_assets).
    """
    S0, mu, sigma = np.asarray(S0, float), np.asarray(mu, float), np.asarray(sigma, float)
    n_assets = len(S0)
    dt = T / n_steps
    L = _cholesky_corr(rho, n_assets)

    paths = np.empty((n_paths, n_steps + 1, n_assets))
    paths[:, 0, :] = S0

    for step in range(n_steps):
        z = _draw_normals(n_paths, n_assets, rng, antithetic)
        dW = (z @ L.T) * np.sqrt(dt)
        S_t = paths[:, step, :]
        if scheme == "euler":
            S_next = S_t + mu * S_t * dt + sigma * S_t * dW
        elif scheme == "milstein":
            S_next = (
                S_t + mu * S_t * dt + sigma * S_t * dW
                + 0.5 * sigma ** 2 * S_t * (dW ** 2 - dt)
            )
        else:
            raise ValueError(f"Unknown scheme: {scheme!r}")
        paths[:, step + 1, :] = np.maximum(S_next, 1e-8)
    return paths


def simulate_skew_correlated_gbm(S0, mu, sigma, rho, skew, T, n_paths, n_steps, rng, antithetic=False):
    """A simulation-friendly proxy for asymmetric dependence between two
    assets, in the spirit of skew-correlated Brownian motion (Pasricha & He):
    correlation increases on joint down-moves relative to up-moves, which is
    the stylised "downside contagion" fact that family of models targets.

    NOTE: this is a practical Euler-scheme approximation (state-dependent
    correlation), not the closed-form Pasricha-He process itself. It is
    intended for stress-testing common downside shocks between two assets.

    skew : float in [0, 1). 0 reduces to standard correlated GBM (Euler).
    """
    S0, mu, sigma = np.asarray(S0, float), np.asarray(mu, float), np.asarray(sigma, float)
    if len(S0) != 2:
        raise ValueError("Skew correlated model currently supports exactly 2 assets")
    dt = T / n_steps
    paths = np.empty((n_paths, n_steps + 1, 2))
    paths[:, 0, :] = S0

    for step in range(n_steps):
        z1 = _draw_normals(n_paths, 1, rng, antithetic)[:, 0]
        eps = rng.standard_normal(n_paths)
        down = z1 < 0
        rho_eff = np.where(down, np.clip(rho + skew * (1 - abs(rho)), -0.999, 0.999), rho)
        z2 = rho_eff * z1 + np.sqrt(np.clip(1 - rho_eff ** 2, 1e-12, None)) * eps
        dW1 = z1 * np.sqrt(dt)
        dW2 = z2 * np.sqrt(dt)
        S_t = paths[:, step, :]
        s1_next = S_t[:, 0] + mu[0] * S_t[:, 0] * dt + sigma[0] * S_t[:, 0] * dW1
        s2_next = S_t[:, 1] + mu[1] * S_t[:, 1] * dt + sigma[1] * S_t[:, 1] * dW2
        paths[:, step + 1, 0] = np.maximum(s1_next, 1e-8)
        paths[:, step + 1, 1] = np.maximum(s2_next, 1e-8)
    return paths


def simulate_mean_reverting_spread(X0, kappa, theta, sigma_x, T, n_paths, n_steps, rng, antithetic=False):
    """Simulate a mean-reverting (Ornstein-Uhlenbeck) spread process directly:

        dX = kappa * (theta - X) dt + sigma_x dW

    Useful for longer-dated spread options where the spread itself, not the
    two individual asset prices, is the natural state variable -- the thesis
    notes this regime matters once maturities run much past ~90 days.
    Returns full paths, shape (n_paths, n_steps + 1).
    """
    dt = T / n_steps
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = X0
    for step in range(n_steps):
        z = _draw_normals(n_paths, 1, rng, antithetic)[:, 0]
        dW = z * np.sqrt(dt)
        X_t = paths[:, step]
        paths[:, step + 1] = X_t + kappa * (theta - X_t) * dt + sigma_x * dW
    return paths


def simulate_jump_diffusion(S0, mu, sigma, rho, jump_intensity, jump_mean, jump_std,
                             T, n_paths, n_steps, rng, antithetic=False):
    """Merton-style jump-diffusion: correlated diffusive shocks plus
    independent compound-Poisson jumps per asset (lognormal jump sizes).
    Useful for stress-testing / fat-tail scenarios on spread and
    better-of payoffs.
    Returns full paths, shape (n_paths, n_steps + 1, n_assets).
    """
    S0, mu, sigma = np.asarray(S0, float), np.asarray(mu, float), np.asarray(sigma, float)
    n_assets = len(S0)
    dt = T / n_steps
    L = _cholesky_corr(rho, n_assets)
    paths = np.empty((n_paths, n_steps + 1, n_assets))
    paths[:, 0, :] = S0
    # Compensate the drift so the jump component doesn't introduce a bias
    # (standard Merton drift correction: E[jump multiplier] - 1).
    jump_comp = jump_intensity * (np.exp(jump_mean + 0.5 * jump_std ** 2) - 1)

    for step in range(n_steps):
        z = _draw_normals(n_paths, n_assets, rng, antithetic)
        dW = (z @ L.T) * np.sqrt(dt)
        S_t = paths[:, step, :]
        diffusive = S_t + (mu - jump_comp) * S_t * dt + sigma * S_t * dW

        n_jumps = rng.poisson(jump_intensity * dt, size=(n_paths, n_assets))
        jump_factor = np.ones((n_paths, n_assets))
        has_jump = n_jumps > 0
        if has_jump.any():
            jump_size = rng.normal(jump_mean, jump_std, size=(n_paths, n_assets))
            jump_factor = np.where(has_jump, np.exp(jump_size * n_jumps), 1.0)

        paths[:, step + 1, :] = np.maximum(diffusive * jump_factor, 1e-8)
    return paths
