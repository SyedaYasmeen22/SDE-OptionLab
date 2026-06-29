"""
SDE-OptionLab -- interactive pricing tool for spread and better-of options
on two correlated assets, built on the sde_optionlab engine.

Run with:
    streamlit run app.py
"""

import sys
import io
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sde_optionlab import models, payoffs, pricers, greeks, diagnostics, estimation

# ---------------------------------------------------------------------------
# Matplotlib dark theme (consistent with the app's dark UI)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor":  "#0d1b2e",
    "axes.facecolor":    "#0d1b2e",
    "axes.edgecolor":    "#2d3748",
    "axes.labelcolor":   "#a0aec0",
    "axes.titlecolor":   "#e2e8f0",
    "axes.titlesize":    11,
    "axes.titleweight":  "600",
    "axes.labelsize":    9,
    "axes.grid":         True,
    "grid.color":        "#1a2744",
    "grid.linewidth":    0.8,
    "xtick.color":       "#718096",
    "ytick.color":       "#718096",
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "text.color":        "#e2e8f0",
    "legend.facecolor":  "#0d1b2e",
    "legend.edgecolor":  "#2d3748",
    "legend.fontsize":   8,
    "lines.linewidth":   2,
    "patch.linewidth":   0,
    "figure.dpi":        110,
})

st.set_page_config(page_title="SDE-OptionLab", page_icon="📈", layout="wide")

MODEL_LABELS = {
    "gbm_exact": "Correlated GBM — exact terminal simulation",
    "gbm_euler": "Correlated GBM — Euler–Maruyama",
    "gbm_milstein": "Correlated GBM — Milstein",
    "skew_gbm": "Skew-correlated GBM (downside-contagion proxy)",
    "jump_diffusion": "Jump-diffusion (Merton-style)",
    "mean_reverting": "Mean-reverting spread (Ornstein–Uhlenbeck)",
}
OPTION_LABELS = {
    "spread": "Spread (S1 − S2)",
    "margrabe": "Margrabe exchange option (K = 0)",
    "better_of": "Better-of (max of assets)",
    "worst_of": "Worst-of (min of assets)",
    "basket": "Basket (weighted average)",
}
ENGINE_LABELS = {
    "monte_carlo": "Monte Carlo",
    "binomial_tree": "Correlated binomial tree (Boyle/CRR)",
    "analytical": "Analytical closed form",
}

# ---------------------------------------------------------------------------
# Sidebar -- Option type & model (outside the form so the UI updates live)
# ---------------------------------------------------------------------------

st.sidebar.title("📈 SDE-OptionLab")
st.sidebar.caption("Spread & better-of option pricing across multiple numerical methods")
st.sidebar.divider()

st.sidebar.markdown("### 1 · Option & model")

option_type = st.sidebar.selectbox(
    "Option type", list(OPTION_LABELS), format_func=lambda x: OPTION_LABELS[x], key="option_type"
)

if option_type in ("spread", "basket"):
    option_sub = st.sidebar.radio("Call / put", ["call", "put"], horizontal=True, key="option_sub")
else:
    option_sub = "call"
    st.sidebar.caption("Margrabe / better-of / worst-of are call-style payoffs by construction.")

model_choices = ["gbm_exact", "gbm_euler", "gbm_milstein", "skew_gbm", "jump_diffusion"]
if option_type == "spread":
    model_choices = model_choices + ["mean_reverting"]

model = st.sidebar.selectbox(
    "Asset dynamics model", model_choices, format_func=lambda m: MODEL_LABELS[m], key="model"
)

gbm_family = model in ("gbm_exact", "gbm_euler", "gbm_milstein")
two_asset_payoff = option_type in ("spread", "margrabe", "better_of", "worst_of")

engine_choices = ["monte_carlo"]
if gbm_family and two_asset_payoff:
    engine_choices.append("binomial_tree")
    engine_choices.append("analytical")

engine = st.sidebar.selectbox(
    "Numerical engine", engine_choices, format_func=lambda e: ENGINE_LABELS[e], key="engine"
)
american = False
if engine == "binomial_tree":
    american = st.sidebar.checkbox("American-style exercise", value=False, key="american")

# ---------------------------------------------------------------------------
# Sidebar -- optional CSV-driven parameter estimation
# ---------------------------------------------------------------------------

with st.sidebar.expander("📂 Estimate parameters from a CSV (optional)"):
    uploaded = st.file_uploader("Two-column price history (chronological order)", type=["csv"])
    if uploaded is not None:
        try:
            hist_df = pd.read_csv(uploaded)
            cols = list(hist_df.columns)
            c1 = st.selectbox("Asset 1 column", cols, key="col1_select")
            c2 = st.selectbox("Asset 2 column", cols, index=min(1, len(cols) - 1), key="col2_select")
            periods = st.number_input(
                "Observations per year (252 daily · 52 weekly · 12 monthly · 4 quarterly)",
                value=252, step=1, key="periods_per_year",
            )
            if st.button("Estimate & apply to inputs below"):
                est = estimation.estimate_params_from_prices(hist_df, c1, c2, periods_per_year=int(periods))
                st.session_state["S1_0"] = round(est["S1_0"], 4)
                st.session_state["S2_0"] = round(est["S2_0"], 4)
                st.session_state["mu1"] = round(float(est["mu"][0]), 4)
                st.session_state["mu2"] = round(float(est["mu"][1]), 4)
                st.session_state["sigma1"] = round(float(est["sigma"][0]), 4)
                st.session_state["sigma2"] = round(float(est["sigma"][1]), 4)
                st.session_state["rho"] = round(float(est["rho"]), 4)
                st.success(f"Estimated from {est['n_obs']} return observations — applied below.")
                st.rerun()
        except Exception as exc:
            st.error(f"Could not estimate parameters: {exc}")
    st.caption(
        "Note: volatility and correlation estimates are reasonably stable from "
        "historical data; drift (μ) estimates are statistically noisy from price "
        "history alone and are usually overridden with a view or set equal to r "
        "for risk-neutral pricing."
    )

# ---------------------------------------------------------------------------
# Sidebar -- parameter form (batched; recomputation happens only on submit)
# ---------------------------------------------------------------------------

st.sidebar.markdown("### 2 · Market data & settings")

with st.sidebar.form("pricing_form"):
    c1, c2 = st.columns(2)
    with c1:
        S1_0 = st.number_input("S1(0)", value=st.session_state.get("S1_0", 100.0), key="S1_0")
    with c2:
        S2_0 = st.number_input("S2(0)", value=st.session_state.get("S2_0", 95.0), key="S2_0")

    if model == "mean_reverting":
        st.caption("The spread X(0) = S1(0) − S2(0) is modelled directly as a mean-reverting process.")
        kappa = st.number_input("Mean-reversion speed κ", value=2.0, min_value=0.0, key="kappa")
        theta = st.number_input("Long-run mean spread θ", value=float(S1_0 - S2_0), key="theta")
        sigma_x = st.number_input("Spread volatility σ_X", value=8.0, min_value=0.0, key="sigma_x")
        mu1 = mu2 = sigma1 = sigma2 = rho = None
    else:
        c1, c2 = st.columns(2)
        with c1:
            mu1 = st.number_input("μ1 (drift, asset 1)", value=st.session_state.get("mu1", 0.05), key="mu1")
            sigma1 = st.number_input("σ1 (vol, asset 1)", value=st.session_state.get("sigma1", 0.25),
                                      min_value=0.0001, key="sigma1")
        with c2:
            mu2 = st.number_input("μ2 (drift, asset 2)", value=st.session_state.get("mu2", 0.04), key="mu2")
            sigma2 = st.number_input("σ2 (vol, asset 2)", value=st.session_state.get("sigma2", 0.20),
                                      min_value=0.0001, key="sigma2")
        rho = st.slider("ρ (correlation)", -0.99, 0.99, value=st.session_state.get("rho", 0.40), key="rho")
        st.caption("For fair-value comparisons against the analytical/binomial benchmarks, set μ1 = μ2 = r.")

    if model == "skew_gbm":
        skew = st.slider("Downside-skew intensity", 0.0, 0.95, 0.3, key="skew")
    else:
        skew = 0.3

    if model == "jump_diffusion":
        st.caption("Merton jump-diffusion parameters (applied independently to each asset):")
        jc1, jc2, jc3 = st.columns(3)
        with jc1:
            jump_intensity = st.number_input("Jump intensity λ (per yr)", value=1.0, min_value=0.0, key="jump_intensity")
        with jc2:
            jump_mean = st.number_input("Jump mean (log-size)", value=-0.05, key="jump_mean")
        with jc3:
            jump_std = st.number_input("Jump std (log-size)", value=0.10, min_value=0.0, key="jump_std")
    else:
        jump_intensity, jump_mean, jump_std = 1.0, -0.05, 0.10

    r = st.number_input("r (risk-free rate)", value=0.05, key="r")
    T = st.number_input("T (years)", value=1.0, min_value=0.0001, key="T")

    if option_type in ("spread", "basket"):
        K = st.number_input("K (strike)", value=0.0, key="K")
    else:
        K = 0.0
        st.caption("K = 0 for Margrabe / better-of / worst-of by construction.")

    weights = None
    if option_type == "basket":
        st.caption("Basket weights (need not sum to 1 — will be normalised):")
        wc1, wc2 = st.columns(2)
        with wc1:
            w1 = st.number_input("Weight, asset 1", value=0.5, key="w1")
        with wc2:
            w2 = st.number_input("Weight, asset 2", value=0.5, key="w2")
        weights = np.array([w1, w2])
        weights = weights / weights.sum()

    st.markdown("**Numerical settings**")
    nc1, nc2 = st.columns(2)
    with nc1:
        n_paths = st.number_input("Monte Carlo paths", value=50_000, min_value=1_000, step=10_000, key="n_paths")
    with nc2:
        n_steps = st.number_input("Time steps", value=100, min_value=1, step=10, key="n_steps")
    vc1, vc2 = st.columns(2)
    with vc1:
        antithetic = st.checkbox("Antithetic variates", value=True, key="antithetic")
    with vc2:
        control_variate = st.checkbox("Control variate", value=True, key="control_variate")
    seed = st.number_input("Random seed", value=42, step=1, key="seed")

    submitted = st.form_submit_button("▶ Run / update pricing", width='stretch')

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_params():
    p = {"r": r, "T": T, "K": K, "option_sub": option_sub}
    if model == "mean_reverting":
        p.update({"S0": np.array([S1_0, S2_0]), "kappa": kappa, "theta": theta, "sigma_x": sigma_x})
    else:
        p.update({
            "S0": np.array([S1_0, S2_0]), "mu": np.array([mu1, mu2]),
            "sigma": np.array([sigma1, sigma2]), "rho": rho,
        })
        if model == "skew_gbm":
            p["skew"] = skew
        if model == "jump_diffusion":
            p.update({"jump_intensity": jump_intensity, "jump_mean": jump_mean, "jump_std": jump_std})
        if option_type == "basket":
            p["weights"] = weights
    return p


def run_pricing():
    params = build_params()
    rng = np.random.default_rng(int(seed))

    result = {"params": params, "option_type": option_type, "model": model, "engine": engine}

    if engine == "monte_carlo" or option_type == "basket":
        mc = pricers.monte_carlo_price(
            option_type, model, params, int(n_paths), int(n_steps), rng,
            antithetic=antithetic, control_variate=control_variate,
        )
        result["price"] = mc["price"]
        result["std_error"] = mc["std_error"]
        result["ci"] = (mc["ci_low"], mc["ci_high"])
        result["payoffs_discounted"] = mc["payoffs_discounted"]
        result["elapsed"] = mc["elapsed_seconds"]

    elif engine == "binomial_tree":
        payoff_fn = payoffs.make_binomial_payoff(option_type, K=K, option_sub=option_sub)
        price = pricers.two_asset_binomial(
            S1_0, S2_0, sigma1, sigma2, rho, r, T, int(n_steps), payoff_fn, american=american
        )
        result["price"] = price
        result["std_error"] = None
        result["ci"] = None
        result["payoffs_discounted"] = None
        result["elapsed"] = None

    elif engine == "analytical":
        if option_type == "margrabe":
            price = pricers.margrabe_price(S1_0, S2_0, sigma1, sigma2, rho, T)
        elif option_type == "better_of":
            price = pricers.better_of_analytical(S1_0, S2_0, sigma1, sigma2, rho, T)
        elif option_type == "worst_of":
            price = pricers.worst_of_analytical(S1_0, S2_0, sigma1, sigma2, rho, T)
        elif option_type == "spread":
            price = pricers.kirk_spread_approx(S1_0, S2_0, sigma1, sigma2, rho, K, r, T, option=option_sub)
        else:
            raise ValueError("No analytical formula available for this option type.")
        result["price"] = price
        result["std_error"] = None
        result["ci"] = None
        result["payoffs_discounted"] = None
        result["elapsed"] = None

    return result


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("📈 SDE-OptionLab")
st.caption(
    "Quantitative Finance · Options Pricing Engine — "
    "pricing spread & better-of options on two correlated assets across "
    "multiple asset-dynamics models, numerical engines, and side-by-side diagnostics."
)
st.divider()

if submitted or "last_result" not in st.session_state:
    if submitted:
        with st.spinner("Pricing…"):
            st.session_state["last_result"] = run_pricing()
    else:
        with st.spinner("Pricing…"):
            st.session_state["last_result"] = run_pricing()

res = st.session_state["last_result"]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Price", f"{res['price']:.4f}")
if res["std_error"] is not None:
    m2.metric("Std. error", f"{res['std_error']:.4f}")
    m3.metric("95% CI", f"[{res['ci'][0]:.4f}, {res['ci'][1]:.4f}]")
    m4.metric("Compute time", f"{res['elapsed']*1000:.1f} ms")
else:
    m2.metric("Std. error", "—")
    m3.metric("95% CI", "—")
    m4.metric("Compute time", "—")

st.caption(
    f"{OPTION_LABELS[res['option_type']]} · {MODEL_LABELS[res['model']]} · {ENGINE_LABELS[res['engine']]}"
    + (" · American" if engine == "binomial_tree" and american else "")
)

st.markdown("""
<div style="height:0.5rem;"></div>
""", unsafe_allow_html=True)
tab_diag, tab_risk, tab_dist, tab_report = st.tabs(
    ["📊 Diagnostics", "🛡️ Greeks & risk", "📉 Payoff distribution", "📄 Report export"]
)

# ---------------------------------------------------------------------------
# Diagnostics tab
# ---------------------------------------------------------------------------
with tab_diag:
    st.subheader("Side-by-side method comparison")
    st.caption("Runs several models at the current path/step settings and compares them against a benchmark price.")

    available_models = ["gbm_exact", "gbm_euler", "gbm_milstein"]
    if option_type == "spread":
        available_models = available_models + ["skew_gbm", "jump_diffusion", "mean_reverting"]
    else:
        available_models = available_models + ["skew_gbm", "jump_diffusion"]

    chosen = st.multiselect(
        "Models to compare", available_models, default=["gbm_exact", "gbm_euler", "gbm_milstein"],
        format_func=lambda m: MODEL_LABELS[m], key="diag_models",
    )

    benchmark_price = None
    if gbm_family and two_asset_payoff:
        try:
            if option_type == "margrabe":
                benchmark_price = pricers.margrabe_price(S1_0, S2_0, sigma1, sigma2, rho, T)
            elif option_type == "better_of":
                benchmark_price = pricers.better_of_analytical(S1_0, S2_0, sigma1, sigma2, rho, T)
            elif option_type == "worst_of":
                benchmark_price = pricers.worst_of_analytical(S1_0, S2_0, sigma1, sigma2, rho, T)
            elif option_type == "spread":
                benchmark_price = pricers.kirk_spread_approx(S1_0, S2_0, sigma1, sigma2, rho, K, r, T, option=option_sub)
        except Exception:
            benchmark_price = None

    if st.button("Run comparison", key="run_comparison") and chosen:
        params = build_params()
        valid_models = [m for m in chosen if not (m == "mean_reverting" and option_type != "spread")]
        with st.spinner("Running method comparison…"):
            table = diagnostics.method_comparison_table(
                option_type, valid_models, params, int(n_paths), int(n_steps),
                benchmark=benchmark_price, seed=int(seed),
            )
        st.dataframe(table.style.format({c: "{:.4f}" for c in table.columns if c != "model"}), width='stretch')
        if benchmark_price is not None:
            st.caption(f"Benchmark (analytical / Kirk's approximation): {benchmark_price:.4f}")

        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.errorbar(
            table["model"], table["price"],
            yerr=1.96 * table["std_error"].fillna(0), fmt="o", capsize=4, color="#2563eb",
        )
        if benchmark_price is not None:
            ax.axhline(benchmark_price, color="#dc2626", linestyle="--", label="Benchmark")
            ax.legend()
        ax.set_ylabel("Price")
        ax.set_title("Method comparison (95% CI)")
        plt.xticks(rotation=20, ha="right")
        st.pyplot(fig, width='stretch')

    st.divider()
    st.subheader("Convergence study")
    st.caption("How the Monte Carlo price stabilises as the path count or step count grows.")

    conv_model = st.selectbox(
        "Model for convergence study", available_models, format_func=lambda m: MODEL_LABELS[m], key="conv_model"
    )
    conv_axis = st.radio("Vary", ["Number of paths", "Number of time steps"], horizontal=True, key="conv_axis")

    if st.button("Run convergence study", key="run_convergence"):
        params = build_params()
        with st.spinner("Running convergence study…"):
            if conv_axis == "Number of paths":
                grid = [1_000, 5_000, 20_000, 50_000, 100_000, int(n_paths)]
                grid = sorted(set(g for g in grid if g <= max(int(n_paths), 100_000)))
                conv_df = diagnostics.path_count_convergence(option_type, conv_model, params, grid, int(n_steps), seed=int(seed))
                x_col = "n_paths"
            else:
                grid = sorted(set([2, 5, 10, 25, 50, int(n_steps)]))
                conv_df = diagnostics.convergence_study(option_type, conv_model, params, int(n_paths), grid, seed=int(seed))
                x_col = "n_steps"

        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.plot(conv_df[x_col], conv_df["price"], marker="o", color="#2563eb")
        if "ci_low" in conv_df:
            ax.fill_between(conv_df[x_col], conv_df["ci_low"], conv_df["ci_high"], alpha=0.2, color="#2563eb")
        if benchmark_price is not None:
            ax.axhline(benchmark_price, color="#dc2626", linestyle="--", label="Benchmark")
            ax.legend()
        ax.set_xscale("log" if x_col == "n_paths" else "linear")
        ax.set_xlabel(x_col)
        ax.set_ylabel("Price")
        ax.set_title(f"Convergence vs. {x_col}")
        st.pyplot(fig, width='stretch')
        st.dataframe(conv_df, width='stretch')

# ---------------------------------------------------------------------------
# Greeks & risk tab
# ---------------------------------------------------------------------------
with tab_risk:
    st.subheader("Greeks (Δ) and hedge ratio")
    st.caption(
        "Computed via bumped Monte Carlo with common random numbers, independent of the engine "
        "selected above — works for every asset-dynamics model."
    )

    greek_paths = st.slider("Paths for Greeks (smaller = faster, noisier)", 5_000, 200_000, 50_000, step=5_000, key="greek_paths")

    if st.button("Compute Greeks", key="compute_greeks"):
        params = build_params()
        with st.spinner("Bumping and repricing…"):
            try:
                g = greeks.deltas(
                    option_type, model, params, greek_paths, int(n_steps), seed=int(seed),
                    antithetic=antithetic, control_variate=control_variate,
                )
                gcols = st.columns(len(g))
                for col, (k, v) in zip(gcols, g.items()):
                    col.metric(k.replace("_", " ").title(), f"{v:.4f}")
            except Exception as exc:
                st.error(f"Could not compute Greeks for this configuration: {exc}")

    st.divider()
    st.subheader("Value at Risk / Conditional VaR")
    st.caption("Computed directly from the simulated discounted-payoff distribution of the last pricing run (Monte Carlo engine only).")

    if res["payoffs_discounted"] is not None:
        conf = st.slider("Confidence level", 0.80, 0.999, 0.95, key="var_conf")
        risk = greeks.value_at_risk(res["payoffs_discounted"], res["price"], confidence=conf)
        rc1, rc2 = st.columns(2)
        rc1.metric(f"VaR ({conf:.1%})", f"{risk['VaR']:.4f}")
        rc2.metric(f"CVaR ({conf:.1%})", f"{risk['CVaR']:.4f}")
    else:
        st.info("Run the pricer with the Monte Carlo engine to see VaR / CVaR on the simulated payoff distribution.")

    st.divider()
    st.subheader("Sensitivity sliders")
    st.caption("Quick repricing (exact-GBM Monte Carlo, smaller sample) as you vary one input at a time.")

    sens_var = st.selectbox("Vary", ["sigma1", "sigma2", "rho", "r", "T"], key="sens_var")
    base_params = build_params()
    if model != "mean_reverting":
        ranges = {
            "sigma1": np.linspace(max(sigma1 * 0.3, 0.01), sigma1 * 2, 12),
            "sigma2": np.linspace(max(sigma2 * 0.3, 0.01), sigma2 * 2, 12),
            "rho": np.linspace(-0.95, 0.95, 12),
            "r": np.linspace(max(r - 0.05, -0.02), r + 0.05, 12),
            "T": np.linspace(max(T * 0.2, 0.05), T * 2, 12),
        }
        if st.button("Run sensitivity sweep", key="run_sensitivity"):
            prices = []
            with st.spinner("Sweeping…"):
                for val in ranges[sens_var]:
                    p = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in base_params.items()}
                    if sens_var == "sigma1":
                        p["sigma"] = np.array([val, sigma2])
                    elif sens_var == "sigma2":
                        p["sigma"] = np.array([sigma1, val])
                    elif sens_var == "rho":
                        p["rho"] = val
                    elif sens_var == "r":
                        p["r"] = val
                    elif sens_var == "T":
                        p["T"] = val
                    rng_s = np.random.default_rng(int(seed))
                    out = pricers.monte_carlo_price(option_type, model, p, 20_000, max(int(n_steps), 10), rng_s, antithetic=True)
                    prices.append(out["price"])
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.plot(ranges[sens_var], prices, marker="o", color="#16a34a")
            ax.set_xlabel(sens_var)
            ax.set_ylabel("Price")
            ax.set_title(f"Price sensitivity to {sens_var}")
            st.pyplot(fig, width='stretch')
    else:
        st.info("Sensitivity sweep is available for the GBM-family models; the mean-reverting spread model can be explored via the convergence study above.")

# ---------------------------------------------------------------------------
# Payoff distribution tab
# ---------------------------------------------------------------------------
with tab_dist:
    st.subheader("Simulated discounted payoff distribution")
    if res["payoffs_discounted"] is not None:
        payoffs_arr = res["payoffs_discounted"]
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.hist(payoffs_arr, bins=60, color="#2563eb", alpha=0.75)
        ax.axvline(res["price"], color="#dc2626", linestyle="--", label=f"Mean = {res['price']:.4f}")
        ax.set_xlabel("Discounted payoff")
        ax.set_ylabel("Frequency")
        ax.legend()
        st.pyplot(fig, width='stretch')

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Mean", f"{np.mean(payoffs_arr):.4f}")
        d2.metric("Std. dev.", f"{np.std(payoffs_arr):.4f}")
        d3.metric("% paths in the money", f"{(payoffs_arr > 0).mean()*100:.1f}%")
        d4.metric("Max payoff (sample)", f"{np.max(payoffs_arr):.4f}")
    else:
        st.info("Run the pricer with the Monte Carlo engine to see the simulated payoff distribution.")

# ---------------------------------------------------------------------------
# Report export tab
# ---------------------------------------------------------------------------
with tab_report:
    st.subheader("Export a summary report")
    st.caption("One-click export of the current price, assumptions, and a payoff chart — suitable for a risk-committee pack.")

    params_for_export = build_params()
    export_dict = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "option_type": option_type,
        "option_sub": option_sub,
        "model": model,
        "engine": engine,
        "american": american if engine == "binomial_tree" else None,
        "inputs": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in params_for_export.items()},
        "n_paths": int(n_paths),
        "n_steps": int(n_steps),
        "antithetic": antithetic,
        "control_variate": control_variate,
        "seed": int(seed),
        "price": res["price"],
        "std_error": res["std_error"],
        "ci_95": list(res["ci"]) if res["ci"] else None,
    }

    colA, colB = st.columns(2)
    with colA:
        st.download_button(
            "⬇ Download JSON summary",
            data=json.dumps(export_dict, indent=2, default=str),
            file_name=f"sde_optionlab_summary_{datetime.now():%Y%m%d_%H%M%S}.json",
            mime="application/json",
            width='stretch',
        )
    with colB:
        flat = {**{f"input_{k}": v for k, v in export_dict["inputs"].items()},
                "price": res["price"], "std_error": res["std_error"]}
        csv_buf = io.StringIO()
        pd.DataFrame([flat]).to_csv(csv_buf, index=False)
        st.download_button(
            "⬇ Download CSV summary",
            data=csv_buf.getvalue(),
            file_name=f"sde_optionlab_summary_{datetime.now():%Y%m%d_%H%M%S}.csv",
            mime="text/csv",
            width='stretch',
        )

    if res["payoffs_discounted"] is not None:
        st.markdown("**One-page PDF report**")
        if st.button("Generate PDF report", key="gen_pdf"):
            fig = plt.figure(figsize=(8.27, 11.69))  # A4
            fig.suptitle("SDE-OptionLab — Pricing Summary", fontsize=16, fontweight="bold")

            ax_text = fig.add_axes([0.08, 0.55, 0.84, 0.32])
            ax_text.axis("off")
            lines = [
                f"Generated: {export_dict['generated_at']}",
                f"Option type: {OPTION_LABELS[option_type]}  ({option_sub})",
                f"Asset model: {MODEL_LABELS[model]}",
                f"Engine: {ENGINE_LABELS[engine]}" + ("  [American]" if american else ""),
                "",
                f"S1(0) = {S1_0},  S2(0) = {S2_0},  K = {K},  T = {T},  r = {r}",
            ]
            if model != "mean_reverting":
                lines.append(f"σ1 = {sigma1},  σ2 = {sigma2},  ρ = {rho},  μ1 = {mu1},  μ2 = {mu2}")
            lines += [
                "",
                f"Price = {res['price']:.4f}",
            ]
            if res["std_error"] is not None:
                lines.append(f"Std. error = {res['std_error']:.4f}   95% CI = [{res['ci'][0]:.4f}, {res['ci'][1]:.4f}]")
            lines.append(f"Monte Carlo paths = {n_paths}, time steps = {n_steps}")
            ax_text.text(0, 1, "\n".join(lines), va="top", fontsize=11, family="monospace")

            ax_hist = fig.add_axes([0.1, 0.08, 0.8, 0.38])
            ax_hist.hist(res["payoffs_discounted"], bins=50, color="#2563eb", alpha=0.75)
            ax_hist.axvline(res["price"], color="#dc2626", linestyle="--")
            ax_hist.set_title("Simulated discounted payoff distribution")
            ax_hist.set_xlabel("Discounted payoff")

            pdf_buf = io.BytesIO()
            fig.savefig(pdf_buf, format="pdf")
            st.download_button(
                "⬇ Download the generated PDF",
                data=pdf_buf.getvalue(),
                file_name=f"sde_optionlab_report_{datetime.now():%Y%m%d_%H%M%S}.pdf",
                mime="application/pdf",
            )
            st.pyplot(fig, width='stretch')
    else:
        st.caption("Run the pricer with the Monte Carlo engine to enable the PDF report (it includes the payoff-distribution chart).")

st.divider()
st.caption(
    "SDE-OptionLab — built on a modular pricing engine (`sde_optionlab/`) covering correlated GBM "
    "(exact, Euler–Maruyama, Milstein), skew-correlated GBM, mean-reverting spread, and Merton "
    "jump-diffusion dynamics; Margrabe / Kirk's-approximation / better-of analytical benchmarks; a "
    "correlated two-asset binomial tree with American exercise; and Monte Carlo with antithetic and "
    "control-variate variance reduction."
)
