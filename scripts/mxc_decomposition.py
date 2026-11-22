"""
M x C covariance decomposition for the SPMO momentum puzzle study.

Identity (exact, per month t):
    R_tilt,b = R_b + Cov_b(z, r),   z_i = s_i / sum_j b_j s_j
             = R_b + rho_b(z, r) * sd_b(z) * sd_b(r)

    b    : base weights over the full S&P 500 universe (cap or equal),
           drifted with returns from the same rebalance date as the tilt
    s    : tilt score fixed at rebalance (max(x, 0) in the top quintile, 0 otherwise)
    r    : stock return in month t
    Cov_b, sd_b, rho_b : cross-sectional moments computed with weights b

    M x C (month t) = Cov_C(z_C, r) - Cov_E(z_E, r)

The product rho * sd(z) * sd(r) is attributed to its three factors with an
exact Shapley split, so the three contributions add up to M x C every month.

Input panel (one row per stock-month), columns:
    date   : holding month (month-end)
    s      : tilt score, fixed over the holding period (0 if not selected)
    b_cap  : drifted cap weight within the universe for that month (sums to 1 per date)
    b_eq   : drifted equal weight within the universe for that month (sums to 1 per date)
    r      : total return in that month (incl. delisting)

Use the SAME convention for the base portfolios as in Section 3 (drifting from the
same rebalance date); if 000 or 010 are rebalanced monthly in the original code,
build b_eq / b_cap the same way, and check with `check_identity`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# 1. Monthly cross-sectional moments
# ----------------------------------------------------------------------------
def _moments(g: pd.DataFrame, bcol: str) -> pd.Series:
    b = g[bcol].to_numpy(float)
    b = b / b.sum()
    s = g["s"].to_numpy(float)
    r = g["r"].to_numpy(float)

    s_bar = (b * s).sum()                 # base-weighted mean score
    z = s / s_bar                         # normalised score, base-weighted mean = 1
    w = b * z                             # tilted weights (sum to 1)

    r_bar = (b * r).sum()                 # base portfolio return R_b
    dz, dr = z - 1.0, r - r_bar
    cov = (b * dz * dr).sum()
    sd_z = np.sqrt((b * dz**2).sum())
    sd_r = np.sqrt((b * dr**2).sum())
    rho = cov / (sd_z * sd_r) if sd_z > 0 and sd_r > 0 else np.nan

    return pd.Series({
        "R_base": r_bar,
        "R_tilt": (w * r).sum(),
        "cov": cov,
        "rho": rho,          # base-weighted IC between score and return
        "sd_z": sd_z,        # concentration of the tilt (coefficient of variation of s)
        "sd_r": sd_r,        # base-weighted cross-sectional return dispersion
        "eff_n": 1.0 / (w**2).sum(),
    })


def monthly_components(panel: pd.DataFrame) -> pd.DataFrame:
    """Return one row per month with cap (C) and equal (E) components and M x C."""
    cap = panel.groupby("date").apply(_moments, bcol="b_cap").add_suffix("_C")
    eq = panel.groupby("date").apply(_moments, bcol="b_eq").add_suffix("_E")
    out = cap.join(eq)
    out["MxC"] = out["cov_C"] - out["cov_E"]
    out = out.join(shapley_split(out))
    return out


# ----------------------------------------------------------------------------
# 2. Exact Shapley split of  a_C b_C c_C - a_E b_E c_E
# ----------------------------------------------------------------------------
def shapley_split(m: pd.DataFrame) -> pd.DataFrame:
    a = (m["rho_E"], m["rho_C"])
    b = (m["sd_z_E"], m["sd_z_C"])
    c = (m["sd_r_E"], m["sd_r_C"])

    def phi(x, y, z):
        dx = x[1] - x[0]
        return dx * ((y[0] * z[0] + y[1] * z[1]) / 3 + (y[1] * z[0] + y[0] * z[1]) / 6)

    return pd.DataFrame({
        "contrib_IC": phi(a, b, c),
        "contrib_concentration": phi(b, a, c),
        "contrib_dispersion": phi(c, a, b),
    })


# ----------------------------------------------------------------------------
# 3. Checks and summary
# ----------------------------------------------------------------------------
def check_identity(m: pd.DataFrame, tol: float = 1e-12) -> None:
    if m.empty:
        raise ValueError("No monthly components to validate")
    for k in ("C", "E"):
        gaps = (m[f"R_tilt_{k}"] - m[f"R_base_{k}"] - m[f"cov_{k}"]).abs()
        invalid = ~(gaps.to_numpy(dtype=float, na_value=np.nan) < tol)
        if invalid.any():
            raise ValueError(f"Covariance identity fails for {k} in {gaps.index[invalid][0]}")
    fields = ["contrib_IC", "contrib_concentration", "contrib_dispersion"]
    parts = m[fields].sum(axis=1, min_count=len(fields))
    gaps = (parts - m["MxC"]).abs()
    invalid = ~(gaps.to_numpy(dtype=float, na_value=np.nan) < tol)
    if invalid.any():
        raise ValueError(f"Shapley parts do not add up in {gaps.index[invalid][0]}")


def nw_tstat(x: pd.Series, lags: int = 6) -> float:
    """Newey-West t-stat of the mean (Bartlett kernel)."""
    x = x.dropna().to_numpy(float)
    n = len(x)
    e = x - x.mean()
    lrv = (e @ e) / n
    for L in range(1, lags + 1):
        lrv += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return x.mean() / np.sqrt(lrv / n)


def summary(m: pd.DataFrame, lags: int = 6) -> pd.DataFrame:
    cols = ["MxC", "contrib_IC", "contrib_concentration", "contrib_dispersion"]
    rows = []
    for c in cols:
        rows.append({"component": c,
                     "mean_pa_%": 12 * 100 * m[c].mean(),
                     "NW_t": nw_tstat(m[c], lags)})
    levels = m[["rho_C", "rho_E", "sd_z_C", "sd_z_E", "sd_r_C", "sd_r_E"]].mean()
    return pd.DataFrame(rows).set_index("component"), levels


# ----------------------------------------------------------------------------
# 4. Chart (12-month rolling, in % per year)
# ----------------------------------------------------------------------------
def plot_rolling(m: pd.DataFrame, path: str = "mxc_decomposition.png", window: int = 12):
    import matplotlib.pyplot as plt

    roll = m[["MxC", "contrib_IC", "contrib_concentration", "contrib_dispersion"]] \
        .rolling(window).sum() * 100
    fig, ax = plt.subplots(figsize=(10, 4.5))
    labels = {"contrib_IC": "IC (rho)", "contrib_concentration": "Concentration (sd z)",
              "contrib_dispersion": "Return dispersion (sd r)"}
    colors = {"contrib_IC": "#1f77b4", "contrib_concentration": "#ff7f0e",
              "contrib_dispersion": "#2ca02c"}
    for c, lab in labels.items():
        ax.plot(roll.index, roll[c], color=colors[c], lw=1.0, label=lab)
    ax.plot(roll.index, roll["MxC"], color="black", lw=1.8, label="M x C (sum of the three)")
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_ylabel(f"Trailing {window}-month sum, % ")
    ax.set_xlabel("")
    ax.set_title("Where does the momentum x market-cap interaction come from?")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    return fig


# ----------------------------------------------------------------------------
# 5. Self-test on synthetic data
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    rows = []
    for t in pd.date_range("2000-01-31", periods=120, freq="ME"):
        n = 500
        cap = rng.lognormal(10, 1.5, n)
        x = rng.normal(0, 1, n)
        s = np.where(x >= np.quantile(x, 0.8), np.maximum(x, 0), 0.0)
        r = 0.01 + 0.004 * x + 0.002 * x * (np.log(cap) - 10) / 1.5 + rng.normal(0, 0.08, n)
        rows.append(pd.DataFrame({"date": t, "s": s, "b_cap": cap / cap.sum(),
                                  "b_eq": 1 / n, "r": r}))
    panel = pd.concat(rows, ignore_index=True)

    m = monthly_components(panel)
    check_identity(m)
    tab, levels = summary(m)
    print(tab.round(2))
    print(levels.round(3))
    plot_rolling(m, "synthetic_test.png")
    print("identity and Shapley checks passed")
