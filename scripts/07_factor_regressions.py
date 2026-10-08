"""
Factor regressions of the market-cap effects and size slopes.

Asks whether known factors explain C (market), C (momentum) and their interaction M x C from
scripts/04_attribution.py, and the size slopes of the momentum x size double sort from
scripts/06_double_sort.py. All dependent variables are long-short return spreads, so they are
regressed without subtracting the risk-free rate; the capped SPMO proxy, a long-only portfolio, is
included for reference as an excess return.

Specifications (monthly, in-sample Apr-1995 .. Dec-2025):
  FF5 + UMD                  Mkt-RF, SMB, HML, RMW, CMA, UMD
  + ST_Rev                   adds the short-term reversal factor (prior-month return)
  + ST_Rev + LT_Rev          adds the long-term reversal factor (returns from months -60 to -13)
Reversal factors come from the Kenneth French data library (scripts/01_fetch_wrds.py).

Standard errors are Newey-West with 6 lags: portfolios are held for six months, so monthly spreads
can be autocorrelated. Alphas are annualised (x 12).

Outputs (output/factor_regressions/)
  summary.md                    tables
  coefficients.csv              every coefficient, t-stat and R^2
  fig1_alphas.png               mean and alpha of each spread under each specification
  fig2_loadings.png             factor loadings in the full specification
  fig3_subperiod_alphas.png     fig1 for 1995-2014 and 2015-2025
  fig4_subperiod_loadings.png   fig2 for 1995-2014 and 2015-2025

Usage:
    python scripts/07_factor_regressions.py [--raw data/raw] [--out output/factor_regressions]
"""
import argparse
import importlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import PercentFormatter

proxy = importlib.import_module("03_spmo_proxy")
attribution = importlib.import_module("04_attribution")
double_sort = importlib.import_module("06_double_sort")

ROOT = proxy.ROOT
LAGS = 6
FF = ["mktrf", "smb", "hml", "rmw", "cma", "umd"]
SPECS = {"FF5 + UMD": FF, "+ ST_Rev": FF + ["st_rev"], "+ ST_Rev + LT_Rev": FF + ["st_rev", "lt_rev"]}
FACTOR_NAMES = {"mktrf": "Mkt-RF", "smb": "SMB", "hml": "HML", "rmw": "RMW", "cma": "CMA", "umd": "UMD",
                "st_rev": "ST_Rev", "lt_rev": "LT_Rev"}
SUBPERIODS = [("1995-2014", 1995, 2014), ("2015-2025", 2015, 2025)]
KEY = ["C (market)", "C (momentum)", "M x C"]


# ---------- Data ----------

def dependent_variables(raw):
    """Monthly spreads to explain (columns) and the risk-free rate."""
    r, _ = attribution.run(raw)
    formula = {name: f for _, name, f, _ in attribution.EFFECTS}
    deps = {k: attribution.evaluate(formula[k], r) for k in KEY}

    ret, _, sig = attribution.signals(raw)
    for name, col in double_sort.SIZES.items():
        cells, _, _ = double_sort.cell_returns(ret, sig, col)
        s = double_sort.size_slopes(cells)
        if name == "current":
            deps["Size slope, average (current size)"] = s["Average"]
            deps["Size slope, M5 (current size)"] = s["M5"]
        deps[f"Size slope, M5 - average ({name} size)"] = s["M5 - average"]

    ff = pd.read_parquet(raw / "ff5_factors_monthly.parquet")
    rev = pd.read_parquet(raw / "ff_reversal_monthly.parquet")
    factors = (ff.merge(rev, on="date", how="inner").assign(month=lambda d: d["date"].dt.to_period("M"))
               .set_index("month").drop(columns="date").astype(float))
    deps["SPMO proxy, excess return (reference)"] = r["111 capped"] - factors["rf"].reindex(r.index)
    return pd.DataFrame(deps), factors


# ---------- Regression ----------

def ols_nw(y, X, lags=LAGS):
    """OLS with an intercept; Newey-West (Bartlett) standard errors."""
    X = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ beta
    xe = X * e[:, None]
    S = xe.T @ xe
    for lag in range(1, lags + 1):
        g = xe[lag:].T @ xe[:-lag]
        S += (1 - lag / (lags + 1)) * (g + g.T)
    inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(inv @ S @ inv))
    return beta, beta / se, 1 - e.var() / y.var()


def regress(deps, factors, cols):
    """Coefficients and NW t-stats of every dependent variable on `cols` (alpha annualised)."""
    rows = []
    for name, y in deps.items():
        d = pd.concat([y.rename("y"), factors[cols]], axis=1, join="inner").dropna()
        beta, t, r2 = ols_nw(d["y"].to_numpy(), d[cols].to_numpy())
        row = {"dependent": name, "n": len(d), "R2": r2, "alpha": 12 * beta[0], "t(alpha)": t[0]}
        for c, b, tc in zip(cols, beta[1:], t[1:]):
            row[c], row[f"t({c})"] = b, tc
        rows.append(row)
    return pd.DataFrame(rows).set_index("dependent")


def raw_mean(deps):
    """Annualised mean with a Newey-West t-stat (regression on a constant)."""
    out = {}
    for name, y in deps.items():
        y = y.dropna().to_numpy()
        beta, t, _ = ols_nw(y, np.empty((len(y), 0)))
        out[name] = (12 * beta[0], t[0])
    return out


# ---------- Report ----------

def cell(v, t):
    return f"{v:+.2f} ({t:.1f})" if abs(v) < 10 else f"{v:+.1f} ({t:.1f})"


def spec_table(res, cols):
    out = pd.DataFrame({"Dependent": res.index,
                        "Alpha, % a year (t)": [f"{a:+.1%} ({t:.2f})" for a, t in zip(res["alpha"], res["t(alpha)"])]})
    for c in cols:
        out[FACTOR_NAMES[c]] = [cell(b, t) for b, t in zip(res[c], res[f"t({c})"])]
    out["R2"] = res["R2"].map("{:.2f}".format).to_numpy()
    return out


def alpha_table(deps, means, results):
    return proxy.md_table(pd.DataFrame({
        "Dependent": list(deps.columns),
        "Mean": [f"{m:+.1%} ({t:.2f})" for m, t in means.values()],
        **{spec: [f"{a:+.1%} ({t:.2f})" for a, t in zip(res["alpha"], res["t(alpha)"])]
           for spec, res in results.items()},
    }))


def summary(deps, means, results, sub, period):
    lines = [
        "# Factor regressions of the market-cap effects",
        "",
        f"Generated by `scripts/07_factor_regressions.py`. In-sample only: {period}, monthly. "
        "Spreads are regressed without subtracting the risk-free rate; the SPMO proxy row uses its "
        f"excess return. t-stats are Newey-West with {LAGS} lags. Reversal factors are from the "
        "Kenneth French data library; the other factors from WRDS.",
        "",
        "Dependent variables: C (market), C (momentum) and M x C as defined in "
        "[output/attribution](../attribution/summary.md); size slopes (S5 - S1) from "
        "[output/double_sort](../double_sort/summary.md).",
        "",
        "## Mean and alpha under each specification",
        "",
        "Annualised, % a year, t-stat in brackets.",
        "",
        alpha_table(deps, means, results),
        "",
    ]
    for spec, cols in SPECS.items():
        lines += [f"## Coefficients: {spec}", "", "Loadings with t-stats in brackets.", "",
                  proxy.md_table(spec_table(results[spec], cols)), ""]
    full = "+ ST_Rev + LT_Rev"
    for label, (part, sub_means, sub_results) in sub.items():
        lines += [
            f"## Subperiod {label} ({len(part)} months)",
            "",
            "Mean and alpha under each specification (% a year, t-stat):",
            "",
            alpha_table(part, sub_means, sub_results),
            "",
            f"Coefficients, FF5 + UMD {full}:",
            "",
            proxy.md_table(spec_table(sub_results[full], SPECS[full])),
            "",
        ]
    lines += [
        "## Figures",
        "",
        "![Alphas](fig1_alphas.png)",
        "",
        "![Loadings](fig2_loadings.png)",
        "",
        "![Subperiod alphas](fig3_subperiod_alphas.png)",
        "",
        "![Subperiod loadings](fig4_subperiod_loadings.png)",
        "",
    ]
    return "\n".join(lines)


# ---------- Figures ----------

FULL = "+ ST_Rev + LT_Rev"
LOADING_LIM = 0.5            # colour scale of the loading heatmaps; larger loadings (market beta) saturate
NOTE = f"Reversal factors: Kenneth French data library. Newey-West t-stats with {LAGS} lags."


def plotted(deps):
    """Dependent variables shown in figures (the lagged-size check stays in the tables) and their labels."""
    names = [n for n in deps.columns if "lagged size" not in n]
    return names, [n.replace(" (current size)", "") for n in names]


def alpha_panel(ax, names, labels, means, results):
    """Horizontal bars: mean and alpha under each specification, with 95% confidence intervals."""
    series = [("Mean", "#898781", [means[n] for n in names])]
    series += [(spec, c, list(zip(results[spec].loc[names, "alpha"], results[spec].loc[names, "t(alpha)"])))
               for spec, c in zip(SPECS, attribution.EFFECT_COLORS)]
    pos = np.arange(len(names))
    h = 0.2
    for i, (lab, c, vt) in enumerate(series):
        v = np.array([x for x, _ in vt])
        t = np.array([y for _, y in vt])
        ax.barh(pos + (i - 1.5) * h, v, h, color=c, label=lab, xerr=1.96 * np.abs(v / t),
                error_kw={"ecolor": attribution.INK_2, "elinewidth": 0.8, "capsize": 2})
    ax.set_yticks(pos, labels)
    ax.invert_yaxis()
    ax.axvline(0, color=attribution.INK_2, linewidth=0.8)
    ax.grid(axis="x"), ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("% a year; bars show 95% confidence intervals (Newey-West)")


def loading_panel(ax, names, labels, res, show_labels=True):
    """Heatmap of the full specification's loadings with t-stats."""
    cols = SPECS[FULL]
    b = res.loc[names, cols].to_numpy(dtype=float)
    t = res.loc[names, [f"t({c})" for c in cols]].to_numpy(dtype=float)
    ax.imshow(b, cmap=attribution.DIVERGING, norm=TwoSlopeNorm(0, -LOADING_LIM, LOADING_LIM), aspect="auto")
    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            ink = attribution.SURFACE if abs(b[i, j]) > 0.6 * LOADING_LIM else attribution.INK
            ax.text(j, i, f"{b[i, j]:+.2f}\n({t[i, j]:.1f})", ha="center", va="center", color=ink, fontsize=8.5,
                    fontweight="bold" if abs(t[i, j]) >= 1.96 else "normal")
    ax.set_xticks(range(len(cols)), [FACTOR_NAMES[c] for c in cols])
    ax.set_yticks(range(len(names)), labels if show_labels else [""] * len(names))
    ax.tick_params(length=0)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def figures(deps, means, results, sub, out, period, n_rebal):
    names, labels = plotted(deps)
    heat_note = f"{NOTE} Blue = positive, red = negative; colour scale capped at +/-{LOADING_LIM}."

    # 1. Full sample: mean and alphas
    fig, ax = plt.subplots(figsize=(12, 7))
    alpha_panel(ax, names, labels, means, results)
    ax.set_title("Mean and factor-adjusted alpha of each spread")
    ax.legend(loc="upper right")
    attribution.footnote(fig, period, n_rebal, NOTE)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    attribution.save(fig, out / "fig1_alphas.png")

    # 2. Full sample: loadings
    fig, ax = plt.subplots(figsize=(12, 6))
    loading_panel(ax, names, labels, results[FULL])
    ax.set_title(f"Factor loadings, FF5 + UMD {FULL} (t-stat; bold if |t| >= 1.96)")
    attribution.footnote(fig, period, n_rebal, heat_note)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    attribution.save(fig, out / "fig2_loadings.png")

    # 3. Subperiods: mean and alphas
    fig, axes = plt.subplots(1, 2, figsize=(17, 7), sharex=True, sharey=True)
    for ax, (label, (part, sub_means, sub_results)), letter in zip(axes, sub.items(), "ab"):
        alpha_panel(ax, names, labels, sub_means, sub_results)
        ax.set_title(f"({letter}) {label} ({len(part)} months)")
    axes[1].invert_yaxis()                                  # shared y: undo the second inversion
    axes[1].legend(loc="upper right")
    fig.suptitle("Mean and factor-adjusted alpha by subperiod", x=0.01, ha="left")
    attribution.footnote(fig, period, n_rebal, NOTE)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    attribution.save(fig, out / "fig3_subperiod_alphas.png")

    # 4. Subperiods: loadings
    fig, axes = plt.subplots(1, 2, figsize=(20, 6.5))
    for i, (ax, (label, (part, _, sub_results)), letter) in enumerate(zip(axes, sub.items(), "ab")):
        loading_panel(ax, names, labels, sub_results[FULL], show_labels=i == 0)
        ax.set_title(f"({letter}) {label} ({len(part)} months)")
    fig.suptitle(f"Factor loadings by subperiod, FF5 + UMD {FULL} (t-stat; bold if |t| >= 1.96)",
                 x=0.01, ha="left")
    attribution.footnote(fig, period, n_rebal, heat_note)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    attribution.save(fig, out / "fig4_subperiod_loadings.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--out", type=Path, default=ROOT / "output" / "factor_regressions")
    args = ap.parse_args()

    deps, factors = dependent_variables(args.raw)
    assert deps.index[0] >= attribution.IS_FIRST_REF + proxy.LAG, "out-of-sample months in the results"
    period = f"{deps.index[0].strftime('%b-%Y')} to {deps.index[-1].strftime('%b-%Y')}"
    n_rebal = len(pd.period_range(attribution.IS_FIRST_REF, deps.index[-1] - proxy.LAG, freq="M")
                  .to_series().loc[lambda s: s.dt.month.isin(proxy.REF_MONTHS)])
    means = raw_mean(deps)
    results = {spec: regress(deps, factors, cols) for spec, cols in SPECS.items()}

    sub = {}
    for label, a, b in SUBPERIODS:
        part = deps[(deps.index.year >= a) & (deps.index.year <= b)]
        sub[label] = (part, raw_mean(part), {spec: regress(part, factors, cols) for spec, cols in SPECS.items()})

    args.out.mkdir(parents=True, exist_ok=True)
    coefs = {("Full sample", spec): res for spec, res in results.items()}
    coefs.update({(label, spec): res for label, (_, _, rs) in sub.items() for spec, res in rs.items()})
    pd.concat(coefs, names=["period", "specification"]).to_csv(args.out / "coefficients.csv", float_format="%.6f")
    figures(deps, means, results, sub, args.out, period, n_rebal)
    (args.out / "summary.md").write_text(summary(deps, means, results, sub, period), encoding="utf-8")
    print(f"Months {deps.index[0]} .. {deps.index[-1]} ({len(deps)}); wrote {args.out}")


if __name__ == "__main__":
    main()
