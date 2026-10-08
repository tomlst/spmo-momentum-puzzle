"""
Out-of-sample evaluation: the SPMO proxy, the S&P 500 and volatility targeting with HAR and EWMA.

The design was frozen at git tag `pre-oos` before this script was run on the out-of-sample period;
every setting below is the in-sample one:

  SPMO proxy      scripts/03_spmo_proxy.py, capped version (9% and 3 x cap weight), rebalances with
                  reference months Feb-1975 .. Aug-1994, holding months Apr-1975 .. Mar-1995
  S&P 500         CRSP S&P 500 value-weighted index with dividends (vwretd), compounded monthly
  HAR             scripts/08_har_vol_target.py: log HAR on 1, 3 and 12-month realised volatility,
                  expanding window re-estimated monthly on out-of-sample data only, at least 24
                  pairs after 12 months of inputs
  EWMA (0.97)     scripts/09_vol_model_comparison.py
  Exposure        15% / forecast, no cap, the rest at the T-bill rate; no transaction or financing costs

Volatility-targeted portfolios are compared with the proxy and the S&P 500 over the months both
forecasts cover; the proxy and the S&P 500 are also compared over the whole out-of-sample period.

`--period is` runs the same code on the in-sample period and checks that it reproduces the HAR and
EWMA returns of scripts 08 and 09; it writes no output.

Outputs (output/oos/)
  summary.md, monthly.csv, fig1_performance.png, fig2_exposure.png, fig3_forecast.png,
  fig4_proxy_vs_sp500.png

Usage:
    python scripts/10_oos.py [--raw data/raw] [--out output/oos] [--period oos|is]
"""
import argparse
import importlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter

proxy = importlib.import_module("03_spmo_proxy")
attribution = importlib.import_module("04_attribution")
har = importlib.import_module("08_har_vol_target")
vmc = importlib.import_module("09_vol_model_comparison")

ROOT = proxy.ROOT
TARGET = har.TARGET
OOS_REFS = (pd.Period("1975-02", "M"), pd.Period("1994-08", "M"))
OOS_HOLD = (OOS_REFS[0] + proxy.LAG, pd.Period("1995-03", "M"))
PROXY, SP500, HAR, EWMA = "SPMO proxy", "S&P 500", "HAR vol target", f"EWMA ({vmc.EWMA_DECAY}) vol target"
COLORS = {PROXY: "#2a78d6", SP500: "#898781", HAR: "#1baf7a", EWMA: "#eda100"}
CRASH = pd.Period("1987-10", "M")


# ---------- Data ----------

def run(raw, first_ref, last_ref):
    """Monthly returns, forecasts and exposures for rebalances with reference month in [first_ref, last_ref]."""
    monthly, daily, _, _ = har.proxy_panel(raw, first_ref=first_ref, last_ref=last_ref)
    rv = har.realised_vol(daily)
    fc_har, _ = har.har_forecasts(rv)
    month_ends = list(pd.Series(daily.index, index=daily.index).groupby(daily.index.to_period("M")).last())
    fc_ewma = vmc.ewma_forecasts(daily, month_ends)

    idx = pd.read_parquet(raw / "sp500_index_daily.parquet", columns=["date", "vwretd"])
    mkt = idx.set_index("date")["vwretd"].astype(float)
    sp = np.expm1(np.log1p(mkt).groupby(mkt.index.to_period("M")).sum())
    ff = pd.read_parquet(raw / "ff5_factors_monthly.parquet", columns=["date", "rf"])
    rf = ff.set_index(ff["date"].dt.to_period("M"))["rf"].astype(float)

    full = pd.DataFrame({PROXY: monthly, SP500: sp.reindex(monthly.index), "rf": rf.reindex(monthly.index),
                         "RV": rv["RV1"].reindex(monthly.index), "RV last month": rv["RV1"].shift(1).reindex(monthly.index)})
    months = monthly.index.intersection(fc_har.index).intersection(fc_ewma.index)
    df = full.loc[months].copy()
    for name, fc in [(HAR, fc_har), (EWMA, fc_ewma)]:
        df[f"forecast {name}"] = fc.loc[months]
        df[f"w {name}"] = TARGET / df[f"forecast {name}"]
        df[name] = df[f"w {name}"] * df[PROXY] + (1 - df[f"w {name}"]) * df["rf"]
    return full, df


# ---------- Statistics ----------

def stats_table(df, names):
    rows, ex = [], {}
    for name in names:
        x = df[name]
        e = x - df["rf"]
        ex[name] = e.to_numpy()
        g = (1 + x).cumprod()
        w = df.get(f"w {name}", pd.Series(1.0, index=df.index))
        rows.append({"Portfolio": name, "CAGR": f"{(1 + x).prod() ** (12 / len(x)) - 1:.1%}",
                     "Volatility": f"{x.std() * 12 ** 0.5:.1%}", "Sharpe": f"{e.mean() / e.std() * 12 ** 0.5:.2f}",
                     "Max drawdown": f"{(g / g.cummax() - 1).min():.1%}", "Avg exposure": f"{w.mean():.2f}",
                     "Exposure range": f"{w.min():.2f}-{w.max():.2f}"})
    return pd.DataFrame(rows), ex


def z_table(ex, pairs):
    return proxy.md_table(pd.DataFrame([{"Comparison": f"{a} vs {b}", "Sharpe difference z": f"{har.jkm(ex[a], ex[b]):+.2f}"}
                                        for a, b in pairs]))


# ---------- Figures ----------

def note(fig, period):
    fig.text(0.01, 0.005, f"Out-of-sample: {period}. Design frozen at git tag pre-oos. Monthly total returns, no "
             "transaction or financing costs. Data: CRSP via WRDS.", ha="left", va="bottom",
             color=attribution.MUTED, fontsize=8.5)


def growth_panels(df, names, title, path, period):
    dates = [df.index[0].to_timestamp(how="start")] + list(df.index.to_timestamp(how="end"))
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [1.5, 1]})
    for k in names:
        g = np.concatenate([[1.0], (1 + df[k]).cumprod().to_numpy()])
        axes[0].plot(dates, g, color=COLORS[k], linewidth=2, label=f"{k}  (${g[-1]:.1f})")
        axes[1].plot(dates, g / np.maximum.accumulate(g) - 1, color=COLORS[k], linewidth=1.3)
    axes[0].set_yscale("log")
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}"))
    axes[0].set_title("(a) Growth of $1 (log scale; final value in brackets)")
    axes[0].legend(loc="upper left")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axes[1].set_title("(b) Drawdown (dashed line: October 1987)")
    for ax in axes:
        ax.axvline(CRASH.to_timestamp(how="start"), color=attribution.MUTED, linewidth=0.9, linestyle="--")
    fig.suptitle(title, x=0.01, ha="left")
    note(fig, period)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    attribution.save(fig, path)


def figures(full, df, out, period, full_period):
    growth_panels(df, [PROXY, SP500, HAR, EWMA], "Out-of-sample: SPMO proxy, S&P 500 and volatility targeting",
                  out / "fig1_performance.png", period)

    x = df.index.to_timestamp(how="start")
    fig, ax = plt.subplots(figsize=(15, 5.5))
    for k in (HAR, EWMA):
        w = df[f"w {k}"]
        ax.step(x, w, where="post", color=COLORS[k], linewidth=1.3,
                label=f"{k}: average {w.mean():.2f}, range {w.min():.2f}-{w.max():.2f}")
    ax.axhline(1, color=attribution.INK_2, linewidth=0.8)
    ax.axvline(CRASH.to_timestamp(how="start"), color=attribution.MUTED, linewidth=0.9, linestyle="--")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Exposure (1 = fully invested)")
    ax.set_title(f"Out-of-sample exposure to the SPMO proxy: {TARGET:.0%} / forecast, no cap (dashed line: October 1987)")
    ax.legend(loc="upper left")
    note(fig, period)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    attribution.save(fig, out / "fig2_exposure.png")

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.bar(x, df["RV"], width=25, align="edge", color="#b9b8b3", label="Realised volatility of the month")
    for k in (HAR, EWMA):
        ax.step(x, df[f"forecast {k}"], where="post", color=COLORS[k], linewidth=1.3, label=f"{k.split(' vol')[0]} forecast")
    ax.axhline(TARGET, color=attribution.INK, linewidth=0.9, linestyle="--", label=f"Target {TARGET:.0%}")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("Annualised volatility")
    ax.set_title("Out-of-sample: volatility forecasts vs realised volatility by month")
    ax.legend(loc="upper right")
    note(fig, period)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    attribution.save(fig, out / "fig3_forecast.png")

    growth_panels(full, [PROXY, SP500], "Out-of-sample: SPMO proxy vs S&P 500 over the whole period",
                  out / "fig4_proxy_vs_sp500.png", full_period)


# ---------- Report ----------

def summary(full, df, period, full_period):
    t1, ex1 = stats_table(df, [PROXY, SP500, HAR, EWMA])
    t2, ex2 = stats_table(full, [PROXY, SP500])
    acc = vmc.accuracy(df["RV"], {"HAR": df[f"forecast {HAR}"], "EWMA (0.97)": df[f"forecast {EWMA}"],
                                  "Last month (RV1)": df["RV last month"]})
    lines = [
        "# Out-of-sample evaluation",
        "",
        "Generated by `scripts/10_oos.py`. The design was frozen at git tag `pre-oos` before this evaluation was "
        "run; all settings are the in-sample ones. No transaction or financing costs.",
        "",
        "## Setup",
        "",
        f"- SPMO proxy: capped version of `scripts/03_spmo_proxy.py`, rebalances with reference months "
        f"{OOS_REFS[0].strftime('%b-%Y')} to {OOS_REFS[1].strftime('%b-%Y')}.",
        "- S&P 500: CRSP S&P 500 value-weighted index including dividends (vwretd).",
        f"- HAR and EWMA ({vmc.EWMA_DECAY}): as in scripts 08 and 09. HAR is re-estimated monthly on an expanding "
        "window of out-of-sample data only (12 months of inputs, then at least 24 pairs). Exposure = "
        f"{TARGET:.0%} / forecast, no cap, rest at the T-bill rate.",
        "- Sharpe uses the T-bill rate. z is the Jobson-Korkie test with Memmel's correction (|z| > 1.96 is "
        "significant at 5%).",
        "",
        f"## Volatility targeting ({period}, {len(df)} months)",
        "",
        "The months covered by both forecasts.",
        "",
        proxy.md_table(t1),
        "",
        z_table(ex1, [(HAR, PROXY), (EWMA, PROXY), (HAR, EWMA), (PROXY, SP500), (HAR, SP500), (EWMA, SP500)]),
        "",
        "Forecast accuracy against the realised volatility of the month (R^2 of log forecasts; mean log error, "
        "positive = realised above forecast; RMSE in volatility units):",
        "",
        proxy.md_table(acc.reset_index().rename(columns={"index": "Forecast"}), floatfmt="{:.3f}"),
        "",
        f"## SPMO proxy vs S&P 500 ({full_period}, {len(full)} months)",
        "",
        "The whole out-of-sample period, including the months before the first HAR forecast.",
        "",
        proxy.md_table(t2),
        "",
        z_table(ex2, [(PROXY, SP500)]),
        "",
        "## Context",
        "",
        "- The out-of-sample period overlaps the sample in which momentum was documented (Jegadeesh and Titman "
        "1993) and falls in a strong-momentum regime (Hwang and Rubesam 2015). It tests this project's "
        "construction choices, not the existence of momentum.",
        "- It includes the October 1987 crash, a one-month shock that a forecast made at the previous month end "
        "cannot anticipate.",
        "",
        "## Figures",
        "",
    ]
    for f, c in [("fig1_performance.png", "Performance"), ("fig2_exposure.png", "Exposure"),
                 ("fig3_forecast.png", "Forecasts"), ("fig4_proxy_vs_sp500.png", "Proxy vs S&P 500")]:
        lines += [f"![{c}]({f})", ""]
    return "\n".join(lines)


def check_in_sample(raw):
    """Run the same code on the in-sample period and compare with the outputs of scripts 08 and 09."""
    _, df = run(raw, attribution.IS_FIRST_REF, None)
    h = pd.read_csv(ROOT / "output" / "har_vol_target" / "monthly.csv", index_col=0)
    v = pd.read_csv(ROOT / "output" / "vol_model_comparison" / "monthly.csv", index_col=0)
    h.index, v.index = pd.PeriodIndex(h.index, freq="M"), pd.PeriodIndex(v.index, freq="M")
    m1, m2 = df.index.intersection(h.index), df.index.intersection(v.index)
    d_har = (df.loc[m1, HAR] - h.loc[m1, "HAR vol target (15%)"]).abs().max()
    d_ewma = (df.loc[m2, EWMA] - v.loc[m2, f"EWMA ({vmc.EWMA_DECAY})"]).abs().max()
    print(f"In-sample check: HAR max abs diff vs script 08 {d_har:.2e} over {len(m1)} months; "
          f"EWMA vs script 09 {d_ewma:.2e} over {len(m2)} months")
    assert d_har < 1e-6 and d_ewma < 1e-6, "the out-of-sample code does not reproduce the in-sample results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--out", type=Path, default=ROOT / "output" / "oos")
    ap.add_argument("--period", choices=["oos", "is"], default="oos")
    args = ap.parse_args()
    if args.period == "is":
        check_in_sample(args.raw)
        return

    full, df = run(args.raw, *OOS_REFS)
    for frame in (full, df):
        assert frame.index[0] >= OOS_HOLD[0] and frame.index[-1] <= OOS_HOLD[1], "months outside the out-of-sample period"
    ref = pd.read_csv(ROOT / "data" / "processed" / "spmo_proxy_returns.csv", index_col=0)["proxy_capped"]
    ref.index = pd.PeriodIndex(ref.index, freq="M")
    gap = (full[PROXY] - ref.loc[full.index]).abs().max()
    assert gap < 1e-7, f"proxy returns differ from scripts/03_spmo_proxy.py by {gap:.2e}"

    span = lambda d: f"{d.index[0].strftime('%b-%Y')} to {d.index[-1].strftime('%b-%Y')}"
    args.out.mkdir(parents=True, exist_ok=True)
    df.rename_axis("month").to_csv(args.out / "monthly.csv", float_format="%.6f")
    figures(full, df, args.out, span(df), span(full))
    (args.out / "summary.md").write_text(summary(full, df, span(df), span(full)), encoding="utf-8")
    print(f"Out-of-sample: {span(full)} ({len(full)} months), volatility targeting {span(df)} ({len(df)} months); "
          f"wrote {args.out}")


if __name__ == "__main__":
    main()
