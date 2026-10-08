"""
Volatility targeting of the SPMO proxy with a HAR forecast of next month's volatility.

Each month end the strategy forecasts the SPMO proxy's realised volatility for the next month and
sets its exposure for that month to

  w = TARGET / forecast        (no cap; the rest, or the borrowing when w > 1, at the T-bill rate)

Forecast: a HAR model (Corsi 2009) in logs, on monthly data,

  log RV(m+1) = b0 + b1 log RV1(m) + b2 log RV3(m) + b3 log RV12(m) + e

RV1, RV3 and RV12 are the annualised realised volatilities over the last 1, 3 and 12 calendar
months, RV(m+1) that of the next month, each sqrt(252 x mean of squared daily returns). The
coefficients are re-estimated every month on an expanding window of the pairs observed so far
(at least MIN_OBS), so each forecast uses only data available at the time. The forecast is
exp(fitted + s^2 / 2), s^2 the residual variance, which corrects the bias of exponentiating a
log forecast.

Daily SPMO proxy returns are rebuilt from the pipeline's start-of-month weights, held within each
month; they are used only for volatility. Managed and unmanaged returns use the pipeline's monthly
returns. In-sample only, starting from the first in-sample holding month, so the first forecasts
come after 12 months of data plus MIN_OBS months of estimation. No transaction or financing costs.

Outputs (output/har_vol_target/)
  summary.md                      method, forecast accuracy, performance
  monthly.csv                     realised vol inputs, forecast, exposure, returns
  fig1_forecast.png               forecast vs realised volatility
  fig2_exposure.png               exposure to the SPMO proxy
  fig3_performance.png            growth of $1, drawdown and rolling 12-month volatility
  fig4_coefficients.png           HAR coefficients as the window expands
  fig5_cumulative_in_sample.png   growth of $1 over the whole in-sample period

Usage:
    python scripts/08_har_vol_target.py [--raw data/raw] [--out output/har_vol_target]
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
regressions = importlib.import_module("07_factor_regressions")

ROOT = proxy.ROOT
TARGET = 0.15
HORIZONS = {"RV1": 1, "RV3": 3, "RV12": 12}          # months
MIN_OBS = 24                                          # regression pairs before the first forecast
PORTFOLIO = "111 capped"                              # the SPMO proxy
SUBPERIODS = [("1995-2014", 1995, 2014), ("2015-2025", 2015, 2025)]
MANAGED, UNMANAGED = "HAR vol target (15%)", "SPMO proxy"
COLORS = {MANAGED: "#eb6834", UNMANAGED: "#2a78d6"}


# ---------- Data ----------

def proxy_panel(raw, lookback_years=1, first_ref=attribution.IS_FIRST_REF, last_ref=None):
    """Monthly returns of the SPMO proxy (pipeline); its daily returns, held within each month from
    the pipeline's start-of-month weights; for each holding month, the weights known at the previous
    month end (before any delisting in the month is known); and the daily stock-return panel, starting
    `lookback_years` before the first holding month. Rebalances with reference month in [first_ref,
    last_ref]; in-sample by default."""
    ret, _, forms = attribution.formations(raw, first_ref, last_ref)
    first, last = forms[0][1][0], forms[-1][1][-1]
    panel = pd.concat([pd.read_parquet(raw / "crsp_daily" / f"{y}.parquet", columns=["permno", "dlycaldt", "dlyret"])
                       for y in range(first.year - lookback_years, last.year + 1)])
    panel = panel.pivot(index="dlycaldt", columns="permno", values="dlyret").astype(float)
    daily = panel[(panel.index.to_period("M") >= first) & (panel.index.to_period("M") <= last)]
    days = daily.groupby(daily.index.to_period("M")).groups
    monthly, dly, known = [], [], {}
    for _, hold, w_t in forms:
        w = w_t[PORTFOLIO]
        r = ret.loc[hold, w.index]
        hw = proxy.hold_weights(r, w)
        monthly.append(pd.Series(np.einsum("ij,ij->i", hw, np.nan_to_num(r.to_numpy())), index=hold))
        cols = daily.columns.get_indexer(w.index)
        prev = w.to_numpy() / w.sum()
        for i, m in enumerate(hold):
            known[m] = pd.Series(prev, index=w.index)
            R = np.zeros((len(days[m]), len(w)))
            R[:, cols >= 0] = np.nan_to_num(daily.loc[days[m]].to_numpy()[:, cols[cols >= 0]])
            value = np.cumprod(1 + R, axis=0) @ hw[i]
            dly.append(pd.Series(value / np.concatenate([[1.0], value[:-1]]) - 1, index=days[m]))
            end = hw[i] * (1 + np.nan_to_num(r.iloc[i].to_numpy()))
            prev = end / end.sum()
    return pd.concat(monthly), pd.concat(dly), known, panel


def proxy_returns(raw):
    """Monthly and daily returns of the SPMO proxy (see proxy_panel)."""
    monthly, daily, _, _ = proxy_panel(raw)
    return monthly, daily


def realised_vol(daily):
    """Annualised realised volatility of each month and over the trailing 1, 3 and 12 months."""
    month = daily.index.to_period("M")
    ss, n = daily.pow(2).groupby(month).sum(), daily.groupby(month).size()
    out = pd.DataFrame({k: np.sqrt(252 * ss.rolling(h).sum() / n.rolling(h).sum()) for k, h in HORIZONS.items()})
    out["RV next"] = out["RV1"].shift(-1)
    return out


# ---------- Forecast ----------

def har_forecasts(rv):
    """Expanding-window HAR forecasts of next month's volatility, made at each month end."""
    X = np.log(rv[list(HORIZONS)])
    y = np.log(rv["RV next"])
    rows, coefs = {}, {}
    valid = X.notna().all(axis=1)
    months = rv.index[valid]
    for i, m in enumerate(months):
        train = months[:i]                                 # pairs whose target month has ended by m
        train = train[y.loc[train].notna()]
        if len(train) < MIN_OBS:
            continue
        A = np.column_stack([np.ones(len(train)), X.loc[train].to_numpy()])
        b, *_ = np.linalg.lstsq(A, y.loc[train].to_numpy(), rcond=None)
        s2 = np.var(y.loc[train].to_numpy() - A @ b, ddof=A.shape[1])
        rows[m + 1] = np.exp(b[0] + X.loc[m].to_numpy() @ b[1:] + s2 / 2)
        coefs[m + 1] = dict(zip(["const", *HORIZONS], b))
    return pd.Series(rows, name="forecast"), pd.DataFrame(coefs).T


def accuracy(rv, forecast):
    """Out-of-sample R^2 of log forecasts against the realised log vol, for HAR and naive forecasts."""
    months = forecast.index.intersection(rv.index)
    actual = np.log(rv["RV1"].loc[months])
    out = {}
    candidates = {"HAR": forecast.loc[months], "Last month (RV1)": rv["RV1"].shift(1).loc[months],
                  "Last 3 months (RV3)": rv["RV3"].shift(1).loc[months], "Last 12 months (RV12)": rv["RV12"].shift(1).loc[months]}
    sst = ((actual - actual.mean()) ** 2).sum()
    for name, f in candidates.items():
        e = actual - np.log(f)
        out[name] = {"R2 (log vol)": 1 - (e ** 2).sum() / sst, "Mean error (log)": e.mean(),
                     "RMSE (vol)": np.sqrt(((rv["RV1"].loc[months] - f) ** 2).mean())}
    return pd.DataFrame(out).T


# ---------- Performance ----------

def perf(r, rf, w):
    ex = r - rf
    wealth = (1 + r).cumprod()
    return {"CAGR": (1 + r).prod() ** (12 / len(r)) - 1, "Volatility": r.std() * 12 ** 0.5,
            "Sharpe": ex.mean() / ex.std() * 12 ** 0.5, "Max drawdown": (wealth / wealth.cummax() - 1).min(),
            "Avg exposure": w.mean(), "Exposure range": f"{w.min():.2f}-{w.max():.2f}"}


def perf_table(df, rf):
    rows = []
    for name, wcol in [(UNMANAGED, None), (MANAGED, "w")]:
        w = pd.Series(1.0, index=df.index) if wcol is None else df[wcol]
        p = perf(df[name], rf.loc[df.index], w)
        rows.append({"Portfolio": name, "CAGR": f"{p['CAGR']:.1%}", "Volatility": f"{p['Volatility']:.1%}",
                     "Sharpe": f"{p['Sharpe']:.2f}", "Max drawdown": f"{p['Max drawdown']:.1%}",
                     "Avg exposure": f"{p['Avg exposure']:.2f}", "Exposure range": p["Exposure range"]})
    sh = {k: (df[k] - rf.loc[df.index]) for k in (UNMANAGED, MANAGED)}
    return proxy.md_table(pd.DataFrame(rows)), sh


def jkm(a, b):
    """Jobson-Korkie z (Memmel correction) for Sharpe(a) - Sharpe(b) on monthly excess returns."""
    s1, s2 = a.mean() / a.std(), b.mean() / b.std()
    rho = np.corrcoef(a, b)[0, 1]
    return (s1 - s2) / np.sqrt((2 - 2 * rho + 0.5 * (s1 ** 2 + s2 ** 2 - 2 * s1 * s2 * rho ** 2)) / len(a))


# ---------- Figures ----------

def note(fig, period, extra=""):
    fig.text(0.01, 0.005, f"In-sample: {period}. Forecast made at each month end from data up to that date. "
             f"Monthly total returns, no transaction or financing costs. Data: CRSP via WRDS.{' ' + extra if extra else ''}",
             ha="left", va="bottom", color=attribution.MUTED, fontsize=8.5)


def figures(df, coefs, out, period):
    x = df.index.to_timestamp(how="start")
    box = {"boxstyle": "round,pad=0.2", "facecolor": attribution.SURFACE, "edgecolor": "none", "alpha": 0.9}

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.bar(x, df["RV1"], width=25, align="edge", color="#b9b8b3", label="Realised volatility of the month")
    ax.step(x, df["forecast"], where="post", color="#eb6834", linewidth=1.4, label="HAR forecast made at the previous month end")
    ax.axhline(TARGET, color=attribution.INK, linewidth=0.9, linestyle="--", label=f"Target {TARGET:.0%}")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("Annualised volatility")
    ax.set_title("SPMO proxy: HAR forecast vs realised volatility by month")
    ax.legend(loc="upper left", bbox_to_anchor=(0.22, 1))
    note(fig, period)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    attribution.save(fig, out / "fig1_forecast.png")

    fig, ax = plt.subplots(figsize=(15, 5.5))
    ax.step(x, df["w"], where="post", color="#eb6834", linewidth=1.3)
    ax.axhline(1, color=attribution.INK_2, linewidth=0.8)
    ax.axhline(df["w"].mean(), color=attribution.INK, linewidth=0.9, linestyle="--")
    ax.annotate(f"Average {df['w'].mean():.2f}; range {df['w'].min():.2f} to {df['w'].max():.2f}; "
                f"above 1 in {(df['w'] > 1).mean():.0%} of months", (0, df["w"].mean()),
                xycoords=("axes fraction", "data"), xytext=(4, 4), textcoords="offset points", fontsize=8.5,
                color=attribution.INK_2, bbox=box)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Exposure (1 = fully invested)")
    ax.set_title(f"Exposure to the SPMO proxy: {TARGET:.0%} / HAR forecast, no cap (above 1 = borrowing at the T-bill rate)")
    note(fig, period)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    attribution.save(fig, out / "fig2_exposure.png")

    dates = [df.index[0].to_timestamp(how="start")] + list(df.index.to_timestamp(how="end"))
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True, gridspec_kw={"height_ratios": [1.5, 1, 1]})
    for k, c in COLORS.items():
        g = np.concatenate([[1.0], (1 + df[k]).cumprod().to_numpy()])
        axes[0].plot(dates, g, color=c, linewidth=2, label=f"{k}  (${g[-1]:.0f})")
        axes[1].plot(dates, g / np.maximum.accumulate(g) - 1, color=c, linewidth=1.3)
        axes[2].plot(df.index.to_timestamp(how="end"), df[k].rolling(12).std() * 12 ** 0.5, color=c, linewidth=1.3)
    axes[0].set_yscale("log")
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}"))
    axes[0].set_title("(a) Growth of $1 (log scale; final value in brackets)")
    axes[0].legend(loc="upper left")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axes[1].set_title("(b) Drawdown")
    axes[2].axhline(TARGET, color=attribution.INK, linewidth=0.9, linestyle="--")
    axes[2].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axes[2].set_title(f"(c) Rolling 12-month volatility of monthly returns (dashed: {TARGET:.0%} target)")
    fig.suptitle("SPMO proxy with and without HAR volatility targeting", x=0.01, ha="left")
    note(fig, period)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    attribution.save(fig, out / "fig3_performance.png")

    fig, ax = plt.subplots(figsize=(14, 5.5))
    for k, c in zip(HORIZONS, attribution.EFFECT_COLORS):
        ax.plot(coefs.index.to_timestamp(how="start"), coefs[k], color=c, linewidth=1.6, label=f"log {k}")
    ax.axhline(0, color=attribution.INK_2, linewidth=0.8)
    ax.set_ylabel("Coefficient")
    ax.set_title("HAR coefficients estimated on the expanding window at each month end")
    ax.legend(loc="upper right")
    note(fig, period)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    attribution.save(fig, out / "fig4_coefficients.png")


# ---------- Report ----------

def summary(df, coefs, acc, rf, check, period):
    full_table, ex = perf_table(df, rf)
    lines = [
        "# HAR volatility targeting of the SPMO proxy",
        "",
        f"Generated by `scripts/08_har_vol_target.py`. In-sample only: {period} ({len(df)} months). No "
        "transaction or financing costs.",
        "",
        "## Method",
        "",
        f"- Exposure for month m+1 = {TARGET:.0%} / forecast of the proxy's realised volatility in m+1, with no "
        "cap. The remainder (or the borrowing, when exposure exceeds 1) earns the one-month T-bill rate.",
        "- Forecast: HAR model in logs (Corsi 2009), log RV(m+1) = b0 + b1 log RV1(m) + b2 log RV3(m) + "
        "b3 log RV12(m), where RV1, RV3 and RV12 are annualised realised volatilities over the last 1, 3 and "
        "12 calendar months, sqrt(252 x mean squared daily return). The forecast is exp(fitted + s^2/2).",
        f"- Coefficients are re-estimated each month on an expanding window of all pairs observed so far "
        f"(at least {MIN_OBS}), so forecasts use only data available at the time. Inputs start at the first "
        "in-sample holding month, so forecasting starts after 12 + "
        f"{MIN_OBS} months.",
        f"- Daily proxy returns are rebuilt from the pipeline's start-of-month weights; compounded, they match "
        f"the pipeline's monthly returns within {check:.2%} a month (delisting returns exist only monthly).",
        "",
        "## Forecast accuracy",
        "",
        "Out-of-sample over the forecast months: R^2 of the log forecast against the realised log volatility, "
        "mean log error (positive = realised above forecast) and RMSE in volatility units. The naive "
        "forecasts use the trailing realised volatility as the forecast.",
        "",
        proxy.md_table(acc.reset_index().rename(columns={"index": "Forecast"}), floatfmt="{:.3f}"),
        "",
        f"Final coefficients (expanding window to {coefs.index[-1].strftime('%b-%Y')}): "
        + ", ".join(f"{k} {v:+.3f}" for k, v in coefs.iloc[-1].items()) + ".",
        "",
        "## Performance",
        "",
        "Same months for both. Sharpe uses the T-bill rate. Sharpe difference z is the Jobson-Korkie test "
        "with Memmel's correction (|z| > 1.96 is significant at 5%).",
        "",
        "### Full period",
        "",
        full_table,
        "",
        f"Sharpe difference z: {jkm(ex[MANAGED].to_numpy(), ex[UNMANAGED].to_numpy()):+.2f}.",
        "",
    ]
    for label, a, b in SUBPERIODS:
        part = df[(df.index.year >= a) & (df.index.year <= b)]
        table, e = perf_table(part, rf)
        span = f"{part.index[0].strftime('%b-%Y')} to {part.index[-1].strftime('%b-%Y')}"
        lines += [f"### {span}", "", table, "",
                  f"Sharpe difference z: {jkm(e[MANAGED].to_numpy(), e[UNMANAGED].to_numpy()):+.2f}.", ""]
    lines += ["## Figures", ""]
    for f, c in [("fig1_forecast.png", "Forecast vs realised"), ("fig2_exposure.png", "Exposure"),
                 ("fig3_performance.png", "Performance"), ("fig4_coefficients.png", "Coefficients"),
                 ("fig5_cumulative_in_sample.png", "Whole in-sample period")]:
        lines += [f"![{c}]({f})", ""]
    lines += ["## Reference", "", "Corsi, F. (2009). A simple approximate long-memory model of realized volatility. "
              "*Journal of Financial Econometrics* 7(2), 174-196.", ""]
    return "\n".join(lines)


def cumulative_figure(monthly, df, rf, out):
    """Growth of $1 over the whole in-sample period. Before the first forecast the HAR strategy is
    fully invested in the proxy, so the two curves coincide until then."""
    first = df.index[0]
    har = monthly.copy()
    har.loc[df.index] = df[MANAGED]
    r = pd.DataFrame({UNMANAGED: monthly, MANAGED: har})
    dates = [r.index[0].to_timestamp(how="start")] + list(r.index.to_timestamp(how="end"))
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axvspan(dates[0], first.to_timestamp(how="start"), color=attribution.GRID, zorder=0)
    ax.annotate("Warm-up: no forecast yet,\nHAR strategy fully invested", (dates[0], 1), xycoords=("data", "axes fraction"),
                xytext=(6, -6), textcoords="offset points", ha="left", va="top", fontsize=8.5, color=attribution.INK_2)
    for k, c in COLORS.items():
        g = np.concatenate([[1.0], (1 + r[k]).cumprod().to_numpy()])
        ex = r[k] - rf.loc[r.index]
        ax.plot(dates, g, color=c, linewidth=2,
                label=f"{k}: cumulative {g[-1] - 1:+.0%}, CAGR {g[-1] ** (12 / len(r)) - 1:.1%}, "
                      f"Sharpe {ex.mean() / ex.std() * 12 ** 0.5:.2f}")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}"))
    ax.set_ylabel("Growth of $1 (log scale)")
    ax.set_title(f"SPMO proxy vs HAR volatility targeting, whole in-sample period "
                 f"({r.index[0].strftime('%b-%Y')} to {r.index[-1].strftime('%b-%Y')})")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.9))
    fig.text(0.01, 0.005, f"In-sample. HAR targeting starts in {first.strftime('%b-%Y')}, after 12 months of inputs and "
             f"{MIN_OBS} months of estimation. Monthly total returns, no transaction or financing costs. Data: CRSP via WRDS.",
             ha="left", va="bottom", color=attribution.MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    attribution.save(fig, out / "fig5_cumulative_in_sample.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--out", type=Path, default=ROOT / "output" / "har_vol_target")
    args = ap.parse_args()

    monthly, daily = proxy_returns(args.raw)
    check = ((1 + daily).groupby(daily.index.to_period("M")).prod() - 1 - monthly).abs().max()
    rv = realised_vol(daily)
    forecast, coefs = har_forecasts(rv)

    ff = pd.read_parquet(args.raw / "ff5_factors_monthly.parquet", columns=["date", "rf"])
    rf = ff.set_index(ff["date"].dt.to_period("M"))["rf"].astype(float)
    months = forecast.index.intersection(monthly.index)
    df = rv.loc[months, list(HORIZONS)].shift(1).add_suffix(" (previous month end)")
    df["RV1"] = rv.loc[months, "RV1"]
    df["forecast"] = forecast.loc[months]
    df["w"] = TARGET / df["forecast"]
    df[UNMANAGED] = monthly.loc[months]
    df[MANAGED] = df["w"] * df[UNMANAGED] + (1 - df["w"]) * rf.loc[months]
    assert months[0] >= attribution.IS_FIRST_REF + proxy.LAG, "out-of-sample months in the results"
    period = f"{months[0].strftime('%b-%Y')} to {months[-1].strftime('%b-%Y')}"
    acc = accuracy(rv, forecast.loc[months])

    args.out.mkdir(parents=True, exist_ok=True)
    df.rename_axis("month").to_csv(args.out / "monthly.csv", float_format="%.6f")
    figures(df, coefs.loc[months], args.out, period)
    cumulative_figure(monthly, df, rf, args.out)
    (args.out / "summary.md").write_text(summary(df, coefs.loc[months], acc, rf, check, period), encoding="utf-8")
    print(f"Months {months[0]} .. {months[-1]} ({len(months)}); daily/monthly max gap {check:.4%}; wrote {args.out}")


if __name__ == "__main__":
    main()
