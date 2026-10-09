"""
Post-OOS observations: analyses run AFTER the one-time out-of-sample evaluation (scripts/10_oos.py).

They are supplementary and must not be read as pre-registered tests. They extend the in-sample
analyses to the whole sample, 1975-2025, with every setting unchanged:

  1. The 2 x 2 x 2 attribution of scripts/04_attribution.py on the out-of-sample period, and a factor
     decomposition of the SPMO proxy's excess return over the S&P 500 there.
  2. M x C over 1975-2025.
  3. The strategies of scripts/10_oos.py over 1978-2025, with HAR estimated on one expanding window
     starting in 1975 (its out-of-sample months equal those of script 10).

Outputs (output/post_oos/)
  summary.md, fig1_full_period.png, fig2_mxc_full_period.png

Usage:
    python scripts/11_post_oos.py [--raw data/raw] [--out output/post_oos]
"""
import argparse
import importlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MultipleLocator, PercentFormatter

proxy = importlib.import_module("03_spmo_proxy")
att = importlib.import_module("04_attribution")
reg = importlib.import_module("07_factor_regressions")
har = importlib.import_module("08_har_vol_target")
oos = importlib.import_module("10_oos")

ROOT = proxy.ROOT
FIRST_REF = oos.OOS_REFS[0]
OOS_END = oos.OOS_HOLD[1]
FORMULA = {name: f for _, name, f, _ in att.EFFECTS}
CODES = list(att.PORTFOLIOS)
FACTORS = ["mktrf", "smb", "hml", "rmw", "cma", "umd"]
NOTE = ("Post-OOS observation: run after the one-time out-of-sample evaluation; not a pre-registered test. "
        "Monthly total returns before costs. Data: CRSP via WRDS.")


def nw_mean(d):
    b, t, _ = reg.ols_nw(np.asarray(d), np.empty((len(d), 0)))
    return 12 * b[0], t[0]


def portfolio_returns(raw):
    """Monthly returns of the 2 x 2 x 2 portfolios (and the capped proxy) for every rebalance since 1975."""
    ret, _, forms = att.formations(raw, first_ref=FIRST_REF)
    out = {}
    for t, hold, w_t in forms:
        for k, w in w_t.items():
            out.setdefault(k, []).append(pd.Series(proxy.hold_returns(ret.loc[hold, w.index], w), index=hold))
    return pd.DataFrame({k: pd.concat(v) for k, v in out.items()})


def market_and_factors(raw, index):
    idx = pd.read_parquet(raw / "sp500_index_daily.parquet", columns=["date", "vwretd"]).set_index("date")["vwretd"]
    sp = np.expm1(np.log1p(idx.astype(float)).groupby(idx.index.to_period("M")).sum()).reindex(index)
    ff = pd.read_parquet(raw / "ff5_factors_monthly.parquet")
    f = ff.set_index(ff["date"].dt.to_period("M")).drop(columns="date").astype(float).reindex(index)
    return sp, f


# ---------- 1. Out-of-sample attribution ----------

def oos_attribution(r, sp, f):
    m = r.index <= OOS_END
    p, rf = r[m], f.loc[r.index[m], "rf"]
    perf = []
    for k, name in [*att.PORTFOLIOS.items(), ("111 capped", "SPMO proxy (capped)")]:
        x = p[k]
        e = x - rf
        perf.append({"Portfolio": f"`{k}` {name}" if k in att.PORTFOLIOS else name,
                     "CAGR": f"{(1 + x).prod() ** (12 / len(x)) - 1:.1%}", "Sharpe": f"{e.mean() / e.std() * 12 ** 0.5:.2f}"})
    x = sp[m]
    e = x - rf
    perf.append({"Portfolio": "S&P 500 (CRSP value-weighted index)", "CAGR": f"{(1 + x).prod() ** (12 / len(x)) - 1:.1%}",
                 "Sharpe": f"{e.mean() / e.std() * 12 ** 0.5:.2f}"})
    effects = []
    for _, name, formula, desc in att.EFFECTS:
        mean, t = nw_mean(att.evaluate(formula, p[CODES]))
        effects.append({"Effect": name, "Description": desc, "Annualised mean (NW t)": f"{mean:+.1%} ({t:.2f})"})
    for name, d in [("M (equal weight)", p["100"] - p["000"]), ("M (cap weight)", p["110"] - p["010"])]:
        mean, t = nw_mean(d)
        effects.append({"Effect": name, "Description": "100 - 000" if "equal" in name else "110 - 010",
                        "Annualised mean (NW t)": f"{mean:+.1%} ({t:.2f})"})

    y = (p["111 capped"] - sp[m]).to_numpy()
    X = f.loc[p.index, FACTORS].to_numpy()
    b, t, r2 = reg.ols_nw(y, X)
    means = 12 * f.loc[p.index, FACTORS].mean().to_numpy()
    decomp = [{"Term": "Alpha", "Loading (NW t)": "", "Contribution, % a year": f"{12 * b[0]:+.1%} (t {t[0]:.2f})"}]
    for i, c in enumerate(FACTORS):
        decomp.append({"Term": reg.FACTOR_NAMES[c], "Loading (NW t)": f"{b[i + 1]:+.2f} ({t[i + 1]:.1f})",
                       "Contribution, % a year": f"{b[i + 1] * means[i]:+.1%}"})
    decomp.append({"Term": "Total (mean excess over the S&P 500)", "Loading (NW t)": f"R^2 {r2:.2f}",
                   "Contribution, % a year": f"{12 * y.mean():+.1%}"})
    premia = {reg.FACTOR_NAMES[c]: v for c, v in zip(FACTORS, means)}
    return pd.DataFrame(perf), pd.DataFrame(effects), pd.DataFrame(decomp), premia, p.index


# ---------- 2. M x C over the whole sample ----------

PERIODS = [("1975-1994 (out-of-sample)", 1975, 1994), ("1995-2014", 1995, 2014), ("2015-2025", 2015, 2025),
           ("1995-2025 (in-sample)", 1995, 2025), ("1975-2025 (whole sample)", 1975, 2025)]
DECADES = [(f"{d}-{d + 9}", d, d + 9) for d in range(1975, 2025, 10)]
EFFECT_COLORS = {"C (market)": "#2a78d6", "C (momentum)": "#eb6834", "M x C": "#1baf7a"}


def mxc_table(r):
    eff = pd.DataFrame({k: att.evaluate(FORMULA[k], r[CODES]) for k in EFFECT_COLORS})
    rows = []
    for label, a, b in PERIODS + DECADES:
        d = eff[(eff.index.year >= a) & (eff.index.year <= b)]
        row = {"Period": label, "Months": len(d)}
        for k in eff:
            mean, t = nw_mean(d[k])
            row[k] = f"{mean:+.1%} ({t:.2f})"
        rows.append(row)
    return eff, pd.DataFrame(rows)


def mxc_figure(r, eff, out):
    x = eff.index.to_timestamp(how="end")
    oos_end = OOS_END.to_timestamp(how="end")
    fig, axes = plt.subplots(3, 1, figsize=(15, 13), gridspec_kw={"height_ratios": [1.4, 1, 1]})
    ax = axes[0]
    for k, c in EFFECT_COLORS.items():
        label = "M x C = C (momentum) - C (market)" if k == "M x C" else f"{k}: {FORMULA[k]}"
        ax.plot(x, eff[k].cumsum(), color=c, linewidth=2.2 if k == "M x C" else 1.4, label=label)
    ax.axvspan(x[0], oos_end, color=att.GRID, zorder=0)
    ax.annotate("Out-of-sample period", (x[0], 1), xycoords=("data", "axes fraction"), xytext=(6, -6),
                textcoords="offset points", va="top", fontsize=9, color=att.INK_2)
    ax.axhline(0, color=att.INK_2, linewidth=0.8)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("Cumulative sum of monthly differences")
    ax.set_title("(a) Cumulative market-cap effects, 1975-2025")
    ax.legend(loc="upper left", bbox_to_anchor=(0.42, 1))

    ax = axes[1]
    yearly = att.evaluate(FORMULA["M x C"], att.annual(r[CODES]))
    ax.bar(yearly.index, yearly, color=[att.POSITIVE if v >= 0 else att.NEGATIVE for v in yearly], width=0.8)
    ax.axvspan(yearly.index[0] - 0.5, OOS_END.year - 0.5, color=att.GRID, zorder=0)
    ax.axhline(0, color=att.INK_2, linewidth=0.8)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title(f"(b) M x C by calendar year ({yearly.index[0]} starts in {r.index[0].strftime('%b')})")

    ax = axes[2]
    pos = np.arange(len(DECADES))
    for i, (k, c) in enumerate(EFFECT_COLORS.items()):
        vals, errs = [], []
        for _, a, b in DECADES:
            mean, t = nw_mean(eff[k][(eff.index.year >= a) & (eff.index.year <= b)])
            vals.append(mean)
            errs.append(1.96 * abs(mean / t))
        ax.bar(pos + (i - 1) * 0.26, vals, 0.26, color=c, yerr=errs, label=k,
               error_kw={"ecolor": att.INK_2, "elinewidth": 0.8, "capsize": 3})
    ax.set_xticks(pos, [d[0] for d in DECADES])
    ax.axhline(0, color=att.INK_2, linewidth=0.8)
    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title("(c) Annualised mean by decade, 95% confidence intervals (Newey-West)")
    ax.legend(loc="upper left")
    fig.suptitle("M x C over the whole sample (cap-weighted 2 x 2 x 2 construction)", x=0.01, ha="left")
    fig.text(0.01, 0.005, NOTE, color=att.MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.02, 1, 0.985))
    att.save(fig, out / "fig2_mxc_full_period.png")


# ---------- 3. Strategies over the whole sample ----------

STRATS = [oos.PROXY, oos.SP500, oos.HAR, oos.EWMA]
EVENTS = [("1987-10", "Oct-1987"), ("2000-03", "Mar-2000"), ("2008-09", "Sep-2008"), ("2020-03", "Mar-2020")]


def strategy_table(df):
    rows = []
    blocks = [("Whole sample", df), ("Out-of-sample part", df[df.index <= OOS_END]),
              ("1995-2014", df[(df.index.year >= 1995) & (df.index.year <= 2014)]), ("2015-2025", df[df.index.year >= 2015])]
    for label, d in blocks:
        span = f"{d.index[0].strftime('%b-%Y')} to {d.index[-1].strftime('%b-%Y')}"
        ex = {k: (d[k] - d["rf"]) for k in STRATS}
        for k in STRATS:
            x = d[k]
            g = (1 + x).cumprod()
            rows.append({"Period": f"{label} ({span})", "Portfolio": k,
                         "CAGR": f"{(1 + x).prod() ** (12 / len(x)) - 1:.1%}", "Volatility": f"{x.std() * 12 ** 0.5:.1%}",
                         "Sharpe": f"{ex[k].mean() / ex[k].std() * 12 ** 0.5:.2f}",
                         "Max drawdown": f"{(g / g.cummax() - 1).min():.1%}",
                         "Sharpe z vs proxy": "" if k in (oos.PROXY, oos.SP500)
                         else f"{har.jkm(ex[k].to_numpy(), ex[oos.PROXY].to_numpy()):+.2f}"})
    decades = []
    for d0 in range(1980, 2030, 10):
        d = df[(df.index.year >= d0) & (df.index.year <= d0 + 9)]
        row = {"Decade": f"{d0}s"}
        for k in STRATS:
            e = d[k] - d["rf"]
            row[k] = f"{e.mean() / e.std() * 12 ** 0.5:.2f}"
        decades.append(row)
    return pd.DataFrame(rows), pd.DataFrame(decades)


def strategy_figure(df, out):
    x = [df.index[0].to_timestamp(how="start")] + list(df.index.to_timestamp(how="end"))
    xs = df.index.to_timestamp(how="start")
    oos_end = OOS_END.to_timestamp(how="end")
    fig, axes = plt.subplots(4, 1, figsize=(15, 17), sharex=True, gridspec_kw={"height_ratios": [1.6, 1, 0.9, 1]})
    for ax in axes:
        ax.axvspan(x[0], oos_end, color=att.GRID, zorder=0)
        for d, _ in EVENTS:
            ax.axvline(pd.Period(d, "M").to_timestamp(how="start"), color=att.MUTED, linewidth=0.8, linestyle="--")
    for k in STRATS:
        c = oos.COLORS[k]
        g = np.concatenate([[1.0], (1 + df[k]).cumprod().to_numpy()])
        axes[0].plot(x, g, color=c, linewidth=2.2 if k == oos.HAR else 1.6, label=f"{k}  (${g[-1]:,.0f})")
        axes[1].plot(x, g / np.maximum.accumulate(g) - 1, color=c, linewidth=1.1)
        e = df[k] - df["rf"]
        axes[3].plot(df.index.to_timestamp(how="end"), e.rolling(60).mean() / e.rolling(60).std() * 12 ** 0.5,
                     color=c, linewidth=1.3, label=k)
    axes[0].set_yscale("log")
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}" if v >= 1 else f"${v:g}"))
    axes[0].set_title("(a) Growth of $1 (log scale; shaded = out-of-sample period)")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0.22, 1))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axes[1].set_title("(b) Drawdown")
    for d, lab in EVENTS:
        axes[1].annotate(lab, (pd.Period(d, "M").to_timestamp(how="start"), 0), xytext=(3, -10),
                         textcoords="offset points", fontsize=8.5, color=att.INK_2)
    for k in (oos.HAR, oos.EWMA):
        axes[2].step(xs, df[f"w {k}"], where="post", color=oos.COLORS[k], linewidth=1.0, label=k)
    axes[2].axhline(1, color=att.INK_2, linewidth=0.8)
    axes[2].set_ylim(bottom=0)
    axes[2].set_title(f"(c) Exposure to the SPMO proxy ({har.TARGET:.0%} / forecast, no cap)")
    axes[2].legend(loc="upper left", bbox_to_anchor=(0.22, 1))
    axes[3].axhline(0, color=att.INK_2, linewidth=0.8)
    axes[3].set_title("(d) Rolling 5-year Sharpe ratio")
    fig.suptitle("SPMO proxy, S&P 500 and volatility targeting, 1978-2025 (HAR on one expanding window from 1975)",
                 x=0.01, ha="left")
    fig.text(0.01, 0.005, NOTE + " No financing costs.", color=att.MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.015, 1, 0.985))
    att.save(fig, out / "fig1_full_period.png")


# ---------- Report ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--out", type=Path, default=ROOT / "output" / "post_oos")
    args = ap.parse_args()

    r = portfolio_returns(args.raw)
    sp, f = market_and_factors(args.raw, r.index)
    perf, effects, decomp, premia, oos_months = oos_attribution(r, sp, f)
    eff, mxc = mxc_table(r)

    _, df = oos.run(args.raw, FIRST_REF, None)
    o = pd.read_csv(ROOT / "output" / "oos" / "monthly.csv", index_col=0)
    o.index = pd.PeriodIndex(o.index, freq="M")
    m = o.index.intersection(df.index)
    gap = max((df.loc[m, k] - o.loc[m, k]).abs().max() for k in (oos.HAR, oos.EWMA))
    assert gap < 1e-6, f"out-of-sample months differ from scripts/10_oos.py by {gap:.2e}"
    strat, decades = strategy_table(df)

    args.out.mkdir(parents=True, exist_ok=True)
    strategy_figure(df, args.out)
    mxc_figure(r, eff, args.out)
    span = f"{oos_months[0].strftime('%b-%Y')} to {oos_months[-1].strftime('%b-%Y')}"
    lines = [
        "# Post-OOS observations",
        "",
        "Generated by `scripts/11_post_oos.py`. **These analyses were run after the one-time out-of-sample "
        "evaluation (`scripts/10_oos.py`, design frozen at tag `pre-oos`). They are supplementary observations, "
        "not pre-registered tests.** Every setting is unchanged from the in-sample analysis. Returns before costs. "
        "t-stats are Newey-West with 6 lags.",
        "",
        f"## 1. The 2 x 2 x 2 attribution on the out-of-sample period ({span})",
        "",
        "Portfolio 111 is uncapped by design; the capped SPMO proxy is shown for reference.",
        "",
        proxy.md_table(perf),
        "",
        proxy.md_table(effects),
        "",
        "Factor means over the same months (% a year): "
        + ", ".join(f"{k} {v:+.1%}" for k, v in premia.items()) + ".",
        "",
        "Decomposition of the capped proxy's monthly excess return over the S&P 500: regression on FF5 + UMD; "
        "contribution = loading x factor mean.",
        "",
        proxy.md_table(decomp),
        "",
        "## 2. M x C over the whole sample",
        "",
        "Cap-weighted 2 x 2 x 2 construction; annualised mean (Newey-West t).",
        "",
        proxy.md_table(mxc),
        "",
        "## 3. Strategies over the whole sample",
        "",
        f"HAR is estimated on one expanding window from 1975, so its out-of-sample months equal those of "
        f"`scripts/10_oos.py` (max abs difference {gap:.1e}); its in-sample months differ slightly from "
        "`scripts/08_har_vol_target.py`, which uses in-sample data only. Sharpe difference z: Jobson-Korkie with "
        "Memmel's correction.",
        "",
        proxy.md_table(strat),
        "",
        "Sharpe ratio by decade:",
        "",
        proxy.md_table(decades),
        "",
        "## Figures",
        "",
        "![Strategies over the whole sample](fig1_full_period.png)",
        "",
        "![M x C over the whole sample](fig2_mxc_full_period.png)",
        "",
    ]
    (args.out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
