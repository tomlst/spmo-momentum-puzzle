"""
In-sample attribution of the SPMO proxy to momentum (M), market cap (C) and volatility (V).

A full 2 x 2 x 2 design. Each portfolio is labelled by the ingredients it uses (M C V):

  000  S&P 500, equal weight
  001  S&P 500, weighted by 1 / sigma
  010  S&P 500, weighted by market cap
  011  S&P 500, weighted by market cap / sigma
  100  top quintile, weighted by momentum
  101  top quintile, weighted by momentum / sigma
  110  top quintile, weighted by momentum x market cap
  111  top quintile, weighted by momentum x market cap / sigma   (the uncapped SPMO proxy)

M = 1 selects the top quintile on momentum / sigma, as in scripts/03_spmo_proxy.py, and weights by
max(momentum, 0). All eight portfolios share that script's universe (S&P 500 members with a valid
signal), reference dates, six-month buy-and-hold from t+2 and return data; only the selection and
the weights differ.

Effects are linear combinations of monthly returns, listed in EFFECTS. A conditional effect
switches one ingredient on with M fixed, averaged over the third ingredient; an interaction is the
difference between a conditional effect with M on and with M off (for C x V: with C on and off).

Only in-sample reference dates (Feb-1995 .. Aug-2025, holding months Apr-1995 .. Dec-2025) are
used. The look-back window of the first one starts in Feb-1994.

Outputs (output/attribution/)
  summary.md                       tables and figures
  monthly_returns.csv              monthly returns of the eight portfolios and the capped proxy
  annual_returns.csv               calendar-year returns
  difference_matrix.csv            annualised mean return difference for every pair of portfolios
  fig1_annual_returns.png          calendar-year returns of the eight portfolios
  fig2_effects_by_year.png         conditional C and V effects and their interaction with M, by year
  fig3_cumulative_growth.png       growth of $1
  fig4_cumulative_effects.png      cumulative conditional C and V effects
  fig5_effects_summary.png         every effect with a 95% confidence interval
  fig6_difference_matrix.png       8 x 8 matrix of pairwise return differences
  fig7_concentration.png           effective number of stocks at each rebalance
  fig8_mxc_monthly_quarterly.png   M x C and its two components by month and by quarter
  fig9_annual_returns_momentum.png panel (b) of fig1 on its own: the momentum-quintile portfolios

Usage:
    python scripts/04_attribution.py [--raw data/raw] [--out output/attribution]
"""
import argparse
import importlib
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter

proxy = importlib.import_module("03_spmo_proxy")    # data loading, signal and holding rules

ROOT = proxy.ROOT
IS_FIRST_REF = pd.Period("1995-02", "M")
WEIGHTING = {  # (C, V) -> weighting in the S&P 500 (M = 0) and in the momentum quintile (M = 1)
    "00": ("equal weight", "momentum"),
    "01": ("1 / sigma", "momentum / sigma"),
    "10": ("market cap", "momentum x market cap"),
    "11": ("market cap / sigma", "momentum x market cap / sigma"),
}
PORTFOLIOS = {f"{m}{cv}": f"{'Top quintile' if m else 'S&P 500'}, {names[m]}"
              for m in (0, 1) for cv, names in WEIGHTING.items()}
REFERENCES = {"111 capped": "SPMO proxy with the weight cap", "SPY": "SPY ETF"}
EFFECTS = [  # (group, name, formula over portfolio codes, description)
    ("Main effects", "M", "((100 - 000) + (101 - 001) + (110 - 010) + (111 - 011)) / 4",
     "momentum selection and weighting"),
    ("Main effects", "C", "((010 - 000) + (011 - 001) + (110 - 100) + (111 - 101)) / 4",
     "market-cap weighting"),
    ("Main effects", "V", "((001 - 000) + (011 - 010) + (101 - 100) + (111 - 110)) / 4",
     "dividing weights by sigma"),
    ("Conditional effects", "C (market)", "((010 - 000) + (011 - 001)) / 2", "C within the S&P 500"),
    ("Conditional effects", "C (momentum)", "((110 - 100) + (111 - 101)) / 2",
     "C within the momentum quintile"),
    ("Conditional effects", "V (market)", "((001 - 000) + (011 - 010)) / 2", "V within the S&P 500"),
    ("Conditional effects", "V (momentum)", "((101 - 100) + (111 - 110)) / 2",
     "V within the momentum quintile"),
    ("Interactions", "M x C", "((110 - 100) + (111 - 101)) / 2 - ((010 - 000) + (011 - 001)) / 2",
     "C (momentum) - C (market)"),
    ("Interactions", "M x V", "((101 - 100) + (111 - 110)) / 2 - ((001 - 000) + (011 - 010)) / 2",
     "V (momentum) - V (market)"),
    ("Interactions", "C x V", "((011 - 010) + (111 - 110)) / 2 - ((001 - 000) + (101 - 100)) / 2",
     "V with cap weighting - V without"),
    ("Interactions", "M x C x V", "((111 - 110) - (101 - 100)) - ((011 - 010) - (001 - 000))",
     "C x V in the momentum quintile - in the S&P 500"),
]
EVENTS = [("Post-COVID rebound", "2020-04", "2021-03")]   # labelled periods in fig 8: (label, first, last month)
YEARLY_PANELS = [("C", ["C (market)", "C (momentum)", "M x C"]),
                 ("V", ["V (market)", "V (momentum)", "M x V"])]

# Chart style. Weighting colours are a validated categorical set; one colour per (C, V) pair
WEIGHT_COLORS = {"00": "#2a78d6", "10": "#eb6834", "01": "#1baf7a", "11": "#4a3aa7"}
EFFECT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
POSITIVE, NEGATIVE = "#2a78d6", "#e34948"
DIVERGING = LinearSegmentedColormap.from_list("diverging", [NEGATIVE, "#f0efec", POSITIVE])
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "figure.titlesize": 13, "figure.titleweight": "bold",
    "legend.frameon": False, "lines.linewidth": 1.8,
})


def evaluate(formula, d):
    """Evaluate an EFFECTS formula on a DataFrame whose columns are portfolio codes, so the
    documented definition is exactly what is computed."""
    return eval(re.sub(r"\b([01]{3})\b", r"d['\1']", formula), {"d": d})


# ---------- Portfolios ----------

def portfolio_weights(x, momentum, sigma, mc):
    """Initial weights at one reference date. All inputs are indexed by the universe."""
    top = x.sort_values(ascending=False).index[:int(round(proxy.QUINTILE * len(x)))]
    m, s, c = momentum[top].clip(lower=0), sigma[top], mc[top]
    if m.sum() <= 0:
        raise ValueError("No selected stock has positive momentum")
    raw = {"000": pd.Series(1.0, index=x.index), "001": 1 / sigma, "010": mc, "011": mc / sigma,
           "100": m, "101": m / s, "110": m * c, "111": m * c / s}
    w = {k: v / v.sum() for k, v in raw.items()}
    w["111 capped"] = proxy.capped_weights(c, x[top].clip(lower=0))
    return w


def signals(raw):
    """Monthly returns and identifiers, and for each in-sample rebalance (reference month, holding
    months, DataFrame over the universe with x, momentum, sigma, market cap and market cap at t-12)."""
    ret, cap, ids, month_end = proxy.load_monthly(raw)
    mem = pd.read_parquet(raw / "sp500_membership.parquet", columns=["permno", "mbrstartdt", "mbrenddt"])
    momentum, sigma = proxy.momentum_and_risk(ret, proxy.daily_sums(raw, ret.index))
    x_all = momentum / sigma
    forms = [t for t in ret.index if t.month in proxy.REF_MONTHS and t >= IS_FIRST_REF
             and t + proxy.LAG <= ret.index[-1]]
    out = []
    for t in forms:
        u = x_all.loc[t, proxy.universe(ret, mem, month_end, t)].dropna().index
        d = pd.DataFrame({"x": x_all.loc[t, u], "momentum": momentum.loc[t, u], "sigma": sigma.loc[t, u],
                          "mc": cap.loc[t, u], "mc_lag": cap.loc[t - proxy.WINDOW, u]})
        if d["mc"].isna().any():
            raise ValueError(f"{t}: market cap missing for {d['mc'].isna().sum()} stocks")
        hold = pd.period_range(t + proxy.LAG, min(t + proxy.LAG + proxy.HOLD - 1, ret.index[-1]), freq="M")
        out.append((t, hold, d))
    return ret, ids, out


def formations(raw):
    """Monthly returns and identifiers, and (reference month, holding months, initial weights of
    every portfolio) for each in-sample rebalance."""
    ret, ids, sig = signals(raw)
    return ret, ids, [(t, hold, portfolio_weights(d["x"], d["momentum"], d["sigma"], d["mc"]))
                      for t, hold, d in sig]


def run(raw):
    """Monthly returns of the portfolios, and the effective number of stocks at each rebalance."""
    ret, _, forms = formations(raw)
    returns, eff_n = {}, {}
    for t, hold, w_t in forms:
        for k, w in w_t.items():
            returns.setdefault(k, []).append(pd.Series(proxy.hold_returns(ret.loc[hold, w.index], w), index=hold))
        eff_n[t] = {k: 1 / (w ** 2).sum() for k, w in w_t.items() if k in PORTFOLIOS}
    rets = pd.DataFrame({k: pd.concat(v) for k, v in returns.items()})
    rets["SPY"] = proxy.etf_monthly(raw, "spy").reindex(rets.index)
    return rets, pd.DataFrame(eff_n).T


# ---------- Statistics ----------

def annual(r):
    return (1 + r).groupby(r.index.year).prod() - 1


def quarterly(r):
    return (1 + r).groupby(r.index.asfreq("Q")).prod() - 1


def mean_t(d):
    """Annualised mean and t-stat (independent months) of a monthly series."""
    return 12 * d.mean(), d.mean() / (d.std() / len(d) ** 0.5)


def performance(r, rf):
    ex = r.sub(rf, axis=0)
    wealth = (1 + r).cumprod()
    return pd.DataFrame({
        "Annual return": (1 + r).prod() ** (12 / len(r)) - 1,
        "Annual volatility": r.std() * 12 ** 0.5,
        "Sharpe ratio": ex.mean() / ex.std() * 12 ** 0.5,
        "Max drawdown": (wealth / wealth.cummax() - 1).min(),
    })


def effect_stats(r, yearly):
    rows = []
    for group, name, formula, desc in EFFECTS:
        mean, t = mean_t(evaluate(formula, r))
        rows.append({"Group": group, "Effect": name, "Formula": formula, "Description": desc,
                     "mean": mean, "half_ci": 1.96 * mean / t, "t": t,
                     "pos_years": (evaluate(formula, yearly) > 0).mean()})
    return pd.DataFrame(rows).set_index("Effect")


def difference_matrix(r):
    """Annualised mean and t-stat of (row - column) for every pair of portfolios."""
    codes = list(PORTFOLIOS)
    mean = pd.DataFrame(np.nan, index=codes, columns=codes)
    t = mean.copy()
    for a in codes:
        for b in codes:
            if a != b:
                mean.loc[a, b], t.loc[a, b] = mean_t(r[a] - r[b])
    return mean, t


# ---------- Figures ----------

def label(code):
    return f"{code}  {WEIGHTING[code[1:]][int(code[0])]}"


def footnote(fig, period, n_rebal, extra=""):
    text = (f"In-sample: {period}, {n_rebal} semi-annual rebalances. Monthly total returns before "
            f"costs. Data: CRSP via WRDS.{' ' + extra if extra else ''}")
    fig.text(0.01, 0.005, text, ha="left", va="bottom", color=MUTED, fontsize=8.5)


def save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def grouped_bars(ax, df, colors):
    n = df.shape[1]
    width = 0.84 / n
    pos = np.arange(len(df))
    for i, (col, c) in enumerate(zip(df, colors)):
        ax.bar(pos + (i - (n - 1) / 2) * width, df[col], width, color=c, label=col,
               edgecolor=SURFACE, linewidth=0.5)
    ax.set_xticks(pos, [f"{y}*" if i == 0 else str(y) for i, y in enumerate(df.index)], rotation=90)
    ax.set_xlim(-0.6, len(df) - 0.4)
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.yaxis.set_major_locator(MaxNLocator(steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))


def panels_by_m(axes):
    """Title one panel per universe and return the portfolio codes it holds."""
    for ax, title in zip(axes, ["(a) S&P 500 universe (M = 0)", "(b) Top momentum quintile (M = 1)"]):
        ax.set_title(title)
    return [[f"{m}{cv}" for cv in WEIGHTING] for m in (0, 1)]


def figures(r, yearly, stats, matrix, eff_n, out, period):
    n = len(eff_n)
    codes = list(PORTFOLIOS)
    colors = [WEIGHT_COLORS[cv] for cv in WEIGHTING]

    # 1. Calendar-year returns, one panel per universe
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True, sharey=True)
    for ax, group in zip(axes, panels_by_m(axes)):
        grouped_bars(ax, yearly[group].rename(columns=label), colors)
        ax.set_ylabel("Calendar-year return")
        ax.legend(ncol=4, loc="upper left")
    fig.suptitle("Calendar-year returns of the eight portfolios (colour = weighting scheme)", x=0.01, ha="left")
    footnote(fig, period, n, "* 1995 covers Apr-Dec.")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save(fig, out / "fig1_annual_returns.png")

    # 9. Calendar-year returns of the momentum quintile alone (panel b of fig1)
    group = [f"1{cv}" for cv in WEIGHTING]
    fig, ax = plt.subplots(figsize=(16, 5.5))
    grouped_bars(ax, yearly[group].rename(columns=label), colors)
    ax.set_ylabel("Calendar-year return")
    ax.legend(ncol=4, loc="upper left")
    ax.set_title("Calendar-year returns of the top momentum quintile (M = 1) by weighting scheme")
    footnote(fig, period, n, "* 1995 covers Apr-Dec.")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, out / "fig9_annual_returns_momentum.png")

    # 2. Conditional effects and interactions by year
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    for ax, (factor, names), letter in zip(axes, YEARLY_PANELS, "ab"):
        df = pd.DataFrame({f"{k}: {stats.loc[k, 'Description']}": evaluate(stats.loc[k, "Formula"], yearly)
                           for k in names})
        grouped_bars(ax, df, EFFECT_COLORS)
        what = "market-cap weighting" if factor == "C" else "dividing weights by sigma"
        ax.set_title(f"({letter}) {factor}: {what}")
        ax.set_ylabel("Difference in calendar-year return")
        ax.legend(ncol=3, loc="upper left")
    fig.suptitle("Does each weighting ingredient help in the S&P 500 and in the momentum quintile?",
                 x=0.01, ha="left")
    footnote(fig, period, n, "Differences of calendar-year returns; formulas in summary.md. * 1995 covers Apr-Dec.")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save(fig, out / "fig2_effects_by_year.png")

    # 3. Growth of $1
    dates = r.index.to_timestamp(how="end")
    growth = (1 + r[codes]).cumprod()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), sharey=True)
    for ax, group in zip(axes, panels_by_m(axes)):
        for k in group:
            ax.plot(dates, growth[k], color=WEIGHT_COLORS[k[1:]], label=f"{label(k)}  (${growth[k].iloc[-1]:.0f})")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}"))
        ax.legend(loc="upper left", title="Final value of $1 in brackets", title_fontsize=9, alignment="left")
    axes[0].set_ylabel("Growth of $1 (log scale)")
    fig.suptitle("Growth of $1 invested in Apr-1995", x=0.01, ha="left")
    footnote(fig, period, n)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, out / "fig3_cumulative_growth.png")

    # 4. Cumulative conditional effects
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), sharey=True)
    for ax, (factor, names), letter in zip(axes, YEARLY_PANELS, "ab"):
        for k, c in zip(names, EFFECT_COLORS):
            ax.plot(dates, evaluate(stats.loc[k, "Formula"], r).cumsum(), color=c,
                    label=f"{k}: {stats.loc[k, 'Description']}")
        ax.axhline(0, color=INK_2, linewidth=0.8)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        what = "market-cap weighting" if factor == "C" else "dividing weights by sigma"
        ax.set_title(f"({letter}) {factor}: {what}")
        ax.legend(loc="upper left")
    axes[0].set_ylabel("Cumulative sum of monthly return differences")
    fig.suptitle("Cumulative effect of each weighting ingredient", x=0.01, ha="left")
    footnote(fig, period, n, "Formulas in summary.md.")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, out / "fig4_cumulative_effects.png")

    # 5. Summary of all effects, grouped
    headers, ticks, pos, y = [], [], [], 0
    for group, g in stats.groupby("Group", sort=False):
        y += 1.4                                             # gap before each group header
        headers.append((group, y))
        y -= 0.3                                             # first row sits close to its header
        for k, row in g.iterrows():
            y += 1
            pos.append(y)
            ticks.append(f"{k}: {row['Description']}")
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(pos, stats["mean"], height=0.6, color=WEIGHT_COLORS["00"], xerr=stats["half_ci"],
            error_kw={"ecolor": INK_2, "elinewidth": 1.2, "capsize": 4})
    ax.set_yticks(pos, ticks)
    for group, gy in headers:
        ax.text(-0.01, gy, group, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontweight="bold", color=INK)
    for p, m, h, t in zip(pos, stats["mean"], stats["half_ci"], stats["t"]):
        ax.annotate(f"{m:+.1%}  (t = {t:.2f})", (m + h, p), xytext=(8, 0), textcoords="offset points",
                    va="center", color=INK_2)
    ax.set_ylim(y + 0.7, 0.6)                                # inverted: first group at the top
    ax.axvline(0, color=INK_2, linewidth=0.8)
    ax.grid(axis="x"), ax.grid(axis="y", visible=False)
    ax.set_xlim(right=(stats["mean"] + stats["half_ci"]).max() + 0.035)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Annualised mean of monthly return differences; bars show 95% confidence intervals")
    ax.set_title("Effects of momentum (M), market cap (C) and volatility (V)", pad=14)
    footnote(fig, period, n, "t-stats assume independent months. Formulas in summary.md.")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, out / "fig5_effects_summary.png")

    # 6. Pairwise difference matrix
    mean, t = matrix
    lim = np.nanmax(np.abs(mean.to_numpy()))
    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(mean.to_numpy(dtype=float), cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim, lim))
    for i, a in enumerate(codes):
        for j, b in enumerate(codes):
            if a == b:
                ax.text(j, i, "-", ha="center", va="center", color=MUTED)
                continue
            m = mean.loc[a, b]
            ink = SURFACE if abs(m) > 0.6 * lim else INK
            ax.text(j, i, f"{m:+.1%}\n({t.loc[a, b]:.2f})", ha="center", va="center", color=ink,
                    fontsize=9, fontweight="bold" if abs(t.loc[a, b]) >= 1.96 else "normal")
    ax.set_xticks(range(len(codes)), codes)
    ax.set_yticks(range(len(codes)), [label(k) for k in codes])
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axhline(3.5, color=SURFACE, linewidth=3), ax.axvline(3.5, color=SURFACE, linewidth=3)
    ax.set_xlabel("Column portfolio")
    ax.set_ylabel("Row portfolio")
    cb = fig.colorbar(im, ax=ax, shrink=0.8, format=PercentFormatter(1.0, decimals=0))
    cb.set_label("Annualised mean return difference, row - column", color=INK_2)
    cb.outline.set_visible(False)
    ax.set_title("Pairwise return differences, row minus column (t-stat; bold if |t| >= 1.96)")
    footnote(fig, period, n, "Codes are M C V: 000-011 hold the S&P 500, 100-111 the momentum quintile.")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, out / "fig6_difference_matrix.png")

    # 7. Concentration
    ref_dates = eff_n.index.to_timestamp(how="end")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), sharey=True)
    for ax, group in zip(axes, panels_by_m(axes)):
        for k in group:
            ax.step(ref_dates, eff_n[k], where="post", color=WEIGHT_COLORS[k[1:]],
                    label=f"{label(k)}  (average {eff_n[k].mean():.0f})")
        ax.set_yscale("log")
        ax.set_ylim(1.5, 800)                                # room for the legend below the lines
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.legend(loc="lower left")
    axes[0].set_ylabel("Effective number of stocks, 1 / sum(w^2) (log scale)")
    fig.suptitle("Concentration of the initial weights at each rebalance", x=0.01, ha="left")
    footnote(fig, period, n)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, out / "fig7_concentration.png")

    # 8. M x C and its two components, by month and by quarter, on a common scale per column
    rows = [("M x C", ""), ("C (momentum)", ""), ("C (market)", "subtracted")]
    columns = [("Monthly", r[codes], 25, 45), ("Quarterly", quarterly(r[codes]), 80, 40)]  # widths in days
    fig, axes = plt.subplots(3, 2, figsize=(18, 12), sharex="col", sharey="col")
    for j, (freq, d, width, pad) in enumerate(columns):
        x = d.index.to_timestamp(how="start")
        mxc = evaluate(stats.loc["M x C", "Formula"], d)
        top = mxc.abs().nlargest(3).index                    # shaded in every row of the column
        for i, (name, desc) in enumerate(rows):
            ax = axes[i, j]
            v = evaluate(stats.loc[name, "Formula"], d)
            for p in top:
                ax.axvspan(p.to_timestamp(how="start") - pd.Timedelta(days=pad),
                           p.to_timestamp(how="end") + pd.Timedelta(days=pad), color=GRID, zorder=0)
            for event, first, last in EVENTS:
                a, b = pd.Period(first, "M").to_timestamp(how="start"), pd.Period(last, "M").to_timestamp(how="end")
                ax.axvspan(a, b, color=GRID, zorder=0)
                if i == 0:
                    ax.text(a + (b - a) / 2, 0.97, f"{event}\n{pd.Period(first, 'M').strftime('%b-%Y')} to "
                            f"{pd.Period(last, 'M').strftime('%b-%Y')}", transform=ax.get_xaxis_transform(),
                            ha="center", va="top", color=INK_2, fontsize=8.5)
            ax.bar(x, v, width=width, align="edge", color=[POSITIVE if y >= 0 else NEGATIVE for y in v])
            ax.axhline(0, color=INK_2, linewidth=0.8)
            ax.axhline(v.mean(), color=INK, linewidth=1, linestyle="--")
            ax.yaxis.set_major_locator(MaxNLocator(steps=[1, 2, 5, 10]))
            ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
            ax.set_title(f"({'abcdef'[2 * i + j]}) {freq}: {name}{', ' + desc if desc else ''}")
            ax.text(1.0, 1.01, f"mean {v.mean():+.2%} (dashed), positive in {(v > 0).mean():.0%} of {len(v)}",
                    transform=ax.transAxes, ha="right", va="bottom", color=INK_2, fontsize=9)
            if i == 0:
                for p in top:
                    ax.annotate(f"{p}: {mxc[p]:+.1%}", (p.to_timestamp(how="start"), mxc[p]),
                                xytext=(6, -4 if mxc[p] > 0 else 4), textcoords="offset points",
                                va="top" if mxc[p] > 0 else "bottom", color=INK_2, fontsize=9)
        axes[1, j].set_ylabel(f"Difference in {freq.lower()} return")
    fig.suptitle("M x C and its two components: M x C = C (momentum quintile) - C (S&P 500)", x=0.01, ha="left")
    footnote(fig, period, n, f"\nC (momentum) = {stats.loc['C (momentum)', 'Formula']}; C (market) = "
             f"{stats.loc['C (market)', 'Formula']}. Quarterly values use compounded quarterly portfolio returns. "
             "Shading marks the three largest |M x C| periods and the labelled post-COVID rebound.")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    save(fig, out / "fig8_mxc_monthly_quarterly.png")


# ---------- Report ----------

def summary(r, stats, matrix, eff_n, rf, period):
    perf = performance(r, rf)
    names = {**PORTFOLIOS, **REFERENCES}
    table = pd.DataFrame({
        "Portfolio": [f"`{k}` {names[k]}" for k in perf.index],
        "Annual return": perf["Annual return"], "Annual volatility": perf["Annual volatility"],
        "Sharpe ratio": perf["Sharpe ratio"].map("{:.2f}".format),
        "Max drawdown": perf["Max drawdown"],
        "Avg. effective N": [f"{eff_n[k].mean():.0f}" if k in eff_n else "" for k in perf.index],
    })
    eff = pd.DataFrame({
        "Group": stats["Group"], "Effect": stats.index, "Description": stats["Description"],
        "Formula": stats["Formula"].map("`{}`".format), "Annualised mean": stats["mean"],
        "t-stat": stats["t"].map("{:.2f}".format), "Years positive": stats["pos_years"],
    })
    mean, t = matrix
    mat = pd.DataFrame({"Row - column": [f"`{a}`" for a in mean.index],
                        **{f"`{b}`": ["" if a == b else f"{mean.loc[a, b]:+.1%} ({t.loc[a, b]:.2f})"
                                      for a in mean.index] for b in mean.columns}})
    lines = [
        "# Attribution: momentum, market cap and volatility",
        "",
        f"Generated by `scripts/04_attribution.py`. In-sample only: {period} ({len(r)} months, "
        f"{len(eff_n)} semi-annual rebalances). Codes give the ingredients used: **M**omentum, market "
        "**C**ap, **V**olatility. M = 1 selects the top quintile on momentum / sigma and weights by "
        "momentum; C = 1 multiplies the weights by market cap; V = 1 divides them by sigma. "
        "Returns are before costs.",
        "",
        "## Portfolios",
        "",
        "Return is the compound annual growth rate. Sharpe ratio uses the one-month T-bill rate. "
        "Effective N is 1 / sum(w^2) of the weights set at each rebalance, averaged over rebalances.",
        "",
        proxy.md_table(table),
        "",
        "`111` is the SPMO proxy without the weight cap, so that neighbouring portfolios differ in "
        "one ingredient only. `111 capped` and SPY are shown for reference.",
        "",
        "## Effects",
        "",
        "Linear combinations of monthly returns. The annualised mean is 12 x the monthly mean; the "
        "t-stat assumes independent months. Years positive is the share of calendar years in which "
        "the same combination of annual returns is positive.",
        "",
        proxy.md_table(eff),
        "",
        "## Pairwise differences",
        "",
        "Annualised mean of the monthly return difference, row minus column, with the t-stat in "
        "brackets. Also in `difference_matrix.csv`.",
        "",
        proxy.md_table(mat),
        "",
        "## Figures",
        "",
    ]
    for f, caption in [
        ("fig1_annual_returns.png", "Calendar-year returns"),
        ("fig2_effects_by_year.png", "Conditional effects by year"),
        ("fig3_cumulative_growth.png", "Growth of $1"),
        ("fig4_cumulative_effects.png", "Cumulative conditional effects"),
        ("fig5_effects_summary.png", "Summary of effects"),
        ("fig6_difference_matrix.png", "Pairwise difference matrix"),
        ("fig7_concentration.png", "Concentration"),
        ("fig8_mxc_monthly_quarterly.png", "M x C and its components by month and by quarter"),
        ("fig9_annual_returns_momentum.png", "Calendar-year returns, momentum quintile"),
    ]:
        lines += [f"![{caption}]({f})", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--out", type=Path, default=ROOT / "output" / "attribution")
    args = ap.parse_args()

    r, eff_n = run(args.raw)
    assert r.index[0] >= IS_FIRST_REF + proxy.LAG, "out-of-sample months in the results"
    period = f"{r.index[0].strftime('%b-%Y')} to {r.index[-1].strftime('%b-%Y')}"
    ff = pd.read_parquet(args.raw / "ff5_factors_monthly.parquet", columns=["date", "rf"])
    rf = ff.set_index(ff["date"].dt.to_period("M"))["rf"].astype(float).reindex(r.index)
    codes = list(PORTFOLIOS)
    yearly = annual(r)
    stats = effect_stats(r[codes], yearly[codes])
    matrix = difference_matrix(r[codes])

    args.out.mkdir(parents=True, exist_ok=True)
    for old in args.out.glob("fig*.png"):                    # the figure set may have been renamed
        old.unlink()
    # SPY is a single CRSP security; its return series is licensed data and is not exported
    r.drop(columns="SPY").rename_axis("month").to_csv(args.out / "monthly_returns.csv", float_format="%.8f")
    yearly.drop(columns="SPY").rename_axis("year").to_csv(args.out / "annual_returns.csv", float_format="%.8f")
    matrix[0].rename_axis("row - column").to_csv(args.out / "difference_matrix.csv", float_format="%.6f")
    figures(r, yearly, stats, matrix, eff_n, args.out, period)
    (args.out / "summary.md").write_text(summary(r, stats, matrix, eff_n, rf, period), encoding="utf-8")
    print(f"Holding months {r.index[0]} .. {r.index[-1]} ({len(r)}); wrote {args.out}")


if __name__ == "__main__":
    main()
