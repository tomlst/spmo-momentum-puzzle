"""
Double sort within the S&P 500: momentum quintile x size quintile.

Tests the M x C interaction of scripts/04_attribution.py on groups of stocks instead of portfolio
weights. At each in-sample rebalance the universe of 04_attribution.py (S&P 500 members with a valid
signal) is sorted into quintiles of x = momentum / sigma, and each momentum quintile into quintiles
of size (a conditional sort, so every cell holds about 20 stocks). Each cell is an equal-weighted,
six-month buy-and-hold portfolio from t+2, as in the other scripts. Equal weights keep a single stock
from dominating a cell, unlike cap weighting (NVDA was 33% of the cap-weighted quintile in 2024Q1).

Size is measured two ways:
  current   market cap at the reference month t
  lagged    market cap at t-12, before the months over which momentum is measured; current cap
            already contains the past year's return, so sorting on it partly sorts on momentum again

The size slope is S5 (largest) - S1 (smallest) within a momentum quintile. Counterparts in the
2 x 2 x 2 attribution:
  average size slope over the five momentum quintiles   ~ C (market)
  size slope in M5, the top momentum quintile            ~ C (momentum)
  M5 slope - average slope                               ~ M x C
Signs and significance are comparable; magnitudes are not, because the attribution tilts weights
continuously while the sort compares extreme quintiles.

Outputs (output/double_sort/)
  summary.md                    tables
  cell_returns.csv              monthly returns of the 25 cells, for both size measures
  fig1_cell_returns.png         annualised mean return of each cell, current and lagged size
  fig2_size_slope.png           size slope by momentum quintile with 95% confidence intervals
  fig3_cumulative_slope.png     cumulative size slope in M5, on average, and their difference
  fig4-6_cell_sharpe_<decade>   Sharpe ratio of each cell in 1995-2004, 2005-2014 and 2015-2025

Usage:
    python scripts/06_double_sort.py [--raw data/raw] [--out output/double_sort]
"""
import argparse
import importlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator, PercentFormatter

proxy = importlib.import_module("03_spmo_proxy")
attribution = importlib.import_module("04_attribution")  # universe, signals, chart style

ROOT = proxy.ROOT
Q = 5
SIZES = {"current": "mc", "lagged": "mc_lag"}
SLOPES = [f"M{m}" for m in range(1, Q + 1)] + ["Average", "M5 - average", "M5 - M1"]
COMPARISON = [("C (market)", "Average"), ("C (momentum)", "M5"), ("M x C", "M5 - average")]
SUBPERIODS = [("1995-2014", 1995, 2014), ("2015-2025", 2015, 2025)]
DECADES = [("1995-2004", 1995, 2004), ("2005-2014", 2005, 2014), ("2015-2025", 2015, 2025)]
SEQUENTIAL = LinearSegmentedColormap.from_list("sequential", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


# ---------- Portfolios ----------

def quintile(s):
    return pd.qcut(s.rank(method="first"), Q, labels=range(1, Q + 1)).astype(int)


def cell_returns(ret, sig, size_col):
    """Monthly returns of the equal-weighted cells, columns (momentum quintile, size quintile), the
    average number of stocks per cell, and the number of stock-rebalances without a size value."""
    parts, counts, dropped = {}, [], 0
    for t, hold, d in sig:
        dropped += d[size_col].isna().sum()
        d = d.dropna(subset=[size_col])
        mq = quintile(d["x"])
        for m in range(1, Q + 1):
            g = d[mq == m]
            sq = quintile(g[size_col])
            for s in range(1, Q + 1):
                members = g.index[sq == s]
                w = pd.Series(1.0 / len(members), index=members)
                r = pd.Series(proxy.hold_returns(ret.loc[hold, members], w), index=hold)
                parts.setdefault((m, s), []).append(r)
                counts.append(len(members))
    r = pd.DataFrame({k: pd.concat(v) for k, v in parts.items()})
    r.columns = pd.MultiIndex.from_tuples(r.columns, names=["momentum", "size"])
    return r, np.mean(counts), dropped


def size_slopes(r):
    """Monthly size slope (S5 - S1) in each momentum quintile, their average, and the M5 contrasts."""
    s = pd.DataFrame({f"M{m}": r[(m, Q)] - r[(m, 1)] for m in range(1, Q + 1)})
    s["Average"] = s[[f"M{m}" for m in range(1, Q + 1)]].mean(axis=1)
    s["M5 - average"] = s["M5"] - s["Average"]
    s["M5 - M1"] = s["M5"] - s["M1"]
    return s


def stats(d):
    mean, t = attribution.mean_t(d)
    return {"mean": mean, "t": t, "half_ci": 1.96 * mean / t}


# ---------- Report ----------

def fmt(st):
    return f"{st['mean']:+.1%} ({st['t']:.2f})"


def cell_table(r):
    rows = []
    for m in range(Q, 0, -1):
        row = {"Momentum quintile": f"M{m}{' (winners)' if m == Q else ' (losers)' if m == 1 else ''}"}
        row.update({f"S{s}{' small' if s == 1 else ' large' if s == Q else ''}": 12 * r[(m, s)].mean()
                    for s in range(1, Q + 1)})
        row["S5 - S1 (t)"] = fmt(stats(r[(m, Q)] - r[(m, 1)]))
        rows.append(row)
    return pd.DataFrame(rows)


def summary(cells, slopes, attr, n_cell, dropped, period, n_rebal):
    lines = [
        "# Double sort: momentum x size within the S&P 500",
        "",
        f"Generated by `scripts/06_double_sort.py`. In-sample only: {period} ({n_rebal} semi-annual "
        "rebalances). Same universe, rebalancing and holding period as the attribution; each rebalance "
        "sorts the universe into quintiles of momentum / sigma, then each momentum quintile into size "
        "quintiles. Cells are equal-weighted. Returns before costs. Annualised means are 12 x the "
        "monthly mean; t-stats assume independent months.",
        "",
        f"Average stocks per cell: {n_cell['current']:.1f} (current size), {n_cell['lagged']:.1f} "
        f"(lagged size). Stock-rebalances without a market cap at t-12, dropped from the lagged sort: "
        f"{dropped['lagged']}.",
        "",
    ]
    for name, r in cells.items():
        what = ("market cap at the reference month" if name == "current"
                else "market cap 12 months earlier, before the momentum window")
        lines += [f"## Annualised mean return by cell: {name} size ({what})", "",
                  proxy.md_table(cell_table(r)), ""]

    table = pd.DataFrame({"Size slope, S5 - S1": SLOPES,
                          **{f"{name} size": [fmt(stats(s[k])) for k in SLOPES] for name, s in slopes.items()}})
    lines += [
        "## Size slope by momentum quintile",
        "",
        "Annualised mean of the monthly S5 - S1 return difference, t-stat in brackets. A positive "
        "slope means large stocks outperformed small stocks within that momentum quintile.",
        "",
        proxy.md_table(table),
        "",
    ]

    comp = pd.DataFrame({
        "2 x 2 x 2 attribution": [f"{a}: {fmt(attr[a])}" for a, _ in COMPARISON],
        "Double sort counterpart": [b for _, b in COMPARISON],
        **{f"{name} size": [fmt(stats(s[b])) for _, b in COMPARISON] for name, s in slopes.items()},
    })
    lines += [
        "## Comparison with the 2 x 2 x 2 attribution",
        "",
        "Signs and significance are comparable; magnitudes are not, since the attribution tilts "
        "weights continuously by market cap while the sort compares extreme size quintiles with "
        "equal weights.",
        "",
        proxy.md_table(comp),
        "",
    ]

    sub = []
    for label, a, b in SUBPERIODS:
        for name, s in slopes.items():
            part = s[(s.index.year >= a) & (s.index.year <= b)]
            sub.append({"Period": label, "Size": name,
                        **{k: fmt(stats(part[k])) for k in ["M5", "Average", "M5 - average"]}})
    lines += ["## Subperiods", "", proxy.md_table(pd.DataFrame(sub)), ""]
    lines += ["## Figures", ""]
    for f, caption in [("fig1_cell_returns.png", "Cell returns"), ("fig2_size_slope.png", "Size slope"),
                       ("fig3_cumulative_slope.png", "Cumulative size slope"),
                       *[(f"fig{4 + i}_cell_sharpe_{label.replace('-', '_')}.png", f"Cell Sharpe ratios, {label}")
                         for i, (label, _, _) in enumerate(DECADES)]]:
        lines += [f"![{caption}]({f})", ""]
    return "\n".join(lines)


# ---------- Figures ----------

def cell_stat(cells, f):
    """f(monthly returns) per cell, rows M5 .. M1 and columns S1 .. S5, for each size measure."""
    return {name: np.array([[f(r[(m, s)]) for s in range(1, Q + 1)] for m in range(Q, 0, -1)])
            for name, r in cells.items()}


def cell_means(cells):
    """Annualised mean return per cell."""
    return cell_stat(cells, lambda x: 12 * x.mean())


def cell_sharpes(cells, rf):
    """Annualised Sharpe ratio per cell (excess over the T-bill rate)."""
    def sharpe(x):
        ex = x - rf.reindex(x.index)
        return ex.mean() / ex.std() * 12 ** 0.5
    return cell_stat(cells, sharpe)


def cell_heatmaps(means, lo, hi, title, period, n_rebal, path, fmt="{:.1%}", label="Annualised mean return",
                  scale_note="Same colour scale in fig1 and the decade figures."):
    """One heatmap per size measure on the colour scale lo .. hi."""
    ink, ink2, surface = attribution.INK, attribution.INK_2, attribution.SURFACE
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (name, v), letter in zip(axes, means.items(), "ab"):
        im = ax.imshow(v, cmap=SEQUENTIAL, vmin=lo, vmax=hi)
        for i in range(Q):
            for j in range(Q):
                dark = (v[i, j] - lo) / (hi - lo) > 0.55
                ax.text(j, i, fmt.format(v[i, j]), ha="center", va="center", color=surface if dark else ink)
        ax.set_xticks(range(Q), ["S1\nsmall", "S2", "S3", "S4", "S5\nlarge"])
        ax.set_yticks(range(Q), ["M5\nwinners", "M4", "M3", "M2", "M1\nlosers"])
        ax.tick_params(length=0)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"({letter}) {name.capitalize()} size")
        ax.set_xlabel("Size quintile within the momentum quintile")
    axes[0].set_ylabel("Momentum quintile (momentum / sigma)")
    fig.subplots_adjust(left=0.08, right=0.86, bottom=0.16, top=0.86, wspace=0.3)
    cb = fig.colorbar(im, cax=fig.add_axes([0.89, 0.2, 0.015, 0.62]),
                      format=PercentFormatter(1.0, decimals=0) if "%" in fmt else None)
    cb.set_label(label, color=ink2)
    cb.outline.set_visible(False)
    fig.suptitle(title, x=0.01, ha="left")
    attribution.footnote(fig, period, n_rebal, f"Lagged size = market cap 12 months before the reference month. "
                         f"{scale_note}")
    attribution.save(fig, path)


def figures(cells, slopes, rf, out, period, n_rebal):
    ink, ink2, surface = attribution.INK, attribution.INK_2, attribution.SURFACE

    # 1. Cell returns, full period
    means = cell_means(cells)
    lo, hi = min(v.min() for v in means.values()), max(v.max() for v in means.values())
    cell_heatmaps(means, lo, hi, "Annualised mean return of each momentum x size cell (equal-weighted)",
                  period, n_rebal, out / "fig1_cell_returns.png", scale_note="")

    # 4-6. Cell Sharpe ratios by decade, on one colour scale
    decades = {}
    for label, a, b in DECADES:
        part = {k: r[(r.index.year >= a) & (r.index.year <= b)] for k, r in cells.items()}
        decades[label] = (part, cell_sharpes(part, rf))
    lo = min(v.min() for _, m in decades.values() for v in m.values())
    hi = max(v.max() for _, m in decades.values() for v in m.values())
    for i, (label, (part, m)) in enumerate(decades.items()):
        idx = part["current"].index
        cell_heatmaps(m, lo, hi, f"Sharpe ratio of each momentum x size cell, {label} (equal-weighted)",
                      f"{idx[0].strftime('%b-%Y')} to {idx[-1].strftime('%b-%Y')}",
                      int(idx.month.isin([4, 10]).sum()), out / f"fig{4 + i}_cell_sharpe_{label.replace('-', '_')}.png",
                      fmt="{:.2f}", label="Annualised Sharpe ratio (T-bill excess return)",
                      scale_note="Same colour scale in the three decade figures.")

    # 2. Size slope by momentum quintile
    keys = SLOPES[:-1]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    pos = np.arange(len(keys)) + np.array([0] * Q + [0.5, 1.0])      # gap before the summary bars
    width = 0.36
    for i, ((name, s), c) in enumerate(zip(slopes.items(), attribution.EFFECT_COLORS)):
        st = [stats(s[k]) for k in keys]
        ax.bar(pos + (i - 0.5) * width, [x["mean"] for x in st], width, color=c, label=f"{name.capitalize()} size",
               yerr=[x["half_ci"] for x in st], error_kw={"ecolor": ink2, "elinewidth": 1, "capsize": 3},
               edgecolor=surface, linewidth=0.5)
    ax.set_xticks(pos, ["M1\nlosers", "M2", "M3", "M4", "M5\nwinners", "Average\n(~ C market)",
                        "M5 - average\n(~ M x C)"])
    ax.axhline(0, color=ink2, linewidth=0.8)
    ax.yaxis.set_major_locator(MaxNLocator(steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("S5 - S1, annualised mean")
    ax.set_title("Size slope (large minus small) within each momentum quintile; bars show 95% confidence intervals")
    ax.legend(loc="upper left")
    attribution.footnote(fig, period, n_rebal, "t-stats assume independent months.")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    attribution.save(fig, out / "fig2_size_slope.png")

    # 3. Cumulative size slope, current and lagged size
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for ax, (name, s), letter in zip(axes, slopes.items(), "ab"):
        dates = s.index.to_timestamp(how="end")
        for k, c in zip(["Average", "M5", "M5 - average"], attribution.EFFECT_COLORS):
            label = {"Average": "Average over momentum quintiles (~ C market)", "M5": "M5, winners (~ C momentum)",
                     "M5 - average": "M5 - average (~ M x C)"}[k]
            ax.plot(dates, s[k].cumsum(), color=c, label=label)
        ax.axhline(0, color=ink2, linewidth=0.8)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_title(f"({letter}) {name.capitalize()} size")
        ax.legend(loc="upper left")
    axes[0].set_ylabel("Cumulative sum of monthly S5 - S1")
    fig.suptitle("Cumulative size slope", x=0.01, ha="left")
    attribution.footnote(fig, period, n_rebal)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    attribution.save(fig, out / "fig3_cumulative_slope.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--out", type=Path, default=ROOT / "output" / "double_sort")
    args = ap.parse_args()

    ret, _, sig = attribution.signals(args.raw)
    cells, n_cell, dropped = {}, {}, {}
    for name, col in SIZES.items():
        cells[name], n_cell[name], dropped[name] = cell_returns(ret, sig, col)
    slopes = {name: size_slopes(r) for name, r in cells.items()}
    index = cells["current"].index
    assert index[0] >= attribution.IS_FIRST_REF + proxy.LAG, "out-of-sample months in the results"
    period = f"{index[0].strftime('%b-%Y')} to {index[-1].strftime('%b-%Y')}"

    # The attribution's effects over the same months, for comparison
    attr_r, _ = attribution.run(args.raw)
    formula = {name: f for _, name, f, _ in attribution.EFFECTS}
    attr = {a: stats(attribution.evaluate(formula[a], attr_r)) for a, _ in COMPARISON}

    args.out.mkdir(parents=True, exist_ok=True)
    pd.concat(cells, axis=1, names=["size measure"]).rename_axis("month").to_csv(
        args.out / "cell_returns.csv", float_format="%.8f")
    ff = pd.read_parquet(args.raw / "ff5_factors_monthly.parquet", columns=["date", "rf"])
    rf = ff.set_index(ff["date"].dt.to_period("M"))["rf"].astype(float)
    figures(cells, slopes, rf, args.out, period, len(sig))
    (args.out / "summary.md").write_text(
        summary(cells, slopes, attr, n_cell, dropped, period, len(sig)), encoding="utf-8")
    print(f"Holding months {index[0]} .. {index[-1]} ({len(index)}); wrote {args.out}")


if __name__ == "__main__":
    main()
