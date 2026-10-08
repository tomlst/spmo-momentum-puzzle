"""
Episode analysis of the M x C interaction from scripts/04_attribution.py.

M x C = C (momentum) - C (market): market-cap weighting within the momentum quintile minus
market-cap weighting within the S&P 500. Most of its in-sample mean comes from a few episodes
(fig8 of 04_attribution.py). For each episode in EPISODES this script reports

  - the rebalances whose portfolios were held,
  - the compounded episode return of the eight portfolios and of C (momentum), C (market), M x C,
  - the stocks that contributed most to C (momentum) and to C (market),
  - the contribution of each S&P 500 size quintile to both effects,
  - the largest holdings of the cap-weighted and the momentum-weighted quintile portfolios.

A stock's contribution to an effect in one month is the effect's formula applied to its
start-of-month weights, times its return; contributions are summed over the episode. They add up
to the sum of the monthly effect, which differs slightly from the compounded episode value.

Outputs (output/episodes/)
  summary.md                    tables per episode
  contributors.csv              every stock's contribution to C (momentum) and C (market)
  fig1_episode_effects.png      C (market), C (momentum) and M x C in each episode
  fig2_contributors.png         largest contributors to C (momentum), and both effects by size quintile

Usage:
    python scripts/05_episodes.py [--raw data/raw] [--out output/episodes]
"""
import argparse
import importlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, PercentFormatter

proxy = importlib.import_module("03_spmo_proxy")
attribution = importlib.import_module("04_attribution")  # portfolios, effect formulas, chart style

ROOT = proxy.ROOT
EPISODES = [  # (name, first month, last month): the largest |M x C| quarters and the post-COVID rebound
    ("Dot-com bust", "2000-10", "2000-12"),
    ("Post-crisis rebound", "2009-04", "2009-06"),
    ("Post-COVID rebound", "2020-04", "2021-03"),
    ("2024 mega-cap rally", "2024-01", "2024-03"),
]
EFFECTS = ["C (market)", "C (momentum)", "M x C"]
TOP = 5
FORMULA = {name: formula for _, name, formula, _ in attribution.EFFECTS}
CODES = list(attribution.PORTFOLIOS)


# ---------- Data ----------

def episode_weights(ret, forms, months):
    """Start-of-month weights of every portfolio over `months`: {code: DataFrame months x PERMNO}."""
    parts, refs = {k: [] for k in CODES}, []
    for t, hold, w_t in forms:
        m = hold.intersection(months)
        if len(m) == 0:
            continue
        refs.append(t)
        for k in CODES:
            w = w_t[k]
            hw = pd.DataFrame(proxy.hold_weights(ret.loc[hold, w.index], w), index=hold, columns=w.index)
            parts[k].append(hw.loc[m])
    cols = sorted(set().union(*(p.columns for v in parts.values() for p in v)))
    return {k: pd.concat(v).reindex(columns=cols).fillna(0.0) for k, v in parts.items()}, refs


def analyse(ret, ids, forms, name, first, last):
    months = pd.period_range(first, last, freq="M")
    w, refs = episode_weights(ret, forms, months)
    r = ret.loc[months, w["000"].columns]
    contrib = {k: w[k] * r.fillna(0.0) for k in CODES}
    monthly = pd.DataFrame({k: c.sum(axis=1) for k, c in contrib.items()})
    episode = (1 + monthly).prod() - 1
    effects = pd.Series({e: attribution.evaluate(FORMULA[e], episode) for e in EFFECTS})

    tickers = ids.loc[ids.index.get_level_values(0).isin(months), "ticker"].groupby(level=1).last()
    stocks = pd.DataFrame({
        "ticker": tickers.reindex(r.columns),
        "return": (1 + r.fillna(0.0)).prod() - 1,
        **{f"{e} contribution": attribution.evaluate(FORMULA[e], contrib).sum() for e in EFFECTS[:2]},
        **{f"{e} weight": attribution.evaluate(FORMULA[e], w).mean() for e in EFFECTS[:2]},
        **{f"w{k}": w[k].iloc[0] for k in ("010", "100", "110")},
    })
    stocks.index.name = "permno"
    # Size quintile within the S&P 500 by average cap weight over the episode (5 = largest)
    stocks["size quintile"] = pd.qcut(w["010"].mean().rank(method="first"), 5, labels=range(1, 6)).astype(int)
    return {"name": name, "months": months, "refs": refs, "episode": episode, "effects": effects,
            "monthly_sum": pd.Series({e: attribution.evaluate(FORMULA[e], monthly).sum() for e in EFFECTS}),
            "stocks": stocks}


# ---------- Report ----------

def period_label(months):
    if len(months) == 3 and months[0].month % 3 == 1:
        return f"{months[0].year}Q{(months[0].month + 2) // 3}"
    return f"{months[0].strftime('%b-%Y')} to {months[-1].strftime('%b-%Y')}"


def contributor_table(stocks, effect):
    col = f"{effect} contribution"
    top = pd.concat([stocks.nlargest(TOP, col), stocks.nsmallest(TOP, col)])
    return pd.DataFrame({
        "Stock": top["ticker"], "Episode return": top["return"],
        "Avg. weight difference": top[f"{effect} weight"], "Contribution": top[col],
    })


def size_table(stocks):
    g = stocks.groupby("size quintile")
    return pd.DataFrame({
        "S&P 500 size quintile": [f"{q}{' (smallest)' if q == 1 else ' (largest)' if q == 5 else ''}" for q in g.groups],
        "Stocks": g.size().values,
        "Avg. return": g["return"].mean().values,
        **{f"{e} contribution": g[f"{e} contribution"].sum().values for e in EFFECTS[:2]},
    })


def holdings_table(stocks):
    top = stocks.nlargest(TOP, "w110")
    return pd.DataFrame({
        "Stock": top["ticker"], "`110` weight": top["w110"], "`100` weight": top["w100"],
        "`010` weight": top["w010"], "Episode return": top["return"],
    })


def summary(results):
    lines = [
        "# M x C episodes",
        "",
        "Generated by `scripts/05_episodes.py`. M x C = C (momentum) - C (market), the interaction "
        "found in [output/attribution](../attribution/summary.md). The episodes are the three largest "
        "|M x C| quarters and the post-COVID rebound, all in-sample. Portfolio codes and formulas as "
        "in the attribution summary; returns before costs.",
        "",
        "## Overview",
        "",
        "Episode values compound each portfolio's monthly returns over the episode and then apply "
        "the effect formula.",
        "",
    ]
    overview = pd.DataFrame({
        "Episode": [f"{x['name']} ({period_label(x['months'])})" for x in results],
        "Rebalances held": [", ".join(t.strftime("%b-%Y") for t in x["refs"]) for x in results],
        "`010` S&P 500 cap weight": [x["episode"]["010"] for x in results],
        "`000` S&P 500 equal weight": [x["episode"]["000"] for x in results],
        **{e: [x["effects"][e] for x in results] for e in EFFECTS},
    })
    lines += [proxy.md_table(overview), ""]
    lines += [
        "Contributions below are summed monthly contributions; their total equals the sum of the "
        "monthly effect, shown for comparison with the compounded value:",
        "",
        proxy.md_table(pd.DataFrame({
            "Episode": [x["name"] for x in results],
            **{f"{e}, sum of months": [x["monthly_sum"][e] for x in results] for e in EFFECTS},
        })),
        "",
    ]
    for x in results:
        s = x["stocks"]
        lines += [
            f"## {x['name']} ({period_label(x['months'])})",
            "",
            "Episode return of each portfolio:",
            "",
            proxy.md_table(pd.DataFrame([x["episode"][CODES].rename(lambda k: f"`{k}`")])),
            "",
            f"C (market) {x['effects']['C (market)']:+.1%}, C (momentum) {x['effects']['C (momentum)']:+.1%}, "
            f"M x C {x['effects']['M x C']:+.1%}.",
            "",
            "**C (momentum): largest positive and negative contributors.** Weight difference is the "
            "cap-weighted minus the non-cap-weighted quintile weight, averaged over V and over the "
            "episode's months.",
            "",
            proxy.md_table(contributor_table(s, "C (momentum)")),
            "",
            "**C (market): largest positive and negative contributors.** A negative C (market) raises "
            "M x C.",
            "",
            proxy.md_table(contributor_table(s, "C (market)")),
            "",
            "**Contributions by size quintile.** Quintiles split the S&P 500 by average cap weight over "
            "the episode; C (market) is spread over many stocks, so it is best read here.",
            "",
            proxy.md_table(size_table(s)),
            "",
            "**Largest holdings of `110` (momentum x market cap) at the start of the episode.**",
            "",
            proxy.md_table(holdings_table(s)),
            "",
        ]
    lines += ["## Figures", "", "![Episode effects](fig1_episode_effects.png)", "",
              "![Contributors](fig2_contributors.png)", ""]
    return "\n".join(lines)


# ---------- Figures ----------

def note(fig, extra):
    fig.text(0.01, 0.005, "In-sample episodes. Monthly total returns before costs. Data: CRSP via WRDS. " + extra,
             ha="left", va="bottom", color=attribution.MUTED, fontsize=8.5)


def figures(results, out):
    colors = attribution.EFFECT_COLORS
    fig, ax = plt.subplots(figsize=(11, 5.5))
    pos = np.arange(len(results))
    width = 0.26
    for i, (e, c) in enumerate(zip(EFFECTS, colors)):
        v = [x["effects"][e] for x in results]
        bars = ax.bar(pos + (i - 1) * width, v, width, color=c, label=e,
                      edgecolor=attribution.SURFACE, linewidth=0.5)
        ax.bar_label(bars, labels=[f"{y:+.1%}" for y in v], padding=3, fontsize=8.5, color=attribution.INK_2)
    ax.set_xticks(pos, [f"{x['name']}\n{period_label(x['months'])}" for x in results])
    ax.axhline(0, color=attribution.INK_2, linewidth=0.8)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("Difference in episode return")
    ax.set_title("Where M x C came from in each episode: M x C = C (momentum) - C (market)")
    ax.legend(loc="upper left")
    note(fig, "Compounded episode returns. C (market) enters M x C with a minus sign.")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    attribution.save(fig, out / "fig1_episode_effects.png")

    fig, axes = plt.subplots(len(results), 2, figsize=(15, 3.8 * len(results)),
                             gridspec_kw={"width_ratios": [1.1, 1]})
    for (left, right), x in zip(axes, results):
        label = f"{x['name']} ({period_label(x['months'])})"
        t = contributor_table(x["stocks"], "C (momentum)").iloc[::-1]
        y = np.arange(len(t))
        left.barh(y, t["Contribution"], height=0.65,
                  color=[attribution.POSITIVE if v >= 0 else attribution.NEGATIVE for v in t["Contribution"]])
        left.set_yticks(y, [f"{s}  ({ret:+.0%})" for s, ret in zip(t["Stock"], t["Episode return"])])
        left.axvline(0, color=attribution.INK_2, linewidth=0.8)
        left.grid(axis="x"), left.grid(axis="y", visible=False)
        left.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
        left.set_title(f"{label}: C (momentum) {x['effects']['C (momentum)']:+.1%}")

        q = size_table(x["stocks"])
        pos = np.arange(len(q))
        for i, (e, c) in enumerate(zip(EFFECTS[:2], attribution.EFFECT_COLORS)):
            right.bar(pos + (i - 0.5) * 0.38, q[f"{e} contribution"], 0.38, color=c, label=e,
                      edgecolor=attribution.SURFACE, linewidth=0.5)
        right.set_xticks(pos, ["1\nsmallest", "2", "3", "4", "5\nlargest"])
        right.axhline(0, color=attribution.INK_2, linewidth=0.8)
        right.yaxis.set_major_locator(MaxNLocator(steps=[1, 2, 5, 10]))
        right.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        right.set_title("By S&P 500 size quintile")
        right.legend(loc="best")
    axes[-1, 0].set_xlabel(f"Contribution to C (momentum): top and bottom {TOP} stocks (episode return in brackets)")
    axes[-1, 1].set_xlabel("S&P 500 size quintile")
    fig.suptitle("Which stocks drove C (momentum), and which size groups drove C (market)", x=0.01, ha="left")
    note(fig, "Contribution = (cap-weighted - non-cap-weighted weight) x return, averaged over V and summed "
         "over months. C (market) enters M x C with a minus sign.")
    fig.tight_layout(rect=(0, 0.02, 1, 0.985))
    attribution.save(fig, out / "fig2_contributors.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--out", type=Path, default=ROOT / "output" / "episodes")
    args = ap.parse_args()

    ret, ids, forms = attribution.formations(args.raw)
    results = [analyse(ret, ids, forms, *ep) for ep in EPISODES]
    assert all(x["refs"][0] >= attribution.IS_FIRST_REF for x in results), "out-of-sample rebalance used"

    args.out.mkdir(parents=True, exist_ok=True)
    pd.concat({x["name"]: x["stocks"] for x in results}, names=["episode"]).to_csv(
        args.out / "contributors.csv", float_format="%.6f")
    figures(results, args.out)
    (args.out / "summary.md").write_text(summary(results), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
