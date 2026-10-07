"""
Build the SPMO proxy and measure how closely it tracks the SPMO ETF.

The proxy keeps the core rules of the S&P 500 Momentum Index and simplifies the rest (see
introduction.md, "Replicating SPMO with CRSP"). At each reference month t (end of February and
August):

  1. Universe   S&P 500 members at the end of month t that have a return in month t.
  2. Momentum   compounded monthly total return over months t-12 .. t-1 (all 12 required).
  3. Risk       annualised std of daily total returns over the same months (>= 150 days).
  4. Signal     x = momentum / risk.
  5. Select     top 20% of the universe by x (count rounded), no buffer.
  6. Weight     w = market cap x max(x, 0), normalised (CRSP total market cap).
  7. Cap        `proxy_capped` only: w <= min(9%, 3 x market-cap weight among the selected),
                excess redistributed pro rata until no weight exceeds its cap. If every stock
                with a positive score is capped (Feb-2009 only), the remainder goes to the other
                selected stocks in proportion to market cap. The cap is applied per PERMNO.
  8. Hold       buy-and-hold for six months from t+2; weights drift with returns, and a stock with
                no return (delisted) drops out with its weight spread pro rata.
  9. Returns    CRSP monthly total returns incl. delisting returns, before costs and fees.

Outputs
  data/processed/spmo_proxy_returns.csv   monthly returns of both versions, Apr-1975 .. Dec-2025
  reports/spmo_replication.md             fit to SPMO over its live history

The report covers only the SPMO period (Nov-2015 onward). Out-of-sample results (reference dates
Feb-1975 .. Aug-1994) are deliberately not reported here.

Usage:
    python scripts/03_spmo_proxy.py [--raw data/raw] [--out data/processed]
                                    [--report reports/spmo_replication.md]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
END = pd.Timestamp("2025-12-31")
REF_MONTHS = (2, 8)            # reference months: February and August
LAG = 2                        # holding starts two months after the reference month
HOLD = 6                       # months held
WINDOW = 12                    # momentum and risk window: months t-12 .. t-1
MIN_DAYS = 150                 # minimum daily returns in the window
QUINTILE = 0.20
MAX_WEIGHT, MAX_MULTIPLE = 0.09, 3.0
FIT_START = pd.Period("2015-11", "M")   # first full month of SPMO returns


# ---------- Data ----------

def load_monthly(raw):
    """Monthly total returns and market caps (month x PERMNO), PERMCO and ticker per (month, PERMNO),
    and the last trading date of each month."""
    m = pd.read_parquet(raw / "crsp_monthly.parquet",
                        columns=["permno", "permco", "ticker", "mthcaldt", "mthret", "mthcap"])
    m = m[m["mthcaldt"] <= END]
    m["ym"] = m["mthcaldt"].dt.to_period("M")
    ret = m.pivot(index="ym", columns="permno", values="mthret").astype(float)
    cap = m.pivot(index="ym", columns="permno", values="mthcap").astype(float)
    ids = m.set_index(["ym", "permno"])[["permco", "ticker"]]
    return ret, cap, ids, m.groupby("ym")["mthcaldt"].max()


def daily_sums(raw, index):
    """Per (month, PERMNO): sum of daily total returns, sum of squares, and number of days."""
    parts = []
    for y in range(index[0].year, index[-1].year + 1):
        d = pd.read_parquet(raw / "crsp_daily" / f"{y}.parquet",
                            columns=["permno", "dlycaldt", "dlyret"]).dropna()
        d["r"] = d["dlyret"].astype(float)
        d["r2"] = d["r"] ** 2
        d["ym"] = d["dlycaldt"].dt.to_period("M")
        parts.append(d.groupby(["ym", "permno"])[["r", "r2"]].agg(["sum", "count"]))
    g = pd.concat(parts)
    return {k: g[c].unstack().reindex(index).fillna(0.0)
            for k, c in [("s1", ("r", "sum")), ("s2", ("r2", "sum")), ("n", ("r", "count"))]}


def etf_monthly(raw, name):
    """Monthly total returns of an ETF, compounded from CRSP daily returns."""
    d = pd.read_parquet(raw / f"{name}_daily.parquet", columns=["date", "dlyret"]).dropna()
    r = np.log1p(d["dlyret"].astype(float)).groupby(d["date"].dt.to_period("M")).sum()
    return np.expm1(r)


# ---------- Signal (steps 2-4) ----------

def signal(ret, daily):
    """x = momentum / annualised risk for every (t, PERMNO), both over months t-12 .. t-1."""
    growth = 1 + ret.shift(1)
    for k in range(2, WINDOW + 1):
        growth = growth * (1 + ret.shift(k))                 # NaN anywhere in the window -> NaN
    momentum = growth - 1
    s1, s2, n = (daily[k].shift(1).rolling(WINDOW).sum() for k in ("s1", "s2", "n"))
    n = n.where(n >= MIN_DAYS)
    var = (s2 - s1 ** 2 / n) / (n - 1)                       # sample variance of daily returns
    return momentum / np.sqrt(252 * var)


# ---------- Universe, weights, holding (steps 1, 5-8) ----------

def universe(ret, mem, month_end, t):
    """S&P 500 members on the last trading day of month t that have a return in month t."""
    date = month_end[t]
    on = mem.loc[(mem["mbrstartdt"] <= date) & (mem["mbrenddt"] >= date), "permno"].unique()
    u = ret.columns.intersection(on)
    return u[ret.loc[t, u].notna().to_numpy()]


def capped_weights(mc, score):
    """Market cap x score, each weight <= min(9%, 3 x its market-cap weight among the selected)."""
    w = mc * score
    w = w / w.sum()
    limit = np.minimum(MAX_WEIGHT, MAX_MULTIPLE * mc / mc.sum())
    if limit.sum() < 1:
        raise ValueError(f"Caps sum to {limit.sum():.3f} < 1; the weight constraint cannot be met")
    capped = pd.Series(False, index=w.index)
    for _ in range(100):
        over = w > limit + 1e-12
        if not over.any():
            break
        capped |= over                                       # fix these at their cap ...
        free = w.where(~capped, 0.0)
        if free.sum() <= 0:                                  # all positive-score stocks capped:
            free = mc.where(~capped, 0.0)                    # spread the rest by market cap
        excess = 1 - limit[capped].sum()
        w = limit.where(capped, free / free.sum() * excess)  # ... and rescale the rest
    return w


def hold_returns(r, w):
    """Monthly returns of a buy-and-hold basket; stocks with a missing return drop out pro rata."""
    r, w = r.to_numpy(), w.to_numpy().copy()
    out = np.empty(len(r))
    for i, r_m in enumerate(r):
        w = np.where(np.isnan(r_m), 0.0, w)
        w = w / w.sum()
        r_m = np.nan_to_num(r_m)
        out[i] = w @ r_m
        w = w * (1 + r_m)                                    # weights drift with returns
    return out


# ---------- Backtest ----------

def run(raw):
    """Return (monthly returns of both versions, initial weights per reference month)."""
    ret, cap, ids, month_end = load_monthly(raw)
    mem = pd.read_parquet(raw / "sp500_membership.parquet", columns=["permno", "mbrstartdt", "mbrenddt"])
    x_all = signal(ret, daily_sums(raw, ret.index))
    forms = [t for t in ret.index if t.month in REF_MONTHS and t + LAG <= ret.index[-1]
             and x_all.loc[t].notna().any()]
    returns, weights = {"proxy": [], "proxy_capped": []}, []
    for t in forms:
        x = x_all.loc[t, universe(ret, mem, month_end, t)].dropna()
        top = x.sort_values(ascending=False).index[:int(round(QUINTILE * len(x)))]
        mc, score = cap.loc[t, top], x[top].clip(lower=0)
        w_t = {"proxy": mc * score / (mc * score).sum(), "proxy_capped": capped_weights(mc, score)}
        hold = pd.period_range(t + LAG, min(t + LAG + HOLD - 1, ret.index[-1]), freq="M")
        for k, w in w_t.items():
            returns[k].append(pd.Series(hold_returns(ret.loc[hold, w.index], w), index=hold))
        weights.append(pd.DataFrame({"ref": t, "permno": top, **ids.loc[t].loc[top].to_dict("list"),
                                     "n_universe": len(x), **{k: w.to_numpy() for k, w in w_t.items()}}))
    return pd.DataFrame({k: pd.concat(v) for k, v in returns.items()}), pd.concat(weights)


# ---------- Report ----------

def md_table(df, floatfmt="{:.1%}"):
    def fmt(v):
        if isinstance(v, float):
            return "" if np.isnan(v) else floatfmt.format(v)
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def cagr(r):
    return (1 + r).prod() ** (12 / len(r)) - 1


def report(rets, weights, raw):
    spmo, spy = etf_monthly(raw, "spmo"), etf_monthly(raw, "spy")
    months = rets.index[rets.index >= FIT_START].intersection(spmo.index)
    p, b = rets.loc[months], spmo.loc[months]
    lines = [
        "# SPMO replication",
        "",
        "Generated by `scripts/03_spmo_proxy.py`. The proxy is defined in "
        "[introduction.md](../introduction.md#replicating-spmo-with-crsp). SPMO and SPY monthly returns "
        "are compounded from CRSP daily total returns; SPMO's are after its expense ratio, the proxy's "
        "are before costs.",
        "",
        f"## Fit to SPMO ({months[0].strftime('%b-%Y')} to {months[-1].strftime('%b-%Y')}, "
        f"{len(months)} months)",
        "",
        "Return is the compound annual growth rate; volatility and tracking error are annualised from "
        "monthly returns (x sqrt(12)).",
        "",
    ]
    rows = [{"": f"`{k}`", "Correlation with SPMO": f"{p[k].corr(b):.3f}",
             "Tracking error": (p[k] - b).std() * 12 ** 0.5, "Annual return": cagr(p[k]),
             "Annual volatility": p[k].std() * 12 ** 0.5} for k in p]
    for name, r in [("SPMO", b), ("SPY", spy.loc[months])]:
        rows.append({"": name, "Correlation with SPMO": "1.000" if name == "SPMO" else f"{r.corr(b):.3f}",
                     "Tracking error": np.nan if name == "SPMO" else (r - b).std() * 12 ** 0.5,
                     "Annual return": cagr(r), "Annual volatility": r.std() * 12 ** 0.5})
    lines += [md_table(pd.DataFrame(rows)), ""]

    yearly = pd.concat({"SPMO": b, **{f"`{k}`": p[k] for k in p}, "SPY": spy.loc[months]}, axis=1)
    yearly = yearly.groupby(yearly.index.year).apply(lambda g: (1 + g).prod() - 1)
    yearly.insert(0, "Year", [f"{y} (Nov-Dec)" if y == months[0].year else str(y) for y in yearly.index])
    yearly["`proxy_capped` - SPMO"] = yearly["`proxy_capped`"] - yearly["SPMO"]
    lines += ["## Calendar-year returns", "", md_table(yearly), ""]

    # Portfolios whose holding period overlaps the fit window
    w = weights[weights["ref"] + LAG + HOLD - 1 >= months[0]]
    by_ref = w.groupby("ref")
    char = pd.DataFrame({
        "": ["Universe size", "Stocks held", "Largest weight", "Top-10 weight", "Effective number of stocks"],
        **{f"`{k}`": [f"{by_ref['n_universe'].first().mean():.0f}", f"{by_ref.size().mean():.0f}",
                      by_ref[k].max().mean(), by_ref[k].apply(lambda s: s.nlargest(10).sum()).mean(),
                      f"{(1 / by_ref[k].apply(lambda s: (s ** 2).sum())).mean():.0f}"] for k in p},
    })
    lines += [
        f"## Portfolio characteristics ({by_ref.ngroups} reference dates, "
        f"{by_ref.ngroups and w['ref'].min().strftime('%b-%Y')} to {w['ref'].max().strftime('%b-%Y')})",
        "",
        "Averages over reference dates of the weights set at each rebalance. Effective number of stocks "
        "is 1 / sum(w^2).",
        "",
        md_table(char),
        "",
    ]

    # The cap is applied per PERMNO; the index applies it per company
    multi = w[w.duplicated(["ref", "permco"], keep=False)]
    combined = multi.groupby(["ref", "permco"]).agg(
        tickers=("ticker", lambda s: " + ".join(sorted(s))), weight=("proxy_capped", "sum"))
    lines += [
        "## Share classes of the same company",
        "",
        "The proxy applies the cap to each PERMNO, whereas the index applies it per company. "
        f"In {combined.index.get_level_values('ref').nunique()} of {by_ref.ngroups} reference dates, "
        "the proxy holds two share classes of one company (same PERMCO).",
        "",
    ]
    over = combined[combined["weight"] > MAX_WEIGHT + 1e-9].reset_index()
    lines += [
        f"Their combined `proxy_capped` weight exceeds {MAX_WEIGHT:.0%} in {len(over)} of these:",
        "",
        md_table(pd.DataFrame({"Reference month": over["ref"].dt.strftime("%b-%Y"),
                               "Share classes": over["tickers"],
                               "Combined `proxy_capped` weight": over["weight"]})),
        "",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--report", type=Path, default=ROOT / "reports" / "spmo_replication.md")
    args = ap.parse_args()

    rets, weights = run(args.raw)
    args.out.mkdir(parents=True, exist_ok=True)
    rets.rename_axis("month").to_csv(args.out / "spmo_proxy_returns.csv", float_format="%.8f")
    args.report.write_text(report(rets, weights, args.raw), encoding="utf-8")
    print(f"Holding months {rets.index[0]} .. {rets.index[-1]} ({len(rets)}); "
          f"wrote {args.out / 'spmo_proxy_returns.csv'} and {args.report}")


if __name__ == "__main__":
    main()
