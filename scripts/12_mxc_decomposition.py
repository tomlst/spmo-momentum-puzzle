"""
Why does M x C exist? A covariance decomposition of the momentum x market-cap interaction.

For a base portfolio with weights b (summing to 1 over the universe) and a tilt score s >= 0 fixed
at rebalance, the tilted portfolio has weights w_i = b_i z_i with z_i = s_i / sum_j b_j s_j, and

    R_w - R_b = Cov_b(z, r) = rho_b(z, r) * sd_b(z) * sd_b(r)          (moments weighted by b)

M x C compares this covariance under a cap-weighted base (C) and a base without market cap (E):

    M x C_t = rho_C sd_C(z) sd_C(r) - rho_E sd_E(z) sd_E(r)

and an exact Shapley split over the 3! orderings attributes each month's value to three
contributions (IC rho, concentration sd(z), dispersion sd(r)) that add up to it. The arithmetic
is in scripts/mxc_decomposition.py; this script builds its stock-month input with the conventions
of scripts/04_attribution.py and checks the result against that script's portfolios.

Mapping to the 2 x 2 x 2 design. The M = 1 weights are base x max(momentum, 0) within the top
quintile on momentum / sigma, so the tilt score is s = max(momentum, 0) for selected stocks and 0
otherwise. Section 3 defines M x C averaged over V, which is the mean of two slices:

    V = 0   cap base 010, base without cap 000;  tilts 110 and 100
    V = 1   cap base 011, base without cap 001;  tilts 111 and 101

Each slice is decomposed on its own and the two are averaged, so the parts still add up to M x C.
Base and tilt weights drift together from the start of the holding period (t+2) and drop a stock
with a missing return in the same way, so w_i = b_i z_i holds in every holding month once s-bar
is recomputed with the drifted b.

Samples: in-sample holding months Apr-1995 .. Dec-2025 (reference dates Feb-1995 .. Aug-2025);
episodes 2000Q4, 2009Q2, Apr-2020 .. Mar-2021 and 2024Q1; and, as an exploratory extension, the
out-of-sample period Apr-1975 .. Mar-1995 (reference dates Feb-1975 .. Aug-1994). The holdout
was used once by scripts/10_oos.py, so the out-of-sample numbers here are not a test.

Outputs (output/mxc_decomposition/)
  summary.md          validation and tables
  monthly.csv         monthly M x C, its three contributions and the moments of both slices
  fig1_rolling.png    trailing 12-month M x C and its three contributions, in-sample

Usage:
    python scripts/12_mxc_decomposition.py [--raw data/raw] [--out output/mxc_decomposition]
"""
import argparse
import importlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
proxy = importlib.import_module("03_spmo_proxy")
att = importlib.import_module("04_attribution")       # also sets the chart style
reg = importlib.import_module("07_factor_regressions")
oos = importlib.import_module("10_oos")
mxc = importlib.import_module("mxc_decomposition")

ROOT = proxy.ROOT
SLICES = {  # slice -> (cap base, base without cap, cap tilt, tilt without cap)
    "V = 0": ("010", "000", "110", "100"),
    "V = 1": ("011", "001", "111", "101"),
}
PARTS = ["MxC", "contrib_IC", "contrib_concentration", "contrib_dispersion"]
PART_NAMES = {"MxC": "M x C", "contrib_IC": "IC contribution (rho)",
              "contrib_concentration": "Concentration contribution (sd z)",
              "contrib_dispersion": "Dispersion contribution (sd r)"}
SAMPLES = {
    "In-sample": (att.IS_FIRST_REF + proxy.LAG, pd.Period("2025-12", "M")),
    "Out-of-sample (exploratory)": oos.OOS_HOLD,
}
EPISODES = [("2000Q4", "2000-10", "2000-12"), ("2009Q2", "2009-04", "2009-06"),
            ("Apr-2020 to Mar-2021", "2020-04", "2021-03"), ("2024Q1", "2024-01", "2024-03")]
SHORT = {"Apr-2020 to Mar-2021": "2020/21"}                   # chart labels
TOL = 1e-12


# ---------- Panel ----------

def build(raw):
    """Stock-month panels per V slice, and the monthly returns of the eight portfolios computed
    with Section 3's holding rule, for every rebalance from Feb-1975."""
    ret, _, sig = att.signals(raw, first_ref=oos.OOS_REFS[0])
    panels, rets = {k: [] for k in SLICES}, {}
    for t, hold, d in sig:
        w = att.portfolio_weights(d["x"], d["momentum"], d["sigma"], d["mc"])
        top = d["x"].sort_values(ascending=False).index[:int(round(proxy.QUINTILE * len(d)))]
        s = d.loc[top, "momentum"].clip(lower=0).reindex(d.index, fill_value=0.0)
        if not np.allclose(w["100"].reindex(d.index, fill_value=0.0) * s.sum(), s, rtol=0, atol=1e-14):
            raise ValueError(f"{t}: tilt score is not proportional to the weights of portfolio 100")
        r = ret.loc[hold, d.index]
        for code in att.PORTFOLIOS:
            rets.setdefault(code, []).append(
                pd.Series(proxy.hold_returns(ret.loc[hold, w[code].index], w[code]), index=hold))
        for name, (cap, eq, _, _) in SLICES.items():
            bc = proxy.hold_weights(r, w[cap].reindex(d.index))
            be = proxy.hold_weights(r, w[eq].reindex(d.index))
            for i, month in enumerate(hold):
                live = bc[i] > 0
                if not np.array_equal(live, be[i] > 0):
                    raise ValueError(f"{month}: the two bases drop different stocks")
                p = pd.DataFrame({"date": month, "ref": t, "s": s.to_numpy()[live],
                                  "b_cap": bc[i][live], "b_eq": be[i][live], "r": r.iloc[i].to_numpy()[live]})
                if p["r"].isna().any():
                    raise ValueError(f"{month}: missing return for a stock with positive weight")
                panels[name].append(p)
    return {k: pd.concat(v, ignore_index=True) for k, v in panels.items()}, \
        pd.DataFrame({k: pd.concat(v) for k, v in rets.items()})


def components(panels):
    """Monthly moments per slice (identity checked), and M x C with its parts averaged over V."""
    m = {}
    for name, p in panels.items():
        m[name] = mxc.monthly_components(p)
        mxc.check_identity(m[name], TOL)
    avg = sum(m[k][PARTS] for k in SLICES) / len(SLICES)
    return m, avg


# ---------- Statistics ----------

def nw_t(d):
    """Newey-West t-stat (6 lags) of the mean; both implementations must agree."""
    t_ref = mxc.nw_tstat(d, reg.LAGS)
    t_07 = reg.ols_nw(d.to_numpy(), np.empty((len(d), 0)), reg.LAGS)[1][0]
    if abs(t_ref - t_07) > 1e-8:
        raise ValueError(f"Newey-West implementations disagree: {t_ref} vs {t_07}")
    return t_ref


def window(df, first, last):
    return df.loc[(df.index >= pd.Period(first, "M")) & (df.index <= pd.Period(last, "M"))]


def contribution_table(avg, m):
    """Mean (% a year) and NW t of M x C and its parts: averaged over V, and per slice."""
    rows = []
    for c in PARTS:
        row = {"": PART_NAMES[c], "Mean": f"{12 * avg[c].mean():+.2%}", "NW t": f"{nw_t(avg[c]):.2f}"}
        for k in SLICES:
            row[f"{k}: mean (NW t)"] = f"{12 * m[k][c].mean():+.2%} ({nw_t(m[k][c]):.2f})"
        rows.append(row)
    return pd.DataFrame(rows)


def levels_table(m):
    """Average monthly levels of rho, sd(z), sd(r) and effective N under each base."""
    rows = []
    for label, col, fmt in [("rho (IC)", "rho", "{:+.3f}"), ("sd(z) (concentration)", "sd_z", "{:.2f}"),
                            ("sd(r) (dispersion, monthly)", "sd_r", "{:.2%}"),
                            ("Effective N of the tilt", "eff_n", "{:.0f}")]:
        row = {"": label}
        for k, (cap, eq, _, _) in SLICES.items():
            row[f"{k} C ({cap})"] = fmt.format(m[k][f"{col}_C"].mean())
            row[f"{k} E ({eq})"] = fmt.format(m[k][f"{col}_E"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def episode_table(avg, m):
    """Sum over each window of the monthly M x C and its parts (averaged over V)."""
    rows = []
    for label, first, last in EPISODES:
        a = window(avg, first, last)
        row = {"Period": label, "Months": str(len(a))}
        row.update({PART_NAMES[c]: f"{a[c].sum():+.2%}" for c in PARTS})
        rows.append(row)
    return pd.DataFrame(rows)


def ic_sign_table(a, mm):
    """Each part's mean (% a year over all months of the sample) split into the months with a negative and a
    non-negative IC, the IC being averaged over the four bases. The two columns add up to the total."""
    rho = sum(mm[k][f"rho_{b}"] for k in SLICES for b in ("C", "E")) / (2 * len(SLICES))
    neg = rho < 0
    rows = [{"": PART_NAMES[c], "Total": f"{12 * a[c].mean():+.2%}",
             f"Months with IC < 0 ({neg.sum()})": f"{12 * a[c].where(neg, 0.0).mean():+.2%}",
             f"Months with IC >= 0 ({(~neg).sum()})": f"{12 * a[c].where(~neg, 0.0).mean():+.2%}"} for c in PARTS]
    return pd.DataFrame(rows)


def episode_levels(m):
    """Average moments over each window, V = 0 slice (cap base 010, equal base 000)."""
    rows = []
    for label, first, last in EPISODES:
        x = window(m["V = 0"], first, last)
        rows.append({"Period": label, "rho C": f"{x['rho_C'].mean():+.3f}", "rho E": f"{x['rho_E'].mean():+.3f}",
                     "sd(z) C": f"{x['sd_z_C'].mean():.2f}", "sd(z) E": f"{x['sd_z_E'].mean():.2f}",
                     "sd(r) C": f"{x['sd_r_C'].mean():.2%}", "sd(r) E": f"{x['sd_r_E'].mean():.2%}"})
    return pd.DataFrame(rows)


# ---------- Validation ----------

def validate_monthly(frame, months, name):
    """Require one finite, complete row for each expected month."""
    if not frame.index.equals(months):
        missing = list(months.difference(frame.index).astype(str)[:5])
        extra = list(frame.index.difference(months).astype(str)[:5])
        raise ValueError(f"{name}: expected {len(months)} ordered, unique months, "
                         f"found {len(frame)} rows (first missing months: {missing}, first extra months: {extra})")
    finite_rows = np.isfinite(frame.to_numpy(dtype=float, na_value=np.nan)).all(axis=1)
    if not finite_rows.all():
        raise ValueError(f"{name}: non-finite values in {', '.join(months[~finite_rows][:5].astype(str))}")


def validate(m, avg, rets):
    """Check monthly coverage and reproduce the Section 3 portfolio returns and M x C."""
    months = pd.period_range(oos.OOS_HOLD[0], proxy.END.to_period("M"), freq="M")
    is_months = pd.period_range(*SAMPLES["In-sample"], freq="M")
    validate_monthly(rets, months, "Section 3 portfolios")
    validate_monthly(avg, months, "Average decomposition")
    gaps = {}
    for k, (cap, eq, tc, te) in SLICES.items():
        validate_monthly(m[k], months, f"Decomposition ({k})")
        for col, code in [("R_tilt_C", tc), ("R_base_C", cap), ("R_tilt_E", te), ("R_base_E", eq)]:
            gaps[f"{k}: {col} vs {code}"] = (m[k][col] - rets[code]).abs().max()
    formula = {name: f for _, name, f, _ in att.EFFECTS}["M x C"]
    gaps["M x C (average of slices) vs Section 3 formula"] = (avg["MxC"] - att.evaluate(formula, rets)).abs().max()
    if max(gaps.values()) > TOL:
        raise ValueError(f"panel does not reproduce the Section 3 portfolios: {gaps}")

    published = pd.read_csv(ROOT / "output" / "attribution" / "monthly_returns.csv", index_col=0)
    published.index = pd.PeriodIndex(published.index, freq="M")
    validate_monthly(published[list(att.PORTFOLIOS)], is_months, "Published Section 3 portfolios")
    is_m = avg.loc[is_months]
    pub_gap = max((m[k].loc[is_months, col] - published[code]).abs().max()
                  for k, codes in SLICES.items()
                  for col, code in zip(["R_base_C", "R_base_E", "R_tilt_C", "R_tilt_E"], codes))
    if pub_gap > 1e-7:                                       # the CSV is rounded to 8 decimals
        raise ValueError(f"panel differs from output/attribution/monthly_returns.csv by {pub_gap}")
    mean, t = att.mean_t(is_m["MxC"])
    return gaps, pub_gap, {"months": len(is_m), "mean": mean, "t": t, "nw": nw_t(is_m["MxC"])}


# ---------- Figure ----------

def figure(avg, out):
    first, last = SAMPLES["In-sample"]
    roll = window(avg, str(first), str(last))[PARTS].rolling(12).sum().dropna()
    x = roll.index.to_timestamp(how="end")
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for label, a, b in EPISODES:
        ax.axvspan(pd.Period(a, "M").to_timestamp(how="start"), pd.Period(b, "M").to_timestamp(how="end"),
                   color=att.GRID, alpha=0.7, linewidth=0, zorder=0)
        ax.text(pd.Period(a, "M").to_timestamp(how="start"), 1.0, SHORT.get(label, label),
                transform=ax.get_xaxis_transform(), ha="center", va="bottom", color=att.MUTED, fontsize=8.5)
    for c, color in zip(PARTS[1:], att.EFFECT_COLORS):
        ax.plot(x, roll[c], color=color, linewidth=1.4, label=PART_NAMES[c])
    ax.plot(x, roll["MxC"], color=att.INK, linewidth=2.2, label="M x C (sum of the three)")
    ax.axhline(0, color=att.INK_2, linewidth=0.8)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("Trailing 12-month sum")
    ax.set_title("M x C and its three contributions, trailing 12 months", pad=16)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.08), ncol=4, fontsize=9)
    att.footnote(fig, f"{first.strftime('%b-%Y')} to {last.strftime('%b-%Y')}", (len(window(avg, str(first), str(last))) + 5) // 6,
                 "\nM x C and its contributions are averaged over V = 0 and V = 1. Shading marks the four periods in "
                 "the episode table.")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    att.save(fig, out / "fig1_rolling.png")


# ---------- Report ----------

def report(m, avg, gaps, pub_gap, repro):
    t = proxy.md_table
    lines = [
        "# Why does M x C exist? Covariance decomposition",
        "",
        "Generated by `scripts/12_mxc_decomposition.py` with `scripts/mxc_decomposition.py`. For a base b and tilt "
        "score s, R_tilt - R_base = Cov_b(z, r) = rho_b x sd_b(z) x sd_b(r) with z = s / sum(b s). M x C is "
        "Cov under the cap base minus Cov under the base without cap, and an exact Shapley split over the six "
        "orderings gives the IC, concentration and dispersion contributions, which add up to M x C every month. "
        "This is an accounting identity: each part is a contribution from one factor, not a cause.",
        "",
        "Tilt score s = max(momentum, 0) within the top quintile on momentum / sigma, 0 elsewhere. Section 3's "
        "M x C averages over V, so the decomposition is run on two slices and averaged: V = 0 (bases 010 and 000, "
        "tilts 110 and 100) and V = 1 (bases 011 and 001, tilts 111 and 101). sd(r) is the monthly cross-sectional "
        "standard deviation of returns.",
        "",
        "## Validation",
        "",
        "Maximum absolute monthly gap between the panel and the Section 3 portfolios, recomputed from raw data "
        "for every month since Apr-1975:",
        "",
        t(pd.DataFrame({"Check": list(gaps), "Max abs gap": [f"{v:.1e}" for v in gaps.values()]})),
        "",
        f"Against the published `output/attribution/monthly_returns.csv` (in-sample, rounded to 8 decimals): "
        f"max abs gap {pub_gap:.1e}. Identity R_tilt - R_base = Cov and the Shapley sum hold to {TOL:.0e} every month.",
        "",
        f"In-sample M x C ({repro['months']} months): {repro['mean']:+.2%} a year, t = {repro['t']:.2f} "
        f"(independent months), {repro['nw']:.2f} (Newey-West, 6 lags). Section 3 reports +2.8%, 2.84 and 2.39.",
        "",
    ]
    for sample, (first, last) in SAMPLES.items():
        a = window(avg, str(first), str(last))
        mm = {k: window(v, str(first), str(last)) for k, v in m.items()}
        head = f"## {sample}: {first.strftime('%b-%Y')} to {last.strftime('%b-%Y')} ({len(a)} months)"
        lines += [head, ""]
        if sample.startswith("Out"):
            lines += ["**Exploratory.** The holdout was already used once by `scripts/10_oos.py`; these numbers "
                      "are a description, not a test.", ""]
        lines += ["Mean contribution (% a year) and Newey-West t (6 lags).", "", t(contribution_table(a, mm)), "",
                  "Average monthly levels of the moments under each base (C = with market cap, E = without).", "",
                  t(levels_table(mm)), "",
                  "Mean contribution (% a year) split by the sign of the IC (averaged over the four bases). Each "
                  "column divides by all months of the sample, so the two columns add up to the total.", "",
                  t(ic_sign_table(a, mm)), ""]
    lines += ["## Episodes (in-sample)", "",
              "Sum of the monthly values over each window, M x C and its parts averaged over V. Too few months "
              "for t-stats.", "", t(episode_table(avg, m)), "",
              "Average moments over each window, V = 0 slice (C = 010, E = 000).", "", t(episode_levels(m)), "",
              "![M x C and its three contributions](fig1_rolling.png)", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--out", type=Path, default=ROOT / "output" / "mxc_decomposition")
    args = ap.parse_args()

    panels, rets = build(args.raw)
    m, avg = components(panels)
    gaps, pub_gap, repro = validate(m, avg, rets)

    args.out.mkdir(parents=True, exist_ok=True)
    slices = pd.concat({k.replace(" ", ""): v for k, v in m.items()}, axis=1)
    monthly = avg.join(slices.set_axis([f"{a}_{b}" for a, b in slices.columns], axis=1))
    monthly.rename_axis("month").to_csv(args.out / "monthly.csv", float_format="%.8f")
    figure(avg, args.out)
    (args.out / "summary.md").write_text(report(m, avg, gaps, pub_gap, repro), encoding="utf-8")
    print(f"M x C in-sample {repro['mean']:+.2%} (t {repro['t']:.2f}, NW {repro['nw']:.2f}); wrote {args.out}")


if __name__ == "__main__":
    main()
