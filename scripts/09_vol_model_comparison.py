"""
Volatility targeting of the SPMO proxy: four forecasts of next month's volatility compared.

Exposure for month m+1 = TARGET / forecast of the proxy's realised volatility in m+1, set at the end
of month m, no cap, the rest at the T-bill rate (scripts/08_har_vol_target.py). Only the forecast
differs:

  HAR        script 08: log HAR on the proxy's realised volatility over the last 1, 3 and 12 months
  Ridge      ridge regression of log RV(m+1) on ten features known at the end of m:
               log proxy RV over the last 1, 3, 6 and 12 months; log ex-ante volatility of the
               weights held in m+1 applied to the constituents' daily returns over the last 63 and
               126 trading days; log S&P 500 RV over the last 1 and 3 months; S&P 500 return over
               the last 1 and 12 months
  GARCH      GARCH(1,1) with Gaussian errors and a constant mean on the proxy's daily returns; the
             forecast is the average variance over the next H trading days
  EWMA       RiskMetrics exponential smoothing of squared daily returns, decay EWMA_DECAY; the
             forecast is the current estimate (flat term structure)

All realised volatilities are annualised root mean squares of daily returns, sqrt(252 x mean r^2).
The weights for m+1 are those known at the end of m: the new rebalance weights in the first month of
a holding period, otherwise the previous month's weights after drift.

Estimation uses only data available at each month end:
  - Ridge, re-estimated monthly on an expanding window of monthly pairs: features standardised on the
    training window, objective ||y - a - Zb||^2 + lambda ||b||^2 with lambda chosen from LAMBDAS by
    rolling-origin validation over the last VALIDATION pairs; forecast exp(fitted + s^2 / 2);
  - GARCH, re-estimated monthly by maximum likelihood on all daily returns up to the month end;
  - EWMA has no estimated parameters.
The models are compared over the months all four cover. In-sample only. No transaction or financing
costs.

Outputs (output/vol_model_comparison/)
  summary.md, monthly.csv, fig1_forecast.png, fig2_exposure.png, fig3_performance.png,
  fig4_ridge_coefficients.png, fig5_garch_parameters.png

Usage:
    python scripts/09_vol_model_comparison.py [--raw data/raw] [--out output/vol_model_comparison]
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
from scipy.optimize import minimize
from scipy.signal import lfilter

proxy = importlib.import_module("03_spmo_proxy")
attribution = importlib.import_module("04_attribution")
regressions = importlib.import_module("07_factor_regressions")
har = importlib.import_module("08_har_vol_target")

ROOT = proxy.ROOT
TARGET = har.TARGET
H = 21                                        # trading days in a month
MIN_FIT = 24                                  # smallest window a ridge validation fit uses (months)
VALIDATION = 24                               # ridge validation pairs at the end of each training window
LAMBDAS = np.logspace(-3, 3, 25)              # candidate ridge penalties on standardised features
GARCH_MIN_DAYS = 504                          # daily returns before the first GARCH estimate
EWMA_DECAY = 0.97
FEATURES = ["log proxy RV1", "log proxy RV3", "log proxy RV6", "log proxy RV12",
            "log ex-ante vol 63d", "log ex-ante vol 126d", "log market RV1", "log market RV3",
            "market return 1m", "market return 12m"]
UNMANAGED = "SPMO proxy"
MODELS = ["HAR", "Ridge", "GARCH(1,1)", f"EWMA ({EWMA_DECAY})"]
COLORS = {UNMANAGED: "#2a78d6", "HAR": "#1baf7a", "Ridge": "#eb6834", "GARCH(1,1)": "#4a3aa7",
          f"EWMA ({EWMA_DECAY})": "#eda100"}


# ---------- Ridge features ----------

def rms_vol(r):
    return np.sqrt(252 * np.mean(np.square(r)))


def trailing_vol(daily, months):
    """Annualised RMS volatility of a daily series over the trailing `months` calendar months."""
    month = daily.index.to_period("M")
    ss, n = daily.pow(2).groupby(month).sum(), daily.groupby(month).size()
    return np.sqrt(252 * ss.rolling(months).sum() / n.rolling(months).sum())


def features(proxy_daily, known, panel, market):
    """Feature matrix indexed by month end m (features for forecasting m+1) and the target log RV(m+1)."""
    months = proxy_daily.index.to_period("M").unique()
    X = pd.DataFrame(index=months)
    for h in (1, 3, 6, 12):
        X[f"log proxy RV{h}"] = np.log(trailing_vol(proxy_daily, h))
    mkt = market.loc[market.index.to_period("M") >= months[0] - 12]
    X["log market RV1"] = np.log(trailing_vol(mkt, 1)).reindex(months)
    X["log market RV3"] = np.log(trailing_vol(mkt, 3)).reindex(months)
    growth = (1 + mkt).groupby(mkt.index.to_period("M")).prod()
    X["market return 1m"] = (growth - 1).reindex(months)
    X["market return 12m"] = (growth.rolling(12).apply(np.prod, raw=True) - 1).reindex(months)

    days = panel.index
    month_end = pd.Series(days, index=days).groupby(days.to_period("M")).last()
    for m in months:
        nxt = m + 1
        if nxt not in known:
            continue
        w = known[nxt]
        cols = panel.columns.get_indexer(w.index)
        end = days.get_loc(month_end[m])
        for n in (63, 126):
            R = np.zeros((n, len(w)))
            R[:, cols >= 0] = np.nan_to_num(panel.iloc[end - n + 1:end + 1].to_numpy()[:, cols[cols >= 0]])
            X.loc[m, f"log ex-ante vol {n}d"] = np.log(rms_vol(R @ w.to_numpy()))
    y = np.log(trailing_vol(proxy_daily, 1)).shift(-1).rename("log RV next")
    return X[FEATURES], y


# ---------- Ridge ----------

def ridge_fit(X, y, lam):
    """Ridge on standardised features with an unpenalised intercept; returns a predictor and the
    standardised coefficients and residual variance."""
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    yc = y - y.mean()
    n, k = Z.shape
    b = np.linalg.solve(Z.T @ Z + lam * np.eye(k), Z.T @ yc)
    resid = yc - Z @ b
    s2 = resid @ resid / max(n - k - 1, 1)
    return (lambda Xn: y.mean() + ((Xn - mu) / sd) @ b), b, s2


def choose_lambda(X, y):
    """Rolling-origin validation over the last VALIDATION pairs of the training window."""
    n = len(y)
    errors = []
    for lam in LAMBDAS:
        e = [ridge_fit(X[:j], y[:j], lam)[0](X[j:j + 1])[0] - y[j] for j in range(n - VALIDATION, n)]
        errors.append(np.mean(np.square(e)))
    return LAMBDAS[int(np.argmin(errors))]


def ridge_forecasts(X, y):
    """Expanding-window ridge forecasts of next month's volatility, made at each month end."""
    valid = X.notna().all(axis=1)
    months = X.index[valid]
    forecasts, coefs = {}, {}
    for i, m in enumerate(months):
        train = months[:i]
        train = train[y.loc[train].notna()]
        if len(train) < MIN_FIT + VALIDATION:
            continue
        Xt, yt = X.loc[train].to_numpy(), y.loc[train].to_numpy()
        lam = choose_lambda(Xt, yt)
        predict, b, s2 = ridge_fit(Xt, yt, lam)
        forecasts[m + 1] = float(np.exp(predict(X.loc[[m]].to_numpy())[0] + s2 / 2))
        coefs[m + 1] = {"lambda": lam, **dict(zip(FEATURES, b))}
    return pd.Series(forecasts), pd.DataFrame(coefs).T


# ---------- GARCH and EWMA ----------

def garch_variance(eps, omega, alpha, beta, var0):
    """Conditional variances sigma^2_t = omega + alpha eps_{t-1}^2 + beta sigma^2_{t-1}, t = 0..n
    (the last one is the one-step-ahead forecast), starting from var0."""
    x = omega + alpha * eps ** 2
    return np.concatenate([[var0], lfilter([1.0], [1.0, -beta], x, zi=[beta * var0])[0]])


def garch_fit(r, start=None):
    """Gaussian MLE of a constant-mean GARCH(1,1); returns (mu, omega, alpha, beta) and the variance path."""
    mu = r.mean()
    eps = r - mu
    var0 = eps.var()

    def nll(p):
        omega, alpha, beta = p
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.9999:
            return 1e10
        s2 = garch_variance(eps, omega, alpha, beta, var0)[:-1]
        return 0.5 * np.sum(np.log(s2) + eps ** 2 / s2)

    x0 = start if start is not None else np.array([var0 * 0.05, 0.08, 0.90])
    res = minimize(nll, x0, method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-6, "maxiter": 4000})
    omega, alpha, beta = res.x
    return (mu, omega, alpha, beta), garch_variance(eps, omega, alpha, beta, var0)


def garch_forecasts(daily, month_ends):
    """Monthly re-estimated GARCH(1,1): average variance over the next H days, annualised as a volatility."""
    r = daily.to_numpy()
    out, params, start = {}, {}, None
    for t in month_ends:
        i = daily.index.get_loc(t)
        if i + 1 < GARCH_MIN_DAYS:
            continue
        (mu, omega, alpha, beta), s2 = garch_fit(r[:i + 1], start)
        start = np.array([omega, alpha, beta])
        p = alpha + beta
        long_run = omega / (1 - p)
        path = long_run + p ** np.arange(H) * (s2[-1] - long_run)       # sigma^2_{t+1} .. sigma^2_{t+H}
        out[t.to_period("M") + 1] = float(np.sqrt(252 * (path.mean() + mu ** 2)))
        params[t.to_period("M") + 1] = {"alpha": alpha, "beta": beta, "persistence": p,
                                        "long-run vol": np.sqrt(252 * long_run)}
    return pd.Series(out), pd.DataFrame(params).T


def ewma_forecasts(daily, month_ends):
    """RiskMetrics EWMA of squared daily returns at each month end, annualised as a volatility."""
    var = daily.pow(2).ewm(alpha=1 - EWMA_DECAY, adjust=False).mean()
    return pd.Series({t.to_period("M") + 1: float(np.sqrt(252 * var.loc[t])) for t in month_ends})


# ---------- Evaluation ----------

def accuracy(actual_vol, forecasts):
    """Out-of-sample R^2 of log forecasts, mean log error and RMSE in volatility units."""
    a = np.log(actual_vol)
    sst = ((a - a.mean()) ** 2).sum()
    rows = {}
    for name, f in forecasts.items():
        e = a - np.log(f)
        rows[name] = {"R2 (log vol)": 1 - (e ** 2).sum() / sst, "Mean error (log)": e.mean(),
                      "RMSE (vol)": np.sqrt(((actual_vol - f) ** 2).mean())}
    return pd.DataFrame(rows).T


def loss_test(actual_vol, f1, f2):
    """Mean difference of squared log errors (f1 - f2) with a Newey-West t-stat (Diebold-Mariano)."""
    a = np.log(actual_vol)
    d = ((a - np.log(f1)) ** 2 - (a - np.log(f2)) ** 2).to_numpy()
    b, t, _ = regressions.ols_nw(d, np.empty((len(d), 0)))
    return b[0], t[0]


def perf_rows(df, rf):
    rows, ex = [], {}
    for name in [UNMANAGED, *MODELS]:
        x = df[name]
        e = x - rf.loc[df.index]
        ex[name] = e.to_numpy()
        g = (1 + x).cumprod()
        w = pd.Series(1.0, index=df.index) if name == UNMANAGED else df[f"w {name}"]
        rows.append({"Portfolio": name if name == UNMANAGED else f"{name} vol target ({TARGET:.0%})",
                     "CAGR": f"{(1 + x).prod() ** (12 / len(x)) - 1:.1%}", "Volatility": f"{x.std() * 12 ** 0.5:.1%}",
                     "Sharpe": f"{e.mean() / e.std() * 12 ** 0.5:.2f}", "Max drawdown": f"{(g / g.cummax() - 1).min():.1%}",
                     "Avg exposure": f"{w.mean():.2f}", "Exposure range": f"{w.min():.2f}-{w.max():.2f}",
                     "Sharpe z vs unmanaged": "" if name == UNMANAGED else f"{har.jkm(ex[name], ex[UNMANAGED]):+.2f}",
                     "Sharpe z vs HAR": "" if name in (UNMANAGED, "HAR") else f"{har.jkm(ex[name], ex['HAR']):+.2f}"})
    return proxy.md_table(pd.DataFrame(rows))


# ---------- Figures ----------

def note(fig, period):
    fig.text(0.01, 0.005, f"In-sample: {period}. Forecasts made at each month end from data up to that date. "
             "Monthly total returns, no transaction or financing costs. Data: CRSP via WRDS.",
             ha="left", va="bottom", color=attribution.MUTED, fontsize=8.5)


def figures(df, coefs, gparams, out, period):
    x = df.index.to_timestamp(how="start")
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.bar(x, df["RV"], width=25, align="edge", color="#b9b8b3", label="Realised volatility of the month")
    for k in MODELS:
        ax.step(x, df[f"forecast {k}"], where="post", color=COLORS[k], linewidth=1.1, label=f"{k} forecast")
    ax.axhline(TARGET, color=attribution.INK, linewidth=0.9, linestyle="--", label=f"Target {TARGET:.0%}")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("Annualised volatility")
    ax.set_title("SPMO proxy: volatility forecasts vs realised volatility by month")
    ax.legend(loc="upper left", bbox_to_anchor=(0.22, 1))
    note(fig, period)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    attribution.save(fig, out / "fig1_forecast.png")

    fig, axes = plt.subplots(len(MODELS), 1, figsize=(15, 3 * len(MODELS)), sharex=True, sharey=True)
    for ax, k in zip(axes, MODELS):
        w = df[f"w {k}"]
        ax.step(x, w, where="post", color=COLORS[k], linewidth=1.2)
        ax.axhline(1, color=attribution.INK_2, linewidth=0.8)
        ax.set_title(f"{k}: average {w.mean():.2f}, range {w.min():.2f}-{w.max():.2f}")
        ax.set_ylim(0, max(df[f"w {m}"].max() for m in MODELS) * 1.05)
    axes[len(MODELS) // 2].set_ylabel("Exposure (1 = fully invested)")
    fig.suptitle(f"Exposure to the SPMO proxy: {TARGET:.0%} / forecast, no cap", x=0.01, ha="left")
    note(fig, period)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    attribution.save(fig, out / "fig2_exposure.png")

    dates = [df.index[0].to_timestamp(how="start")] + list(df.index.to_timestamp(how="end"))
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [1.5, 1]})
    for k, c in COLORS.items():
        g = np.concatenate([[1.0], (1 + df[k]).cumprod().to_numpy()])
        axes[0].plot(dates, g, color=c, linewidth=2 if k == "HAR" else 1.5, label=f"{k}  (${g[-1]:.0f})")
        axes[1].plot(dates, g / np.maximum.accumulate(g) - 1, color=c, linewidth=1.1)
    axes[0].set_yscale("log")
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}"))
    axes[0].set_title("(a) Growth of $1 (log scale; final value in brackets)")
    axes[0].legend(loc="upper left")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axes[1].set_title("(b) Drawdown")
    fig.suptitle(f"SPMO proxy: unmanaged and volatility-targeted ({TARGET:.0%}) with four forecasts", x=0.01, ha="left")
    note(fig, period)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    attribution.save(fig, out / "fig3_performance.png")

    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    cx = coefs.index.to_timestamp(how="start")
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948",
               "#898781", "#0d366b"]
    for f, c in zip(FEATURES, palette):
        axes[0].plot(cx, coefs[f], color=c, linewidth=1.3, label=f)
    axes[0].axhline(0, color=attribution.INK_2, linewidth=0.8)
    axes[0].set_ylabel("Coefficient on the standardised feature")
    axes[0].set_title("(a) Ridge coefficients, re-estimated each month on the expanding window")
    axes[0].legend(loc="upper left", ncol=2, fontsize=8.5, bbox_to_anchor=(1.0, 1.0))
    axes[1].step(cx, coefs["lambda"], where="post", color=attribution.INK, linewidth=1.2)
    axes[1].set_yscale("log")
    axes[1].set_ylim(LAMBDAS[0] / 2, LAMBDAS[-1] * 2)
    axes[1].set_ylabel("lambda")
    axes[1].set_title(f"(b) Penalty chosen by rolling-origin validation from {len(LAMBDAS)} values, "
                      f"{LAMBDAS[0]:g} to {LAMBDAS[-1]:g}")
    note(fig, period)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    attribution.save(fig, out / "fig4_ridge_coefficients.png")

    fig, axes = plt.subplots(2, 1, figsize=(14, 7.5), sharex=True)
    gx = gparams.index.to_timestamp(how="start")
    for k, c in [("alpha", "#eb6834"), ("beta", "#2a78d6"), ("persistence", attribution.INK)]:
        axes[0].plot(gx, gparams[k], color=c, linewidth=1.4, label=k if k != "persistence" else "alpha + beta")
    axes[0].set_title("(a) GARCH(1,1) parameters, re-estimated each month end on all daily returns so far")
    axes[0].legend(loc="center right")
    axes[1].plot(gx, gparams["long-run vol"], color="#4a3aa7", linewidth=1.4)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axes[1].set_title("(b) Implied long-run volatility, sqrt(252 x omega / (1 - alpha - beta))")
    note(fig, period)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    attribution.save(fig, out / "fig5_garch_parameters.png")


# ---------- Report ----------

def summary(df, coefs, gparams, acc, tests, rf, period):
    last, gl = coefs.iloc[-1], gparams.iloc[-1]
    lines = [
        "# Volatility targeting of the SPMO proxy: four forecasting models",
        "",
        f"Generated by `scripts/09_vol_model_comparison.py`. In-sample only: {period} ({len(df)} months), the "
        "months covered by all four forecasts. No transaction or financing costs.",
        "",
        "## Method",
        "",
        f"- Exposure for month m+1 = {TARGET:.0%} / forecast of the proxy's realised volatility in m+1, set at the end "
        "of m, no cap; the rest (or the borrowing) at the T-bill rate. Same rule for every model.",
        "- **HAR**: `scripts/08_har_vol_target.py`, log HAR on 1, 3 and 12-month realised volatility.",
        "- **Ridge**: log RV(m+1) on ten features known at the end of m (log proxy RV over 1, 3, 6 and 12 months; "
        "log ex-ante volatility of the next month's weights over the constituents' last 63 and 126 daily returns; "
        "log S&P 500 RV over 1 and 3 months; S&P 500 return over 1 and 12 months). Features standardised on the "
        f"training window; lambda from {len(LAMBDAS)} log-spaced values ({LAMBDAS[0]:g} to {LAMBDAS[-1]:g}) by "
        f"rolling-origin validation over the last {VALIDATION} months; forecast exp(fitted + s^2/2).",
        f"- **GARCH(1,1)**: Gaussian, constant mean, on the proxy's daily returns; re-estimated each month end by "
        f"maximum likelihood on all daily returns so far (at least {GARCH_MIN_DAYS}). Forecast = average conditional "
        f"variance over the next {H} trading days, annualised.",
        f"- **EWMA ({EWMA_DECAY})**: sigma^2_t = {EWMA_DECAY} sigma^2_(t-1) + {1 - EWMA_DECAY:.2f} r^2_(t-1) on daily "
        "returns (RiskMetrics, zero mean), taken at the month end and annualised; no parameters estimated.",
        "",
        "## Forecast accuracy",
        "",
        "Against the realised volatility of the next calendar month. R^2 of log forecasts; mean log error (positive "
        "= realised above forecast); RMSE in volatility units.",
        "",
        proxy.md_table(acc.reset_index().rename(columns={"index": "Forecast"}), floatfmt="{:.3f}"),
        "",
        "Diebold-Mariano comparison with HAR on squared log errors (negative = model more accurate than HAR; "
        "Newey-West t, 6 lags):",
        "",
        proxy.md_table(pd.DataFrame([{"Comparison": k, "Mean loss difference": f"{v[0]:+.4f}", "t-stat": f"{v[1]:.2f}"}
                                     for k, v in tests.items()])),
        "",
        f"Final ridge coefficients ({coefs.index[-1].strftime('%b-%Y')}, lambda {last['lambda']:.3g}): "
        + ", ".join(f"{f} {last[f]:+.3f}" for f in FEATURES) + ".",
        "",
        f"Final GARCH(1,1) estimates: alpha {gl['alpha']:.3f}, beta {gl['beta']:.3f}, persistence "
        f"{gl['persistence']:.3f}, long-run volatility {gl['long-run vol']:.1%}.",
        "",
        "## Performance",
        "",
        "Sharpe uses the T-bill rate. z is the Jobson-Korkie test with Memmel's correction (|z| > 1.96 is "
        "significant at 5%).",
        "",
        "### Common period",
        "",
        perf_rows(df, rf),
        "",
    ]
    for label, a, b in har.SUBPERIODS:
        part = df[(df.index.year >= a) & (df.index.year <= b)]
        span = f"{part.index[0].strftime('%b-%Y')} to {part.index[-1].strftime('%b-%Y')}"
        lines += [f"### {span}", "", perf_rows(part, rf), ""]
    lines += ["## Figures", ""]
    for f, c in [("fig1_forecast.png", "Forecasts"), ("fig2_exposure.png", "Exposure"),
                 ("fig3_performance.png", "Performance"), ("fig4_ridge_coefficients.png", "Ridge coefficients"),
                 ("fig5_garch_parameters.png", "GARCH parameters")]:
        lines += [f"![{c}]({f})", ""]
    lines += ["## References", "",
              "- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of "
              "Econometrics* 31(3), 307-327.",
              "- Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of "
              "Financial Econometrics* 7(2), 174-196.",
              "- J.P. Morgan/Reuters (1996). *RiskMetrics Technical Document*, 4th edition.", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--out", type=Path, default=ROOT / "output" / "vol_model_comparison")
    args = ap.parse_args()

    monthly, daily, known, panel = har.proxy_panel(args.raw)
    idx = pd.read_parquet(args.raw / "sp500_index_daily.parquet", columns=["date", "vwretd"])
    market = idx.set_index("date")["vwretd"].astype(float)
    month_ends = list(pd.Series(daily.index, index=daily.index).groupby(daily.index.to_period("M")).last())

    rv = har.realised_vol(daily)
    fc = {"HAR": har.har_forecasts(rv)[0]}
    X, y = features(daily, known, panel, market)
    fc["Ridge"], coefs = ridge_forecasts(X, y)
    fc["GARCH(1,1)"], gparams = garch_forecasts(daily, month_ends)
    fc[f"EWMA ({EWMA_DECAY})"] = ewma_forecasts(daily, month_ends)

    months = monthly.index
    for f in fc.values():
        months = months.intersection(f.index)
    assert months[0] >= attribution.IS_FIRST_REF + proxy.LAG, "out-of-sample months in the results"
    ff = pd.read_parquet(args.raw / "ff5_factors_monthly.parquet", columns=["date", "rf"])
    rf = ff.set_index(ff["date"].dt.to_period("M"))["rf"].astype(float)

    df = pd.DataFrame(index=months)
    df["RV"] = rv["RV1"].loc[months]
    df[UNMANAGED] = monthly.loc[months]
    for k in MODELS:
        df[f"forecast {k}"] = fc[k].loc[months]
        df[f"w {k}"] = TARGET / df[f"forecast {k}"]
        df[k] = df[f"w {k}"] * df[UNMANAGED] + (1 - df[f"w {k}"]) * rf.loc[months]
    period = f"{months[0].strftime('%b-%Y')} to {months[-1].strftime('%b-%Y')}"

    acc = accuracy(df["RV"], {**{k: df[f"forecast {k}"] for k in MODELS},
                              "Last month (RV1)": rv["RV1"].shift(1).loc[months]})
    tests = {f"{k} vs HAR": loss_test(df["RV"], df[f"forecast {k}"], df["forecast HAR"]) for k in MODELS[1:]}

    args.out.mkdir(parents=True, exist_ok=True)
    df.rename_axis("month").to_csv(args.out / "monthly.csv", float_format="%.6f")
    figures(df, coefs.loc[months], gparams.loc[months], args.out, period)
    (args.out / "summary.md").write_text(summary(df, coefs.loc[months], gparams.loc[months], acc, tests, rf, period),
                                         encoding="utf-8")
    print(f"Months {months[0]} .. {months[-1]} ({len(months)}); wrote {args.out}")


if __name__ == "__main__":
    main()
