# Introduction

## Motivation

The Invesco S&P 500 Momentum ETF (SPMO) has been one of the standout equity ETFs of the past two
years. Measured with CRSP total returns, it returned 45.8% in 2024 and 26.6% in 2025, against 24.9%
and 17.7% for the S&P 500 (SPY). Over its full history (Nov-2015 to Dec-2025) it compounded at 17.5%
a year versus 14.3% for SPY. The record is not one-sided: SPMO lagged SPY in 2016, 2019, 2021 and
2023.

SPMO is often described simply as "momentum", but the index behind it combines three ingredients:

1. **Momentum.** Stocks are ranked on their 12-month return, skipping the most recent month.
2. **Volatility.** The ranking signal is momentum divided by return volatility, so a calm winner
   ranks above a volatile one with the same return.
3. **Market capitalisation.** The selected stocks are weighted by market cap times a momentum score,
   so the largest winners dominate the portfolio.

Building a replication of SPMO showed that the weighting scheme has a large effect on performance.
That raises the question of what SPMO's outperformance actually rewards.

## Research questions

1. **Attribution.** Does SPMO's outperformance come from momentum, from the volatility adjustment, or
   from market-cap weighting?
2. **Improvement.** Given the answer, can the construction be improved?

The analysis has three steps. It first replicates SPMO's index with CRSP data, extending it back to
1975. It then attributes performance to the three ingredients. Finally, it tests alternative
constructions.

## How the S&P 500 Momentum Index is built

The summary below follows the *S&P Momentum Indices Methodology* (S&P Dow Jones Indices, August 2026).
Page numbers refer to that document.

| Step | Rule | Page |
|---|---|---|
| Universe | S&P 500 constituents on the rebalancing reference date | 5 |
| Eligibility | at least 150 trading days in the 12-month measurement period | 6 |
| Momentum | 12-month price change ending one month before the reference date. If 12 months of history are unavailable, 9 months are used. At least 10 months of trading are required | 18 |
| Risk adjustment | momentum ÷ standard deviation of daily price returns over the same period | 18 |
| Score | z-score of risk-adjusted momentum within the universe, winsorised at ±3. Momentum score = 1 + z if z > 0, otherwise 1 / (1 − z) | 20 |
| Selection | top quintile by score (count rounded to the nearest integer), with a 20% buffer for current constituents | 7 |
| Weighting | float-adjusted market cap × momentum score, capped at the lower of 9% and 3× the stock's market-cap weight | 8 |
| Rebalancing | semi-annual. Reference dates are the last business day of February and August; changes take effect after the close on the third Friday of March and September | 10 |

## Replicating SPMO with CRSP

### Construction of the proxy

The proxy follows the index rules, with simplifications where CRSP data or monthly frequency
requires them.

| Step | Proxy rule |
|---|---|
| Reference dates | End of February and end of August of each year (*t*) |
| Universe | S&P 500 constituents at the end of month *t* that have a return in month *t*, which excludes stocks that have already delisted |
| Momentum | Compounded 12-month total return (dividends included) over months *t*−12 to *t*−1, skipping the reference month. All 12 months are required. Example: for the Feb-2014 reference date, the window is Feb-2013 to Jan-2014 |
| Risk | Standard deviation of daily total returns over the same 12 months, annualised by √252. At least 150 trading days are required |
| Signal | *x* = momentum ÷ risk |
| Selection | Top 20% by *x*, with the count rounded (about 100 stocks). No buffer |
| Weighting | *w* ∝ market cap × max(*x*, 0), using CRSP total market cap and the raw *x* (no z-score transform) |
| Cap (`proxy_capped` only) | Each stock is capped at the lower of 9% and 3× its market-cap weight among the selected stocks. Excess weight is redistributed pro rata, iterating until every stock satisfies the cap. If every stock with a positive score is capped (only Feb-2009, when 19 of 99 selected stocks had positive momentum), the remainder goes to the other selected stocks in proportion to market cap. The cap applies to each PERMNO. The methodology says "weight in the index"; we read this as the weight within the selected constituents |
| Holding | Six months starting in month *t*+2 (April–September and October–March). Positions are bought and held, so weights drift with prices. A delisted stock's weight is redistributed pro rata to the remaining holdings |
| Returns | CRSP monthly total returns, including dividends and delisting returns, before transaction costs and fees |

Reference dates run from Feb-1975 to Aug-2025, and holding months from Apr-1975 to Dec-2025.

### Fit to SPMO

The comparison covers Nov-2015 to Dec-2025. Return is the compound annual growth rate; volatility and
tracking error are annualised from monthly returns. The proxy is built by
[scripts/03_spmo_proxy.py](scripts/03_spmo_proxy.py); calendar-year returns and portfolio
characteristics are in [reports/spmo_replication.md](reports/spmo_replication.md).

| | Correlation with SPMO | Tracking error | Annual return | Annual volatility |
|---|---|---|---|---|
| `proxy` (uncapped) | 0.979 | 3.5% | 18.4% | 17.0% |
| `proxy_capped` | 0.992 | 2.0% | 17.0% | 15.8% |
| SPMO (actual) | — | — | 17.5% | 15.9% |

Adding the cap raises the correlation from 0.979 to 0.992 and lowers the tracking error from 3.5% to
2.0%, because SPMO itself applies it (in 2024 the uncapped proxy returned 62.1%, the capped one 48.3%
and SPMO 45.8%). Over the whole in-sample period the cap barely changes performance (12.7% vs 12.4% a
year, Sharpe 0.63 for both); it matters for replication accuracy rather than for long-run returns.
The capped proxy returns 0.5% a year less than SPMO, even though SPMO's returns are after fees.

### Remaining differences from the index

| # | Item | Proxy | S&P 500 Momentum Index / SPMO | Page | Impact |
|---|---|---|---|---|---|
| 1 | Return type | total return for momentum and σ | price return for both | 18 | Small: reorders only a few stocks |
| 2 | Short history | excluded if under 12 months | falls back to 9 months; requires 10 months of trading | 18 | Very small: affects only recent listings |
| 3 | Score in weights | raw *x* | winsorised z-score mapped to 1 + z or 1/(1 − z); about 1.4–4 for selected stocks | 20 | Large when uncapped, because raw *x* concentrates weights. Small once capped |
| 4 | Buffer | none | 20% buffer: top 80% always selected; current constituents within 120% retained first; then new names ranked 80–100% | 7 | Small: mainly lowers turnover |
| 5 | Cap | none in `proxy`; same rule in `proxy_capped` | lower of 9% and 3× market-cap weight | 8 | Large: sets concentration and volatility |
| 6 | Market cap | CRSP total market cap | float-adjusted market cap | 8 | Small: CRSP has no float data |
| 7 | Effective date | holdings start at the beginning of April and October | after the close on the third Friday of March and September | 10 | Small: monthly data can only approximate this, so about a third of March and September is misaligned |
| 8 | Costs | before fees and trading costs | ETF expense ratio of 0.13% a year, plus trading costs and tracking error | Invesco | About 0.1–0.2% a year |
| 9 | Universe and selection size | S&P 500 at the reference date, top quintile | same | 5, 7 | None |
| 10 | Cap unit | each PERMNO (share class) | each company, e.g. GOOGL and GOOG combined | 8 | Small: two share classes of one company exceed 9% combined at 3 of 21 rebalances since Aug-2015, all Alphabet (largest 18.0%) |

## Sample design

| Period | Reference dates | Holding months | Role |
|---|---|---|---|
| In-sample | Feb-1995 to Aug-2025 | Apr-1995 to Dec-2025 | Attribution, model design and selection |
| Out-of-sample | Feb-1975 to Aug-1994 | Apr-1975 to Mar-1995 | Evaluated once, after all design choices are fixed |

Each six-month holding period is assigned to a sample by its reference date, so no portfolio spans
both samples.

The out-of-sample period is a held-out *earlier* period rather than a later one. SPMO's live history
(2015–2025) falls inside the in-sample period. The 1975–1994 period comes from the same
survivorship-free S&P 500 universe but covers a different market regime, so it tests whether
conclusions drawn in the recent, mega-cap-dominated era generalise. The momentum effect itself was
documented on data that overlaps this period (Jegadeesh and Titman, 1993). The out-of-sample test
therefore evaluates this project's construction choices, not the existence of momentum.

## Data

All data comes from WRDS: CRSP (CIZ v2) for constituents, prices and returns; Compustat for
fundamentals and GICS sectors; and the Fama-French factor library. See the [README](README.md) for
the sources and [reports/data_validation.md](reports/data_validation.md) for the data-quality checks.
