# S&P 500 Momentum Research

This project asks where the outperformance of the Invesco S&P 500 Momentum ETF (SPMO) comes from:
momentum, the volatility adjustment, or market-cap weighting. It then tests whether the
construction can be improved. The analysis replicates SPMO's index on a point-in-time,
survivorship-free S&P 500 universe from 1975 to 2025. In-sample is 1995–2025 and out-of-sample is
1975–1994. See [introduction.md](introduction.md).

**Status:** data layer complete and validated; SPMO replication validated (correlation 0.992 with
SPMO); in-sample attribution done ([output/attribution](output/attribution/summary.md)); improvement
in progress.

## Repository layout

```
├── introduction.md            motivation, research questions, SPMO replication, sample design
├── data/
│   ├── raw/                   WRDS extract (not versioned; MANIFEST.json is)
│   └── processed/             derived series, e.g. SPMO proxy returns (not versioned)
├── output/
│   ├── attribution/           in-sample M x C x V attribution: summary.md, figures, tables
│   ├── double_sort/           momentum x size double sort within the S&P 500
│   ├── episodes/              what drove the M x C interaction in its four largest episodes
│   ├── factor_regressions/    FF5 + UMD + reversal-factor regressions of the market-cap effects
│   ├── har_vol_target/        HAR volatility targeting of the SPMO proxy
│   ├── oos/                   out-of-sample evaluation (1975-1994), run once
│   ├── post_oos/              supplementary analyses run after the out-of-sample evaluation
│   └── vol_model_comparison/  HAR vs ridge, GARCH(1,1) and EWMA forecasts for volatility targeting
├── reports/
│   ├── data_validation.md     generated data-quality report
│   └── spmo_replication.md    generated fit of the SPMO proxy to the ETF
├── scripts/
│   ├── 01_fetch_wrds.py       download all raw data from WRDS
│   ├── 02_validate_data.py    data-quality checks -> reports/data_validation.md
│   ├── 03_spmo_proxy.py       SPMO proxy backtest -> reports/spmo_replication.md
│   ├── 04_attribution.py      2 x 2 x 2 attribution (in-sample) -> output/attribution/
│   ├── 05_episodes.py         stock and size-group contributions to M x C episodes -> output/episodes/
│   ├── 06_double_sort.py      momentum x size double sort (in-sample) -> output/double_sort/
│   ├── 07_factor_regressions.py  factor regressions (in-sample) -> output/factor_regressions/
│   ├── 08_har_vol_target.py   HAR volatility targeting (in-sample) -> output/har_vol_target/
│   ├── 09_vol_model_comparison.py  HAR vs ridge, GARCH, EWMA -> output/vol_model_comparison/
│   ├── 10_oos.py              out-of-sample evaluation (1975-1994), run once -> output/oos/
│   └── 11_post_oos.py         post-OOS observations, not pre-registered -> output/post_oos/
└── requirements.txt
```

## Data

All data comes from WRDS, except the reversal factors, which come from the Kenneth French data library. Prices and returns span 1974-01 to 2025-12. That gives the first
rebalance (reference date Feb-1975) a full 12-month look-back.

| File | WRDS source | Content |
|---|---|---|
| `sp500_membership` | `crsp.dsp500list_v2` | S&P 500 constituent spells |
| `crsp_monthly`, `crsp_daily/<year>` | `crsp.msf_v2`, `crsp.dsf_v2` | prices, returns (incl. delisting), shares, volume |
| `crsp_stocknames` | `crsp.stocknames_v2` | ticker, CUSIP, name and exchange history |
| `ccm_link` | `crsp.ccmxpf_lnkhist` | CRSP PERMNO to Compustat GVKEY links |
| `comp_gics_history` | `comp.co_hgic` | point-in-time GICS classification (from 1999-06) |
| `comp_company` | `comp.company` | header GICS and SIC (fallback before 1999) |
| `comp_funda` | `comp.funda` | annual fundamentals: book-equity and profitability inputs |
| `ff5_factors_{daily,monthly}` | `ff.fivefactors_*` | Fama-French five factors, momentum (UMD), risk-free rate |
| `ff_reversal_monthly` | French data library (not WRDS) | short-term and long-term reversal factors |
| `sp500_index_daily` | `crsp.dsp500_v2` | S&P 500 index returns and level |
| `spy_daily`, `spmo_daily` | `crsp.dsf_v2` | SPY and SPMO ETF benchmarks |

`data/raw/MANIFEST.json` records the exact SQL, row counts and date range of every table.

### Design choices

- **Universe.** Every security that was an S&P 500 constituent at any time since 1974, taken from
  CRSP's historical constituent list rather than today's members, so there is no survivorship bias.
  Membership is applied month by month in research code.
- **Returns.** CRSP CIZ (v2) monthly returns include delisting returns; daily returns are kept
  as reported.
- **CRSP-Compustat link.** Link types `LC`, `LU`, `LS` with primary flags `P`, `C` and `J`. Including
  `J` keeps secondary share classes (GOOG, NWS, FOX, ...) linked to their company; where several links
  are valid on a date the priority is `P` > `C` > `J`.
- **Sectors.** Point-in-time GICS where available (1999-06 onward). Earlier months use the Compustat
  header GICS, which is backfilled and therefore not point-in-time, with CRSP SIC as the last
  fallback.
- **Book equity.** Fama-French definition (stockholders' equity + deferred taxes - preferred stock),
  using the fiscal year ending in calendar year *t*-1 from July of year *t*.

### Data quality

`scripts/02_validate_data.py` checks completeness against the manifest, key uniqueness, validity
intervals, trading-calendar alignment, constituent counts, return consistency (monthly vs.
compounded daily), value ranges and identifier coverage, and exits non-zero on any structural
failure. All structural checks pass; the remaining notes (e.g. months in which CRSP has no return
for a constituent, pre-1999 sector coverage) are documented with how research code handles them in
[reports/data_validation.md](reports/data_validation.md).

## Reproducing

Requires Python 3.10+ and a WRDS account with access to CRSP, Compustat and the Fama-French library.

```bash
pip install -r requirements.txt
export WRDS_USERNAME=<your-username>       # PowerShell: $env:WRDS_USERNAME="<your-username>"
python scripts/01_fetch_wrds.py            # ~30 min; prompts for the password once and can save it to pgpass
python scripts/02_validate_data.py
python scripts/03_spmo_proxy.py            # ~15 s
python scripts/04_attribution.py           # ~20 s
python scripts/05_episodes.py              # ~20 s
python scripts/06_double_sort.py           # ~30 s
python scripts/07_factor_regressions.py    # ~40 s
python scripts/08_har_vol_target.py        # ~30 s
python scripts/09_vol_model_comparison.py  # ~1.5 min
python scripts/10_oos.py                   # out-of-sample; design frozen at tag pre-oos
python scripts/11_post_oos.py              # supplementary, run after the out-of-sample evaluation
```

Raw WRDS data is licensed and is not included in this repository.
