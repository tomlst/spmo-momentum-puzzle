"""
Download the raw research dataset from WRDS.

Universe: every security that was an S&P 500 constituent at any time on or after
UNIVERSE_FROM (CRSP crsp.dsp500list_v2), i.e. point-in-time and survivorship-free.
Prices and returns cover START..END; Compustat annuals start earlier so that
book equity is available for the first formation year.

Usage:
    set WRDS_USERNAME=<your username>      (password is read from pgpass)
    python scripts/01_fetch_wrds.py [--out data/raw]

Output: one parquet file per table (daily CRSP split by year) plus MANIFEST.json
recording the SQL, row counts and date ranges of every table.
"""
import argparse
import json
import os
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import wrds

ROOT = Path(__file__).resolve().parents[1]

START, END = "1974-01-01", "2025-12-31"
UNIVERSE_FROM = "1974-01-01"
FUNDA_START = "1970-01-01"
SPY_PERMNO, SPMO_PERMNO = 84398, 15725

UNIVERSE = f"SELECT DISTINCT permno FROM crsp.dsp500list_v2 WHERE mbrenddt >= DATE '{UNIVERSE_FROM}'"
# P = primary, C = primary (Compustat-assigned), J = secondary share class (e.g. GOOG vs GOOGL)
LINK_FILTER = "linktype IN ('LC', 'LU', 'LS') AND linkprim IN ('P', 'C', 'J')"
UNIVERSE_GVKEYS = f"SELECT DISTINCT gvkey FROM crsp.ccmxpf_lnkhist WHERE lpermno IN ({UNIVERSE}) AND {LINK_FILTER}"

INT_COLS = {"permno", "permco", "shrout", "siccd", "totcnt", "indno", "indfam", "fyear", "fyr", "sich"}

# name -> (sql, date columns)
TABLES = {
    "sp500_membership": (
        "SELECT permno, indno, indfam, mbrstartdt, mbrenddt, mbrflg FROM crsp.dsp500list_v2 "
        "ORDER BY permno, mbrstartdt",
        ["mbrstartdt", "mbrenddt"]),
    "crsp_monthly": (
        "SELECT permno, permco, ticker, issuernm, siccd, primaryexch, sharetype, securitytype, securitysubtype, "
        "mthcaldt, mthprc, mthprcflg, mthcap, mthret, mthretx, mthretflg, mthdelflg, mthvol, shrout, "
        f"mthcumfacpr, mthcumfacshr FROM crsp.msf_v2 WHERE permno IN ({UNIVERSE}) "
        f"AND mthcaldt BETWEEN DATE '{START}' AND DATE '{END}' ORDER BY permno, mthcaldt",
        ["mthcaldt"]),
    "crsp_stocknames": (
        "SELECT permno, permco, namedt, nameenddt, securitybegdt, securityenddt, hdrcusip, cusip, ticker, "
        "issuernm, primaryexch, shareclass, sharetype, securitytype, securitysubtype, siccd "
        f"FROM crsp.stocknames_v2 WHERE permno IN ({UNIVERSE}) ORDER BY permno, namedt",
        ["namedt", "nameenddt", "securitybegdt", "securityenddt"]),
    "ccm_link": (
        "SELECT gvkey, lpermno AS permno, lpermco AS permco, linktype, linkprim, liid, linkdt, linkenddt "
        f"FROM crsp.ccmxpf_lnkhist WHERE lpermno IN ({UNIVERSE}) AND {LINK_FILTER} ORDER BY permno, linkdt",
        ["linkdt", "linkenddt"]),
    "comp_gics_history": (
        "SELECT gvkey, indtype, gsector, ggroup, gind, gsubind, indfrom, indthru "
        f"FROM comp.co_hgic WHERE gvkey IN ({UNIVERSE_GVKEYS}) ORDER BY gvkey, indfrom",
        ["indfrom", "indthru"]),
    "comp_company": (
        "SELECT gvkey, conm, sic, naics, gsector, ggroup, gind, gsubind, ipodate, dldte, dlrsn "
        f"FROM comp.company WHERE gvkey IN ({UNIVERSE_GVKEYS}) ORDER BY gvkey",
        ["ipodate", "dldte"]),
    "comp_funda": (
        "SELECT gvkey, datadate, fyear, fyr, curcd, sich, seq, ceq, pstk, pstkrv, pstkl, txditc, at, lt, "
        "csho, prcc_f, revt, cogs, xsga, xint, ib, ni, oancf "
        f"FROM comp.funda WHERE gvkey IN ({UNIVERSE_GVKEYS}) "
        "AND indfmt = 'INDL' AND datafmt = 'STD' AND popsrc = 'D' AND consol = 'C' "
        f"AND datadate BETWEEN DATE '{FUNDA_START}' AND DATE '{END}' ORDER BY gvkey, datadate",
        ["datadate"]),
    "ff5_factors_monthly": (
        "SELECT date, mktrf, smb, hml, rmw, cma, rf, umd FROM ff.fivefactors_monthly "
        f"WHERE date BETWEEN DATE '{START}' AND DATE '{END}' ORDER BY date",
        ["date"]),
    "ff5_factors_daily": (
        "SELECT date, mktrf, smb, hml, rmw, cma, rf, umd FROM ff.fivefactors_daily "
        f"WHERE date BETWEEN DATE '{START}' AND DATE '{END}' ORDER BY date",
        ["date"]),
    "sp500_index_daily": (
        "SELECT caldt AS date, vwretd, vwretx, ewretd, ewretx, sprtrn, spindx, totval, totcnt "
        f"FROM crsp.dsp500_v2 WHERE caldt BETWEEN DATE '{START}' AND DATE '{END}' ORDER BY caldt",
        ["date"]),
    "spy_daily": (
        "SELECT permno, dlycaldt AS date, dlyret, dlyretx, dlyprc, ticker FROM crsp.dsf_v2 "
        f"WHERE permno = {SPY_PERMNO} AND dlycaldt <= DATE '{END}' ORDER BY dlycaldt",
        ["date"]),
    "spmo_daily": (
        "SELECT permno, dlycaldt AS date, dlyret, dlyretx, dlyprc, ticker FROM crsp.dsf_v2 "
        f"WHERE permno = {SPMO_PERMNO} AND dlycaldt <= DATE '{END}' ORDER BY dlycaldt",
        ["date"]),
}

DAILY_SQL = (
    "SELECT permno, permco, dlycaldt, dlyret, dlyretx, dlyretmissflg, dlydelflg, dlyprc, dlyprcflg, dlyclose, "
    "dlyopen, dlyhigh, dlylow, dlyvol, dlycap, shrout, dlycumfacpr, dlycumfacshr FROM crsp.dsf_v2 "
    f"WHERE permno IN ({UNIVERSE}) AND dlycaldt BETWEEN DATE '{{y}}-01-01' AND DATE '{{y}}-12-31' "
    "ORDER BY permno, dlycaldt")


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Nullable dtypes: identifiers -> Int64, measures -> Float64, text -> string."""
    df = df.convert_dtypes(convert_integer=False, convert_boolean=False)
    for c in INT_COLS & set(df.columns):
        df[c] = df[c].astype("Float64").astype("Int64")
    return df


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def summarize(df: pd.DataFrame, sql: str, date_col: str | None) -> dict:
    info = {"rows": len(df), "columns": list(df.columns), "sql": sql}
    if "permno" in df.columns:
        info["n_permno"] = int(df["permno"].nunique())
    if date_col:
        info["date_min"] = str(df[date_col].min().date())
        info["date_max"] = str(df[date_col].max().date())
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "raw")
    args = ap.parse_args()

    username = os.environ.get("WRDS_USERNAME")
    if not username:
        raise SystemExit("Set the WRDS_USERNAME environment variable.")
    db = wrds.Connection(wrds_username=username)

    manifest = {
        "source": "WRDS (CRSP CIZ v2, Compustat, Fama-French)",
        "pulled_at": datetime.now().isoformat(timespec="seconds"),
        "params": {"start": START, "end": END, "universe_from": UNIVERSE_FROM, "funda_start": FUNDA_START},
        "versions": {p: version(p) for p in ("wrds", "pandas", "pyarrow")},
        "datasets": {},
    }

    for name, (sql, date_cols) in TABLES.items():
        df = normalize(db.raw_sql(sql, date_cols=date_cols))
        write_parquet(df, args.out / f"{name}.parquet")
        manifest["datasets"][name] = summarize(df, sql, date_cols[0])
        print(f"{name:22s} {len(df):>9,} rows")

    years = range(int(START[:4]), int(END[:4]) + 1)
    daily = {"layout": "crsp_daily/<year>.parquet", "sql": DAILY_SQL, "rows": {}}
    for y in years:
        df = normalize(db.raw_sql(DAILY_SQL.format(y=y), date_cols=["dlycaldt"]))
        write_parquet(df, args.out / "crsp_daily" / f"{y}.parquet")
        daily["rows"][str(y)] = len(df)
        print(f"crsp_daily/{y}        {len(df):>9,} rows")
    manifest["datasets"]["crsp_daily"] = daily
    db.close()

    with open(args.out / "MANIFEST.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Done -> {args.out}")


if __name__ == "__main__":
    main()
