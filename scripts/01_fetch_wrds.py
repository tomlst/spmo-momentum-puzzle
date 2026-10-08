"""
Download the raw research dataset from WRDS and the Kenneth French data library.

Universe: every security that was an S&P 500 constituent at any time on or after
UNIVERSE_FROM (CRSP crsp.dsp500list_v2), i.e. point-in-time and survivorship-free.
Prices and returns cover START..END; Compustat annuals start earlier so that
book equity is available for the first formation year.

Reversal factors (short-term and long-term) are not in the WRDS Fama-French tables used here and
come from the French data library directly; the library revises its files, so the MANIFEST keeps
each file's URL, CRSP vintage and download time.

Usage:
    set WRDS_USERNAME=<your username>      (password is read from pgpass)
    python scripts/01_fetch_wrds.py [--out data/raw] [--only DATASET ...]

--only refreshes the named datasets and updates their entries in an existing MANIFEST.json.

Output: one parquet file per table (daily CRSP split by year) plus MANIFEST.json
recording the SQL, row counts and date ranges of every table.
"""
import argparse
import io
import json
import os
import re
import urllib.request
import zipfile
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

# Kenneth French data library: dataset -> {column: zip file}. Monthly values are in percent.
FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FRENCH = {
    "ff_reversal_monthly": {"st_rev": "F-F_ST_Reversal_Factor_CSV.zip", "lt_rev": "F-F_LT_Reversal_Factor_CSV.zip"},
}
SOURCE = "WRDS (CRSP CIZ v2, Compustat, Fama-French) and the Kenneth French data library"

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


def french_monthly(files: dict[str, str]) -> tuple[pd.DataFrame, dict[str, str]]:
    """Monthly factors from the French library as decimals, dated at month start like the WRDS
    Fama-French tables, plus the CRSP vintage (YYYYMM) each file was built from."""
    cols, notes = [], {}
    for col, name in files.items():
        with urllib.request.urlopen(FRENCH_URL + name, timeout=60) as resp:
            z = zipfile.ZipFile(io.BytesIO(resp.read()))
        text = z.read(z.namelist()[0]).decode("latin-1")
        lines = text.splitlines()
        vintage = re.search(r"created using the (\d{6}) CRSP database", text)
        notes[col] = vintage.group(1) if vintage else "unknown"
        start = next(i for i, line in enumerate(lines) if line.startswith(",")) + 1
        rows = []
        for line in lines[start:]:
            if not line.strip():                               # the monthly block ends at a blank line
                break
            ym, value = (v.strip() for v in line.split(","))
            rows.append((pd.Timestamp(f"{ym[:4]}-{ym[4:]}-01"), float(value)))
        sr = pd.DataFrame(rows, columns=["date", col]).set_index("date")[col]
        cols.append(sr.mask(sr <= -99.99) / 100)
    df = pd.concat(cols, axis=1).loc[START:END].reset_index()
    return df, notes


def summarize(df: pd.DataFrame, sql: str | None, date_col: str | None) -> dict:
    info = {"rows": len(df), "columns": list(df.columns)}
    if sql:
        info["sql"] = sql
    if "permno" in df.columns:
        info["n_permno"] = int(df["permno"].nunique())
    if date_col:
        info["date_min"] = str(df[date_col].min().date())
        info["date_max"] = str(df[date_col].max().date())
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--only", nargs="+", metavar="DATASET", choices=[*TABLES, "crsp_daily", *FRENCH],
                    help="refresh only these datasets and update their entries in MANIFEST.json")
    args = ap.parse_args()
    wanted = set(args.only or [*TABLES, "crsp_daily", *FRENCH])
    now = datetime.now().isoformat(timespec="seconds")

    path = args.out / "MANIFEST.json"
    if args.only:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "pulled_at": now,
            "params": {"start": START, "end": END, "universe_from": UNIVERSE_FROM, "funda_start": FUNDA_START},
            "versions": {p: version(p) for p in ("wrds", "pandas", "pyarrow")},
            "datasets": {},
        }
    manifest["source"] = SOURCE

    if wanted & {*TABLES, "crsp_daily"}:
        username = os.environ.get("WRDS_USERNAME")
        if not username:
            raise SystemExit("Set the WRDS_USERNAME environment variable.")
        db = wrds.Connection(wrds_username=username)
        for name, (sql, date_cols) in TABLES.items():
            if name not in wanted:
                continue
            df = normalize(db.raw_sql(sql, date_cols=date_cols))
            write_parquet(df, args.out / f"{name}.parquet")
            manifest["datasets"][name] = summarize(df, sql, date_cols[0])
            print(f"{name:22s} {len(df):>9,} rows")

        if "crsp_daily" in wanted:
            years = range(int(START[:4]), int(END[:4]) + 1)
            daily = {"layout": "crsp_daily/<year>.parquet", "sql": DAILY_SQL, "rows": {}}
            for y in years:
                df = normalize(db.raw_sql(DAILY_SQL.format(y=y), date_cols=["dlycaldt"]))
                write_parquet(df, args.out / "crsp_daily" / f"{y}.parquet")
                daily["rows"][str(y)] = len(df)
                print(f"crsp_daily/{y}        {len(df):>9,} rows")
            manifest["datasets"]["crsp_daily"] = daily
        db.close()

    for name, files in FRENCH.items():
        if name not in wanted:
            continue
        df, notes = french_monthly(files)
        df = normalize(df)
        write_parquet(df, args.out / f"{name}.parquet")
        manifest["datasets"][name] = {**summarize(df, None, "date"), "pulled_at": now,
                                      "urls": {c: FRENCH_URL + f for c, f in files.items()}, "crsp_vintage": notes}
        print(f"{name:22s} {len(df):>9,} rows")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Done -> {args.out}")


if __name__ == "__main__":
    main()
