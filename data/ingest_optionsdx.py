"""Ingest optionsDX .7z archives into unified Parquet schema.

optionsDX uses a WIDE format: each row contains BOTH call and put data for
a given (date, expiry, strike) combination. Columns are prefixed C_ and P_.

This script:
  - Extracts .7z archives from data/raw/optionsdx/
  - Reads the wide-format CSVs (.txt files inside)
  - Melts each row into two rows: one call, one put
  - Normalizes to the unified schema
  - Writes yearly Parquet files to data/processed/

Handles two naming patterns:
  Yearly:    spy_eod_2010-bndqqt.7z  (2010-2021)
  Quarterly: spy_eod_2022q1-ww0cra.7z (2022-2023)

Usage:
    python data/ingest_optionsdx.py
    python data/ingest_optionsdx.py --year 2020
    python data/ingest_optionsdx.py --force
"""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path

import click
import pandas as pd
from loguru import logger

RAW_DIR = Path(__file__).parent / "raw" / "optionsdx"
PROCESSED_DIR = Path(__file__).parent / "processed"

# Regex patterns for the two naming conventions
_YEARLY_RE = re.compile(r"spy_eod_(\d{4})-\w+\.7z$")
_QUARTERLY_RE = re.compile(r"spy_eod_(\d{4})q(\d)-\w+\.7z$")

SCHEMA_COLS = [
    "trade_date", "expiry", "dte", "strike", "option_type",
    "bid", "ask", "mid", "last_price", "volume", "open_interest",
    "iv", "delta", "gamma", "theta", "vega", "rho",
    "underlying_close", "bid_ask_pct", "source",
]


def _parse_archive_year_quarter(filename: str) -> tuple[int, int | None]:
    """Extract year and optional quarter from a .7z filename."""
    m = _QUARTERLY_RE.match(filename)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = _YEARLY_RE.match(filename)
    if m:
        return int(m.group(1)), None
    raise ValueError(f"Unrecognized archive filename: {filename}")


def _process_wide_csv(csv_path: Path) -> pd.DataFrame:
    """Read an optionsDX wide-format CSV and melt into long format.

    optionsDX format has one row per (date, expiry, strike) with columns:
      [QUOTE_DATE], [UNDERLYING_LAST], [EXPIRE_DATE], [DTE], [STRIKE],
      [C_DELTA], [C_GAMMA], [C_VEGA], [C_THETA], [C_RHO], [C_IV],
      [C_VOLUME], [C_LAST], [C_BID], [C_ASK], [C_SIZE],
      [P_DELTA], [P_GAMMA], [P_VEGA], [P_THETA], [P_RHO], [P_IV],
      [P_VOLUME], [P_LAST], [P_BID], [P_ASK], [P_SIZE]

    We melt each row into two rows: one call, one put.
    """
    logger.debug(f"  Processing: {csv_path.name}")

    df = pd.read_csv(csv_path, low_memory=False)

    # Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]

    # Verify we have the expected wide format
    if "[STRIKE]" not in df.columns:
        logger.warning(f"  Skipping {csv_path.name}: no [STRIKE] column. Cols: {list(df.columns)[:5]}")
        return pd.DataFrame()

    # Shared columns
    shared = {
        "[QUOTE_DATE]": "trade_date",
        "[EXPIRE_DATE]": "expiry",
        "[DTE]": "dte",
        "[STRIKE]": "strike",
        "[UNDERLYING_LAST]": "underlying_close",
    }

    # Per-side column mapping (C_ for calls, P_ for puts)
    side_map = {
        "DELTA": "delta",
        "GAMMA": "gamma",
        "VEGA": "vega",
        "THETA": "theta",
        "RHO": "rho",
        "IV": "iv",
        "VOLUME": "volume",
        "LAST": "last_price",
        "BID": "bid",
        "ASK": "ask",
    }

    # Rename shared columns
    rename = {k: v for k, v in shared.items() if k in df.columns}
    df = df.rename(columns=rename)

    # Build call rows
    call_rename = {}
    for suffix, target in side_map.items():
        col = f"[C_{suffix}]"
        if col in df.columns:
            call_rename[col] = target
    calls = df[list(rename.values()) + list(call_rename.keys())].copy()
    calls = calls.rename(columns=call_rename)
    calls["option_type"] = "C"

    # Build put rows
    put_rename = {}
    for suffix, target in side_map.items():
        col = f"[P_{suffix}]"
        if col in df.columns:
            put_rename[col] = target
    puts = df[list(rename.values()) + list(put_rename.keys())].copy()
    puts = puts.rename(columns=put_rename)
    puts["option_type"] = "P"

    # Concatenate
    long = pd.concat([calls, puts], ignore_index=True)

    # Parse dates
    long["trade_date"] = pd.to_datetime(long["trade_date"].astype(str).str.strip(), format="mixed", dayfirst=False)
    long["expiry"] = pd.to_datetime(long["expiry"].astype(str).str.strip(), format="mixed", dayfirst=False)

    # Numeric columns
    for col in ["strike", "bid", "ask", "last_price", "volume",
                 "iv", "delta", "gamma", "theta", "vega", "rho",
                 "underlying_close", "dte"]:
        if col in long.columns:
            long[col] = pd.to_numeric(long[col], errors="coerce")

    # Fill missing numeric with 0
    for col in ["bid", "ask", "last_price", "volume",
                 "gamma", "theta", "vega", "rho"]:
        if col in long.columns:
            long[col] = long[col].fillna(0)

    # open_interest is not in optionsDX free EOD data — set to 0
    long["open_interest"] = 0

    # Recompute DTE from dates (more reliable than the file's DTE which can be fractional)
    long["dte"] = (long["expiry"] - long["trade_date"]).dt.days

    # Derived fields
    long["mid"] = (long["bid"] + long["ask"]) / 2
    long["bid_ask_pct"] = (long["ask"] - long["bid"]) / long["mid"].replace(0, float("nan"))

    # Tag source
    long["source"] = "optionsdx"

    return long


def _extract_and_read_7z(archive_path: Path, tmp_dir: Path) -> pd.DataFrame:
    """Extract a .7z archive and read all CSVs/TXTs inside."""
    try:
        import py7zr
    except ImportError:
        raise ImportError("py7zr required: pip install py7zr")

    extract_dir = tmp_dir / archive_path.stem
    extract_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Extracting {archive_path.name}...")
    with py7zr.SevenZipFile(archive_path, mode="r") as z:
        z.extractall(path=extract_dir)

    # Find all data files recursively
    data_files = sorted(extract_dir.rglob("*.txt")) + sorted(extract_dir.rglob("*.csv"))
    if not data_files:
        logger.warning(f"  No data files found in {archive_path.name}")
        return pd.DataFrame()

    logger.info(f"  Found {len(data_files)} data files in {archive_path.name}")

    dfs = []
    for f in data_files:
        df = _process_wide_csv(f)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def _write_yearly_parquet(df: pd.DataFrame, year: int) -> Path:
    """Write a DataFrame to a yearly Parquet file with unified schema."""
    df = df.dropna(subset=["strike", "trade_date", "expiry"])
    df = df[df["dte"] >= 0]

    for col in SCHEMA_COLS:
        if col not in df.columns:
            if col in ("trade_date", "expiry", "option_type", "source"):
                df[col] = ""
            else:
                df[col] = 0.0

    df = df[SCHEMA_COLS]

    out_path = PROCESSED_DIR / f"spy_options_{year}.parquet"
    df.to_parquet(out_path, index=False, engine="pyarrow")
    logger.info(f"Wrote {len(df):,} rows to {out_path}")
    return out_path


def ingest(year: int | None = None, force: bool = False) -> None:
    """Ingest optionsDX .7z archives and write yearly Parquet files."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_DIR.exists():
        logger.error(f"Raw directory not found: {RAW_DIR}")
        logger.info("Place optionsDX .7z files in data/raw/optionsdx/")
        return

    archives = sorted(RAW_DIR.glob("*.7z"))
    if not archives:
        logger.warning(f"No .7z archives found in {RAW_DIR}")
        return

    logger.info(f"Found {len(archives)} .7z archives in {RAW_DIR}")

    # Group archives by year
    year_archives: dict[int, list[tuple[Path, int | None]]] = {}
    for archive in archives:
        try:
            yr, qtr = _parse_archive_year_quarter(archive.name)
        except ValueError as e:
            logger.warning(f"  Skipping: {archive.name} ({e})")
            continue
        if year is not None and yr != year:
            continue
        year_archives.setdefault(yr, []).append((archive, qtr))

    if not year_archives:
        logger.warning("No matching archives found")
        return

    tmp_dir = Path(tempfile.mkdtemp(prefix="optionsdx_"))
    try:
        for yr in sorted(year_archives.keys()):
            out_path = PROCESSED_DIR / f"spy_options_{yr}.parquet"
            if out_path.exists() and not force:
                logger.info(f"Skipping {yr}: {out_path.name} already exists (use --force)")
                continue

            archive_list = year_archives[yr]
            quarters = [q for _, q in archive_list if q is not None]
            if quarters:
                logger.info(f"Processing {yr}: {len(archive_list)} quarterly archives (q{',q'.join(str(q) for q in sorted(quarters))})")
            else:
                logger.info(f"Processing {yr}: yearly archive")

            year_dfs = []
            for archive_path, qtr in sorted(archive_list, key=lambda x: x[1] or 0):
                df = _extract_and_read_7z(archive_path, tmp_dir)
                if not df.empty:
                    year_dfs.append(df)

            if not year_dfs:
                logger.warning(f"  No data extracted for {yr}")
                continue

            combined = pd.concat(year_dfs, ignore_index=True)
            combined = combined.drop_duplicates(
                subset=["trade_date", "expiry", "strike", "option_type"],
                keep="first",
            )
            logger.info(f"  {yr}: {len(combined):,} rows after dedup")
            _write_yearly_parquet(combined, yr)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.debug(f"Cleaned up temp directory")

    # Summary
    processed = sorted(PROCESSED_DIR.glob("spy_options_[0-9][0-9][0-9][0-9].parquet"))
    if processed:
        years_done = [p.stem.replace("spy_options_", "") for p in processed]
        logger.info(f"Total: {len(processed)} yearly Parquet files in {PROCESSED_DIR}")
        logger.info(f"Years: {', '.join(years_done)}")


@click.command()
@click.option("--year", type=int, default=None, help="Process only this year")
@click.option("--force", is_flag=True, help="Re-ingest even if Parquet exists")
@click.option("--debug", is_flag=True, help="Enable DEBUG logging")
def main(year: int | None, force: bool, debug: bool) -> None:
    """Ingest optionsDX .7z archives into unified Parquet files."""
    from utils.logging import setup_logging
    setup_logging("DEBUG" if debug else None)
    ingest(year, force=force)


if __name__ == "__main__":
    main()
