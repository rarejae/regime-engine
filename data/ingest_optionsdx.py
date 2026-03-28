"""Ingest optionsDX CSV files into unified Parquet schema.

Reads optionsDX monthly CSVs from data/raw/optionsdx/, normalizes column names,
computes derived fields (dte, mid, bid_ask_pct), tags source='optionsdx',
and writes yearly Parquet files to data/processed/.

optionsDX CSV format (EOD):
    Columns vary slightly by year but generally include:
    [QUOTE_DATE], [EXPIRE_DATE], [STRIKE], [C/P], [BID], [ASK], [LAST],
    [VOLUME], [OPEN_INT], [IV], [DELTA], [GAMMA], [THETA], [VEGA], [RHO],
    [UNDERLYING_LAST]

Usage:
    python -m data.ingest_optionsdx
    python -m data.ingest_optionsdx --year 2020
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import click
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from utils.logging import setup_logging

RAW_DIR = Path(__file__).parent / "raw" / "optionsdx"
PROCESSED_DIR = Path(__file__).parent / "processed"

# Mapping from optionsDX column names (various formats) to unified schema
COLUMN_MAP = {
    # Quote date variants
    " [QUOTE_DATE]": "trade_date",
    "[QUOTE_DATE]": "trade_date",
    "QUOTE_DATE": "trade_date",
    "quote_date": "trade_date",
    # Expiry variants
    " [EXPIRE_DATE]": "expiry",
    "[EXPIRE_DATE]": "expiry",
    "EXPIRE_DATE": "expiry",
    "expire_date": "expiry",
    # Strike
    " [STRIKE]": "strike",
    "[STRIKE]": "strike",
    "STRIKE": "strike",
    "strike": "strike",
    # Option type
    " [C/P]": "option_type",
    "[C/P]": "option_type",
    "C/P": "option_type",
    "c_p": "option_type",
    # Pricing
    " [BID]": "bid",
    "[BID]": "bid",
    "BID": "bid",
    " [ASK]": "ask",
    "[ASK]": "ask",
    "ASK": "ask",
    " [LAST]": "last_price",
    "[LAST]": "last_price",
    "LAST": "last_price",
    # Volume / OI
    " [VOLUME]": "volume",
    "[VOLUME]": "volume",
    "VOLUME": "volume",
    " [OPEN_INT]": "open_interest",
    "[OPEN_INT]": "open_interest",
    "OPEN_INT": "open_interest",
    # Greeks
    " [IV]": "iv",
    "[IV]": "iv",
    "IV": "iv",
    " [DELTA]": "delta",
    "[DELTA]": "delta",
    "DELTA": "delta",
    " [GAMMA]": "gamma",
    "[GAMMA]": "gamma",
    "GAMMA": "gamma",
    " [THETA]": "theta",
    "[THETA]": "theta",
    "THETA": "theta",
    " [VEGA]": "vega",
    "[VEGA]": "vega",
    "VEGA": "vega",
    " [RHO]": "rho",
    "[RHO]": "rho",
    "RHO": "rho",
    # Underlying
    " [UNDERLYING_LAST]": "underlying_close",
    "[UNDERLYING_LAST]": "underlying_close",
    "UNDERLYING_LAST": "underlying_close",
}

DATE_FORMATS = ["%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"]


def _parse_date(s: str) -> datetime:
    """Try multiple date formats for optionsDX date columns."""
    s = s.strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {s!r}")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename optionsDX columns to unified schema names."""
    rename = {}
    for col in df.columns:
        col_stripped = col.strip()
        if col in COLUMN_MAP:
            rename[col] = COLUMN_MAP[col]
        elif col_stripped in COLUMN_MAP:
            rename[col] = COLUMN_MAP[col_stripped]
    df = df.rename(columns=rename)
    return df


def _process_csv(csv_path: Path) -> pd.DataFrame:
    """Read and normalize a single optionsDX CSV file."""
    logger.debug(f"Processing {csv_path.name}")

    df = pd.read_csv(csv_path, low_memory=False)
    df = _normalize_columns(df)

    required = {"trade_date", "expiry", "strike", "option_type"}
    missing = required - set(df.columns)
    if missing:
        logger.warning(f"Skipping {csv_path.name}: missing columns {missing}")
        return pd.DataFrame()

    # Parse dates
    df["trade_date"] = pd.to_datetime(df["trade_date"].astype(str).apply(lambda x: _parse_date(x)))
    df["expiry"] = pd.to_datetime(df["expiry"].astype(str).apply(lambda x: _parse_date(x)))

    # Normalize option type to single char
    df["option_type"] = df["option_type"].astype(str).str.strip().str.upper().str[0]
    df = df[df["option_type"].isin(["C", "P"])]

    # Numeric columns
    for col in ["strike", "bid", "ask", "last_price", "volume", "open_interest",
                 "iv", "delta", "gamma", "theta", "vega", "rho", "underlying_close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill missing numeric with 0
    for col in ["bid", "ask", "last_price", "volume", "open_interest",
                 "gamma", "theta", "vega", "rho"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Compute derived fields
    df["dte"] = (df["expiry"] - df["trade_date"]).dt.days
    df["mid"] = (df["bid"] + df["ask"]) / 2
    df["bid_ask_pct"] = ((df["ask"] - df["bid"]) / df["mid"].replace(0, float("nan")))

    # Tag source
    df["source"] = "optionsdx"

    # Ensure all schema columns present
    for col in ["rho", "underlying_close"]:
        if col not in df.columns:
            df[col] = 0.0

    return df


def ingest(year: int | None = None) -> None:
    """Ingest optionsDX CSVs and write yearly Parquet files.

    Args:
        year: If specified, only process CSVs for that year. Otherwise process all.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_DIR.exists():
        logger.error(f"Raw directory not found: {RAW_DIR}")
        logger.info("Download optionsDX SPY EOD CSVs from https://www.optionsdx.com/product/spy-option-chains/")
        return

    csv_files = sorted(RAW_DIR.glob("*.csv"))
    if not csv_files:
        logger.warning(f"No CSV files found in {RAW_DIR}")
        return

    logger.info(f"Found {len(csv_files)} CSV files in {RAW_DIR}")

    all_dfs: list[pd.DataFrame] = []
    for csv_path in csv_files:
        df = _process_csv(csv_path)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        logger.error("No data processed from CSVs")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Combined: {len(combined)} rows")

    # Drop invalid rows
    combined = combined.dropna(subset=["strike", "trade_date", "expiry"])
    combined = combined[combined["dte"] >= 0]

    # Group by year and write Parquet
    combined["year"] = combined["trade_date"].dt.year
    years = sorted(combined["year"].unique())

    if year is not None:
        years = [y for y in years if y == year]

    for yr in years:
        yr_df = combined[combined["year"] == yr].drop(columns=["year"])

        # Select unified schema columns
        schema_cols = [
            "trade_date", "expiry", "dte", "strike", "option_type",
            "bid", "ask", "mid", "last_price", "volume", "open_interest",
            "iv", "delta", "gamma", "theta", "vega", "rho",
            "underlying_close", "bid_ask_pct", "source",
        ]
        for col in schema_cols:
            if col not in yr_df.columns:
                yr_df[col] = 0.0 if col not in ("trade_date", "expiry", "option_type", "source") else ""

        yr_df = yr_df[schema_cols]

        out_path = PROCESSED_DIR / f"spy_options_{yr}.parquet"
        yr_df.to_parquet(out_path, index=False, engine="pyarrow")
        logger.info(f"Wrote {len(yr_df):,} rows to {out_path}")


@click.command()
@click.option("--year", type=int, default=None, help="Process only this year")
@click.option("--debug", is_flag=True, help="Enable DEBUG logging")
def main(year: int | None, debug: bool) -> None:
    """Ingest optionsDX CSVs into unified Parquet files."""
    setup_logging("DEBUG" if debug else None)
    ingest(year)


if __name__ == "__main__":
    main()
