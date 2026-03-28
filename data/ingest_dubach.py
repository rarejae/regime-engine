"""Ingest Dubach dataset (philippdubach/options-dataset-hist) into unified Parquet schema.

Downloads (if needed) and normalizes the Dubach SPY options Parquet file,
maps columns to the unified schema, tags source='dubach', and writes
yearly Parquet files to data/processed/.

Dubach download URL:
    https://static.philippdubach.com/data/options/spy/options.parquet

Usage:
    python -m data.ingest_dubach
    python -m data.ingest_dubach --download   # force re-download
    python -m data.ingest_dubach --year 2009
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import click
import pandas as pd
from loguru import logger

from utils.logging import setup_logging

RAW_DIR = Path(__file__).parent / "raw" / "dubach"
PROCESSED_DIR = Path(__file__).parent / "processed"
DUBACH_URL = "https://static.philippdubach.com/data/options/spy/options.parquet"
DUBACH_FILE = RAW_DIR / "spy_options_dubach.parquet"


def download_dubach(force: bool = False) -> Path:
    """Download the Dubach SPY options Parquet if not already present.

    Args:
        force: Re-download even if file exists.

    Returns:
        Path to the downloaded file.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if DUBACH_FILE.exists() and not force:
        logger.info(f"Dubach file already exists: {DUBACH_FILE}")
        return DUBACH_FILE

    logger.info(f"Downloading Dubach SPY options from {DUBACH_URL}")
    urllib.request.urlretrieve(DUBACH_URL, str(DUBACH_FILE))
    logger.info(f"Downloaded to {DUBACH_FILE}")
    return DUBACH_FILE


def _normalize_dubach(df: pd.DataFrame) -> pd.DataFrame:
    """Map Dubach columns to unified schema."""
    # The Dubach dataset uses various column naming conventions.
    # We normalize to the unified schema.
    col_map = {}
    cols_lower = {c.lower(): c for c in df.columns}

    # Date columns
    for candidate in ["quote_date", "date", "quotedate", "trade_date"]:
        if candidate in cols_lower:
            col_map[cols_lower[candidate]] = "trade_date"
            break

    for candidate in ["expiration", "expire_date", "expiry", "expirationdate"]:
        if candidate in cols_lower:
            col_map[cols_lower[candidate]] = "expiry"
            break

    # Standard fields
    field_candidates = {
        "strike": ["strike", "strike_price"],
        "option_type": ["type", "option_type", "cp_flag", "c_p", "cp"],
        "bid": ["bid", "best_bid"],
        "ask": ["ask", "best_offer", "best_ask"],
        "last_price": ["last", "last_price", "close"],
        "volume": ["volume", "vol"],
        "open_interest": ["open_interest", "openinterest", "oi"],
        "iv": ["implied_volatility", "iv", "impl_volatility"],
        "delta": ["delta"],
        "gamma": ["gamma"],
        "theta": ["theta"],
        "vega": ["vega"],
        "rho": ["rho"],
        "underlying_close": ["underlying_last", "stock_price", "underlying_close", "spot_price"],
    }

    for target, candidates in field_candidates.items():
        for candidate in candidates:
            if candidate in cols_lower:
                col_map[cols_lower[candidate]] = target
                break

    df = df.rename(columns=col_map)

    # Ensure dates are datetime
    if "trade_date" in df.columns:
        df["trade_date"] = pd.to_datetime(df["trade_date"])
    if "expiry" in df.columns:
        df["expiry"] = pd.to_datetime(df["expiry"])

    # Normalize option type
    if "option_type" in df.columns:
        df["option_type"] = df["option_type"].astype(str).str.strip().str.upper().str[0]
        df = df[df["option_type"].isin(["C", "P"])]

    # Numeric columns
    for col in ["strike", "bid", "ask", "last_price", "volume", "open_interest",
                 "iv", "delta", "gamma", "theta", "vega", "rho", "underlying_close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["bid", "ask", "last_price", "volume", "open_interest",
                 "gamma", "theta", "vega", "rho"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Derived fields
    if "trade_date" in df.columns and "expiry" in df.columns:
        df["dte"] = (df["expiry"] - df["trade_date"]).dt.days
    if "bid" in df.columns and "ask" in df.columns:
        df["mid"] = (df["bid"] + df["ask"]) / 2
        df["bid_ask_pct"] = (df["ask"] - df["bid"]) / df["mid"].replace(0, float("nan"))

    # Tag source
    df["source"] = "dubach"

    # Fill missing columns
    for col in ["rho", "underlying_close"]:
        if col not in df.columns:
            df[col] = 0.0

    return df


def ingest(year: int | None = None, force_download: bool = False) -> None:
    """Ingest Dubach Parquet and write yearly unified Parquet files.

    Args:
        year: If specified, only output that year.
        force_download: Re-download the source file.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    dubach_path = download_dubach(force=force_download)

    logger.info(f"Reading Dubach Parquet: {dubach_path}")
    df = pd.read_parquet(dubach_path)
    logger.info(f"Loaded {len(df):,} rows, columns: {list(df.columns)}")

    df = _normalize_dubach(df)

    # Drop invalid
    required = {"trade_date", "expiry", "strike", "option_type"}
    missing = required - set(df.columns)
    if missing:
        logger.error(f"Missing required columns after normalization: {missing}")
        return

    df = df.dropna(subset=["strike", "trade_date", "expiry"])
    if "dte" in df.columns:
        df = df[df["dte"] >= 0]

    # Group by year
    df["year"] = df["trade_date"].dt.year
    years = sorted(df["year"].unique())

    if year is not None:
        years = [y for y in years if y == year]

    schema_cols = [
        "trade_date", "expiry", "dte", "strike", "option_type",
        "bid", "ask", "mid", "last_price", "volume", "open_interest",
        "iv", "delta", "gamma", "theta", "vega", "rho",
        "underlying_close", "bid_ask_pct", "source",
    ]

    for yr in years:
        yr_df = df[df["year"] == yr].drop(columns=["year"])

        for col in schema_cols:
            if col not in yr_df.columns:
                yr_df[col] = 0.0 if col not in ("trade_date", "expiry", "option_type", "source") else ""

        yr_df = yr_df[schema_cols]

        # For years that overlap with optionsDX, save with dubach suffix
        # to avoid overwriting. The validation script uses both.
        out_path = PROCESSED_DIR / f"spy_options_{yr}_dubach.parquet"
        yr_df.to_parquet(out_path, index=False, engine="pyarrow")
        logger.info(f"Wrote {len(yr_df):,} rows to {out_path}")


@click.command()
@click.option("--year", type=int, default=None, help="Output only this year")
@click.option("--download", is_flag=True, help="Force re-download source file")
@click.option("--debug", is_flag=True, help="Enable DEBUG logging")
def main(year: int | None, download: bool, debug: bool) -> None:
    """Ingest Dubach Parquet into unified yearly Parquet files."""
    setup_logging("DEBUG" if debug else None)
    ingest(year, force_download=download)


if __name__ == "__main__":
    main()
