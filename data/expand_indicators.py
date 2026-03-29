"""Fetch new indicator series and merge into raw_indicators.parquet.

Idempotent: skips columns that already exist with >50% coverage.
Use --force to re-fetch everything.

Usage:
    python data/expand_indicators.py
    python data/expand_indicators.py --force
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PARQUET_PATH = Path("data/macro/raw_indicators.parquet")
BACKUP_PATH = Path("data/macro/raw_indicators_backup.parquet")

FRED_SERIES = {
    "ISM_NEW_ORDERS": {"series_id": "NAPMNOI", "frequency": "monthly"},
    "ISM_INVENTORIES": {"series_id": "NAPMII", "frequency": "monthly"},
    "STLFSI": {"series_id": "STLFSI2", "frequency": "weekly"},
    "M2": {"series_id": "M2SL", "frequency": "monthly"},
    "UST_3M": {"series_id": "DGS3MO", "frequency": "daily"},
}

YFINANCE_TICKERS = {
    "SPY_CLOSE": {"ticker": "SPY", "column": "Close"},
    "QQQ_CLOSE": {"ticker": "QQQ", "column": "Close"},
}


def _fetch_fred(series_id: str, start: str, end: str) -> pd.Series:
    try:
        from fredapi import Fred
        key = os.environ.get("FRED_API_KEY")
        if key:
            fred = Fred(api_key=key)
            data = fred.get_series(series_id, observation_start=start, observation_end=end)
            data.index = pd.to_datetime(data.index)
            logger.info(f"  FRED ({series_id}): {len(data)} obs via fredapi")
            return data.dropna()
    except Exception as e:
        logger.warning(f"  fredapi failed for {series_id}: {e}")

    try:
        url = (f"https://fred.stlouisfed.org/graph/fredgraph.csv?"
               f"id={series_id}&cosd={start}&coed={end}")
        df = pd.read_csv(url, index_col=0, parse_dates=True)
        s = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
        logger.info(f"  FRED ({series_id}): {len(s)} obs via CSV")
        return s
    except Exception as e:
        logger.error(f"  All FRED methods failed for {series_id}: {e}")
        return pd.Series(dtype=float)


def _fetch_yfinance(ticker: str, start: str, end: str, col: str = "Close") -> pd.Series:
    try:
        import yfinance as yf
        data = yf.download(ticker, start=start, end=end, progress=False)
        if data is None or data.empty:
            return pd.Series(dtype=float)
        prices = data[col]
        if hasattr(prices, "columns"):
            prices = prices.iloc[:, 0]
        prices.index = pd.to_datetime(prices.index).tz_localize(None)
        logger.info(f"  yfinance ({ticker}): {len(prices)} obs")
        return prices
    except Exception as e:
        logger.error(f"  yfinance failed for {ticker}: {e}")
        return pd.Series(dtype=float)


def expand_indicators(force: bool = False) -> None:
    if not PARQUET_PATH.exists():
        logger.error(f"Not found: {PARQUET_PATH}")
        return

    raw = pd.read_parquet(PARQUET_PATH)
    raw.index = pd.to_datetime(raw.index)
    raw = raw.sort_index()
    logger.info(f"Loaded: {raw.shape[0]} rows x {raw.shape[1]} cols")

    if not BACKUP_PATH.exists():
        raw.to_parquet(BACKUP_PATH)
        logger.info(f"Backup saved to {BACKUP_PATH}")

    start = str(raw.index.min().date())
    end = str(raw.index.max().date())
    added = 0

    # FRED
    for col_name, info in FRED_SERIES.items():
        if col_name in raw.columns and not force and raw[col_name].notna().mean() > 0.5:
            logger.info(f"  {col_name}: exists ({raw[col_name].notna().mean():.0%}), skip")
            continue
        logger.info(f"Fetching {col_name}...")
        s = _fetch_fred(info["series_id"], start, end)
        if s.empty:
            continue
        raw[col_name] = s.reindex(raw.index, method="ffill")
        logger.info(f"  {col_name}: {raw[col_name].notna().mean():.1%} coverage")
        added += 1

    # yfinance
    for col_name, info in YFINANCE_TICKERS.items():
        if col_name in raw.columns and not force and raw[col_name].notna().mean() > 0.5:
            logger.info(f"  {col_name}: exists ({raw[col_name].notna().mean():.0%}), skip")
            continue
        logger.info(f"Fetching {col_name}...")
        s = _fetch_yfinance(info["ticker"], start, end, info["column"])
        if s.empty:
            continue
        raw[col_name] = s.reindex(raw.index, method="ffill")
        logger.info(f"  {col_name}: {raw[col_name].notna().mean():.1%} coverage")
        added += 1

    # Derived columns
    if "ISM_NEW_ORDERS" in raw.columns and "ISM_INVENTORIES" in raw.columns:
        if "ISM_NO_INV_SPREAD" not in raw.columns or force:
            raw["ISM_NO_INV_SPREAD"] = raw["ISM_NEW_ORDERS"] - raw["ISM_INVENTORIES"]
            logger.info(f"  ISM_NO_INV_SPREAD: {raw['ISM_NO_INV_SPREAD'].notna().mean():.1%}")
            added += 1

    if "UST_3M" in raw.columns and "UST_10Y" in raw.columns:
        if "UST_10Y_3M" not in raw.columns or force:
            raw["UST_10Y_3M"] = raw["UST_10Y"] - raw["UST_3M"]
            logger.info(f"  UST_10Y_3M: {raw['UST_10Y_3M'].notna().mean():.1%}")
            added += 1

    if "COPPER" in raw.columns and "GOLD" in raw.columns:
        if "COPPER_GOLD_RATIO" not in raw.columns or force:
            raw["COPPER_GOLD_RATIO"] = raw["COPPER"] / raw["GOLD"]
            logger.info(f"  COPPER_GOLD_RATIO: {raw['COPPER_GOLD_RATIO'].notna().mean():.1%}")
            added += 1

    if "QQQ_CLOSE" in raw.columns and "SPY_CLOSE" in raw.columns:
        if "QQQ_SPY_RATIO" not in raw.columns or force:
            raw["QQQ_SPY_RATIO"] = raw["QQQ_CLOSE"] / raw["SPY_CLOSE"]
            logger.info(f"  QQQ_SPY_RATIO: {raw['QQQ_SPY_RATIO'].notna().mean():.1%}")
            added += 1

    if "M2" in raw.columns:
        if "M2_YOY" not in raw.columns or force:
            raw["M2_YOY"] = (raw["M2"] / raw["M2"].shift(252) - 1) * 100
            logger.info(f"  M2_YOY: {raw['M2_YOY'].notna().mean():.1%}")
            added += 1

    if "UST_10Y" in raw.columns:
        if "MOVE_PROXY" not in raw.columns or force:
            raw["MOVE_PROXY"] = raw["UST_10Y"].diff().rolling(21, min_periods=10).std() * np.sqrt(252) * 100
            logger.info(f"  MOVE_PROXY: {raw['MOVE_PROXY'].notna().mean():.1%}")
            added += 1

    raw = raw.sort_index()
    raw.to_parquet(PARQUET_PATH)
    logger.info(f"\nSaved: {raw.shape[0]} rows x {raw.shape[1]} cols ({added} new)")

    # Coverage report
    new_cols = ["ISM_NEW_ORDERS", "ISM_INVENTORIES", "ISM_NO_INV_SPREAD",
                "STLFSI", "M2", "M2_YOY", "UST_3M", "UST_10Y_3M",
                "SPY_CLOSE", "QQQ_CLOSE", "QQQ_SPY_RATIO",
                "COPPER_GOLD_RATIO", "MOVE_PROXY"]
    logger.info("\nNew column coverage:")
    for c in new_cols:
        if c in raw.columns:
            pct = raw[c].notna().mean() * 100
            fv = raw[c].first_valid_index()
            logger.info(f"  {c:25s}: {pct:5.1f}%  (from {fv})")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    expand_indicators(force=args.force)
