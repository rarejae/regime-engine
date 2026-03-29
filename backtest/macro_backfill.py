"""Historical macro data backfill — two-stage pipeline.

STAGE A — download_raw_indicators():
    Pulls raw indicator values from FRED for 2005–2023+.
    Saves to data/macro/raw_indicators.parquet.
    This file is cached permanently — subsequent runs skip all API calls.
    Use --refresh to force a full re-download.

STAGE B — compute_condition_vectors():
    Reads raw_indicators.parquet from DISK (zero API calls).
    Applies z-score normalization using CURRENT engine config weights.
    Builds a ConditionVector for each trading day.
    Saves to data/macro/condition_vectors.parquet.
    This file is ALWAYS recomputed because it depends on engine weights.

Usage:
    python -m backtest.macro_backfill                   # download if needed + compute CVs
    python -m backtest.macro_backfill --refresh          # force re-download from FRED
    python -m backtest.macro_backfill --start 2005-01-01 --end 2024-12-31
"""

from __future__ import annotations

import os
import pickle
from datetime import date
from pathlib import Path
from typing import Optional

import click
import numpy as np
import pandas as pd
from loguru import logger

from engine.condition_vector import ConditionVector

MACRO_DIR = Path(__file__).parent.parent / "data" / "macro"
RAW_INDICATORS_PATH = MACRO_DIR / "raw_indicators.parquet"
CONDITION_VECTORS_PATH = MACRO_DIR / "condition_vectors.pickle"

# Key FRED series for axis scoring.
# Maps our indicator IDs to FRED series IDs.
FRED_SERIES: dict[str, str | None] = {
    # Growth
    "ISM_MFG": "MANEMP",
    "CLAIMS_4WK": "IC4WSA",
    "GDP_NOWCAST": None,  # Not on FRED — filled separately
    "HOUSING_STARTS": "HOUST",
    "NFP_3MA": "PAYEMS",
    # Inflation
    "CPI_YOY": "CPIAUCSL",
    "PCE_CORE_YOY": "PCEPILFE",
    "BREAKEVEN_5Y": "T5YIE",
    "WTI": "DCOILWTICO",
    # Liquidity
    "FFR_LEVEL": "FEDFUNDS",
    "FED_BS_DELTA_3M": "WALCL",
    "NFCI": "NFCI",
    # Rates
    "UST_2Y": "DGS2",
    "UST_10Y": "DGS10",
    "UST_2S10S": None,  # Derived: DGS10 - DGS2
    "REAL_YIELD_10Y": "DFII10",
    "MORT_30Y": "MORTGAGE30US",
    # Volatility (from FRED where available)
    "MOVE_INDEX": "MOVE",
    # Credit
    "HY_OAS": "BAMLH0A0HYM2",
    "IG_OAS": "BAMLC0A4CBBB",
    # FX
    "DXY": "DTWEXBGS",
}

# Yahoo-sourced indicators (fetched via yfinance for backfill)
YAHOO_SERIES: dict[str, str] = {
    "VIX": "^VIX",
    "VIX3M": "^VIX3M",   # 3-month VIX for term structure
    "VVIX": "^VVIX",
    "SKEW": "^SKEW",     # CBOE SKEW index
    "GOLD": "GC=F",
    "COPPER": "HG=F",
    "SPX_PRICE": "^GSPC",  # Raw price — transformed to 3-month return below
}


def _fetch_fred_series(series_id: str, start: str, end: str | None) -> pd.Series:
    """Fetch a single FRED series."""
    from fredapi import Fred

    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        raise ValueError("FRED_API_KEY environment variable required")

    fred = Fred(api_key=api_key)
    return fred.get_series(series_id, observation_start=start, observation_end=end).dropna()


def _fetch_yahoo_series(ticker: str, start: str, end: str | None) -> pd.Series:
    """Fetch a single Yahoo Finance series."""
    import yfinance as yf

    end_str = end or pd.Timestamp.now().strftime("%Y-%m-%d")
    data = yf.download(ticker, start=start, end=end_str, progress=False, auto_adjust=True)
    if data.empty:
        return pd.Series(dtype=float)
    # Handle multi-level columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data["Close"].dropna()


# ── STAGE A: Raw Data Download ────────────────────────────────────────────────


def download_raw_indicators(
    start_date: str = "2005-01-01",
    end_date: str | None = None,
    force_refresh: bool = False,
) -> Path:
    """Stage A: Download and cache raw indicator values. Returns path to parquet file.

    If raw_indicators.parquet already exists AND covers the requested date range,
    all API calls are skipped.

    Args:
        start_date: Start of history pull.
        end_date: End of history pull (defaults to today).
        force_refresh: If True, re-download everything.

    Returns:
        Path to the cached raw_indicators.parquet file.
    """
    MACRO_DIR.mkdir(parents=True, exist_ok=True)

    # Check if cache already covers the requested range
    if RAW_INDICATORS_PATH.exists() and not force_refresh:
        existing = pd.read_parquet(RAW_INDICATORS_PATH)
        existing_start = existing.index.min()
        existing_end = existing.index.max()
        req_start = pd.Timestamp(start_date)
        req_end = pd.Timestamp(end_date) if end_date else pd.Timestamp.now()

        if existing_start <= req_start and existing_end >= req_end - pd.Timedelta(days=7):
            logger.info(
                f"Using cached raw indicators ({existing_start.date()} to {existing_end.date()}). "
                f"Skipping download. Use --refresh to force re-download."
            )
            return RAW_INDICATORS_PATH

        # Existing file doesn't cover full range — do incremental fetch
        logger.info(
            f"Cached data covers {existing_start.date()} to {existing_end.date()}. "
            f"Fetching missing dates..."
        )
        # For simplicity, re-download the full range (incremental merge is complex)
        # but only when the range is actually different
        if existing_end >= req_end - pd.Timedelta(days=7):
            logger.info("Cache is up to date. Skipping download.")
            return RAW_INDICATORS_PATH

    logger.info(f"Downloading raw indicators from FRED and Yahoo ({start_date} to {end_date or 'today'})...")

    all_series: dict[str, pd.Series] = {}
    failed: list[str] = []

    # Fetch FRED series
    for ind_id, fred_id in FRED_SERIES.items():
        if fred_id is None:
            continue
        try:
            data = _fetch_fred_series(fred_id, start_date, end_date)
            all_series[ind_id] = data
            logger.info(f"  FRED {ind_id} ({fred_id}): {len(data)} observations")
        except Exception as e:
            logger.warning(f"  FRED {ind_id} ({fred_id}): FAILED — {e}")
            failed.append(ind_id)

    # Fetch Yahoo series
    for ind_id, ticker in YAHOO_SERIES.items():
        try:
            data = _fetch_yahoo_series(ticker, start_date, end_date)
            if not data.empty:
                all_series[ind_id] = data
                logger.info(f"  Yahoo {ind_id} ({ticker}): {len(data)} observations")
            else:
                logger.warning(f"  Yahoo {ind_id} ({ticker}): empty result")
                failed.append(ind_id)
        except Exception as e:
            logger.warning(f"  Yahoo {ind_id} ({ticker}): FAILED — {e}")
            failed.append(ind_id)

    if not all_series:
        raise RuntimeError("No indicator data fetched — check API keys and network")

    # Combine into single DataFrame, forward-fill for different frequencies
    df = pd.DataFrame(all_series)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.ffill()

    # Derived series
    if "UST_10Y" in df.columns and "UST_2Y" in df.columns:
        df["UST_2S10S"] = df["UST_10Y"] - df["UST_2Y"]
    if "HY_OAS" in df.columns and "IG_OAS" in df.columns:
        df["HY_IG_DIFF"] = df["HY_OAS"] - df["IG_OAS"]

    # CPI YoY computation (CPIAUCSL is index level → YoY pct change)
    if "CPI_YOY" in df.columns:
        df["CPI_YOY"] = df["CPI_YOY"].pct_change(periods=252) * 100
    if "PCE_CORE_YOY" in df.columns:
        df["PCE_CORE_YOY"] = df["PCE_CORE_YOY"].pct_change(periods=252) * 100

    # SPX 3-month return (transform raw price to rolling 63-day return)
    if "SPX_PRICE" in df.columns:
        df["SPX_RTN_3M"] = df["SPX_PRICE"].pct_change(periods=63) * 100  # percent
        df = df.drop(columns=["SPX_PRICE"])
        logger.info("  Derived SPX_RTN_3M: 63-day rolling return from SPX price")

    # VIX term structure: VIX3M - VIX (positive = contango, negative = backwardation)
    if "VIX3M" in df.columns and "VIX" in df.columns:
        df["VIX_TERM"] = df["VIX3M"] - df["VIX"]
        logger.info("  Derived VIX_TERM: VIX3M - VIX")

    # NFP: transform from employment LEVEL to 3-month average of monthly change
    if "NFP_3MA" in df.columns:
        # PAYEMS is monthly employment level; compute month-over-month change
        # then 3-month rolling mean. Forward-fill means daily values are constant
        # within a month, so pct_change(21) gives approx monthly change.
        nfp_chg = df["NFP_3MA"].pct_change(periods=21) * 100  # monthly % change
        df["NFP_3MA"] = nfp_chg.rolling(63).mean()  # 3-month average
        logger.info("  Derived NFP_3MA: 3-month rolling average of monthly employment change")

    # Save to Parquet
    df.to_parquet(RAW_INDICATORS_PATH, engine="pyarrow")

    logger.info(
        f"Saved raw indicators to {RAW_INDICATORS_PATH} "
        f"({len(df):,} rows x {len(df.columns)} columns)"
    )
    if failed:
        logger.warning(f"Failed indicators (will be NULL): {failed}")

    return RAW_INDICATORS_PATH


# ── STAGE B: Condition Vector Computation ─────────────────────────────────────


# Axis mappings: indicator → (axis, polarity, weight)
# Weights are used for proportional redistribution when indicators are missing.
AXIS_MAP: dict[str, tuple[str, int, float]] = {
    "ISM_MFG": ("GROWTH", 1, 0.12),         # Monthly, lagging — reduced weight
    "CLAIMS_4WK": ("GROWTH", -1, 0.23),     # Weekly — responsive
    "HOUSING_STARTS": ("GROWTH", 1, 0.10),  # Monthly, lagging — reduced weight
    "NFP_3MA": ("GROWTH", 1, 0.15),         # Monthly change rate — moderate lag
    "SPX_RTN_3M": ("GROWTH", 1, 0.40),      # Daily — most responsive growth signal
    "CPI_YOY": ("INFLATION", 1, 0.30),
    "PCE_CORE_YOY": ("INFLATION", 1, 0.30),
    "BREAKEVEN_5Y": ("INFLATION", 1, 0.25),
    "WTI": ("INFLATION", 1, 0.15),
    "FFR_LEVEL": ("LIQUIDITY", -1, 0.25),
    "FED_BS_DELTA_3M": ("LIQUIDITY", 1, 0.20),
    "UST_2S10S": ("LIQUIDITY", 1, 0.20),
    "REAL_YIELD_10Y": ("LIQUIDITY", -1, 0.20),
    "NFCI": ("LIQUIDITY", -1, 0.15),
    "VIX": ("RISK", -1, 0.30),
    "VVIX": ("RISK", -1, 0.10),
    "HY_OAS": ("RISK", -1, 0.25),
    "IG_OAS": ("RISK", -1, 0.10),
    "MOVE_INDEX": ("RISK", -1, 0.15),
    "GOLD": ("RISK", -1, 0.10),
}


def compute_condition_vectors(
    raw_indicators_path: Path | str | None = None,
    lookback: int = 252 * 5,
    trading_days: list[date] | None = None,
) -> dict[date, ConditionVector]:
    """Stage B: Compute ConditionVectors from cached raw data using current config.

    Reads raw_indicators.parquet from disk (ZERO API calls).
    Applies rolling z-score normalization and computes axis scores.
    Handles missing indicators by redistributing weights proportionally.

    Args:
        raw_indicators_path: Path to raw_indicators.parquet. Defaults to standard location.
        lookback: Rolling window for z-score computation (default 5 years).
        trading_days: If provided, only compute for these dates.

    Returns:
        Dict mapping date → ConditionVector.
    """
    path = Path(raw_indicators_path) if raw_indicators_path else RAW_INDICATORS_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Raw indicators not found at {path}. "
            f"Run: python -m backtest.macro_backfill"
        )

    logger.info(f"Reading cached raw indicators from {path}")
    macro_df = pd.read_parquet(path)
    macro_df.index = pd.to_datetime(macro_df.index)
    logger.info(f"  {len(macro_df):,} rows x {len(macro_df.columns)} columns")

    # Compute rolling z-scores
    rolling_mean = macro_df.rolling(window=lookback, min_periods=60).mean()
    rolling_std = macro_df.rolling(window=lookback, min_periods=60).std()
    z_scores = (macro_df - rolling_mean) / rolling_std.replace(0, np.nan)
    z_scores = z_scores.clip(-3, 3)

    # Determine which dates to process
    dates_to_process = macro_df.index
    if trading_days is not None:
        trading_set = set(pd.Timestamp(d) for d in trading_days)
        dates_to_process = macro_df.index[macro_df.index.isin(trading_set)]

    logger.info(f"Computing ConditionVectors for {len(dates_to_process)} dates...")

    condition_vectors: dict[date, ConditionVector] = {}
    prev_scores: dict[str, float] | None = None
    missing_logged: set[str] = set()

    for dt in dates_to_process:
        row_z = z_scores.loc[dt]
        row_raw = macro_df.loc[dt]

        # Compute axis scores with proportional weight redistribution
        axis_raw: dict[str, float] = {"GROWTH": 0.0, "INFLATION": 0.0, "LIQUIDITY": 0.0, "RISK": 0.0}
        axis_total_weight: dict[str, float] = {"GROWTH": 0.0, "INFLATION": 0.0, "LIQUIDITY": 0.0, "RISK": 0.0}
        axis_available_weight: dict[str, float] = {"GROWTH": 0.0, "INFLATION": 0.0, "LIQUIDITY": 0.0, "RISK": 0.0}

        for ind_id, (axis, polarity, weight) in AXIS_MAP.items():
            axis_total_weight[axis] += weight
            if ind_id in row_z.index and not np.isnan(row_z[ind_id]):
                axis_available_weight[axis] += weight
                axis_raw[axis] += row_z[ind_id] * polarity * weight
            elif ind_id not in missing_logged:
                logger.debug(f"  Missing indicator {ind_id} on {dt.date()} — redistributing weight")
                missing_logged.add(ind_id)

        # Normalize: redistribute missing weight proportionally
        scores: dict[str, float] = {}
        for axis in ["GROWTH", "INFLATION", "LIQUIDITY", "RISK"]:
            if axis_available_weight[axis] > 0:
                # Normalize by available weight (equivalent to proportional redistribution)
                raw = axis_raw[axis] / axis_available_weight[axis]
                scores[axis] = float(np.tanh(raw) * 100)
            else:
                scores[axis] = 0.0

        # Transition deltas
        dt_date = dt.date() if hasattr(dt, "date") else dt
        if prev_scores is not None:
            growth_delta = scores["GROWTH"] - prev_scores["GROWTH"]
            inflation_delta = scores["INFLATION"] - prev_scores["INFLATION"]
            liquidity_delta = scores["LIQUIDITY"] - prev_scores["LIQUIDITY"]
            risk_delta = scores["RISK"] - prev_scores["RISK"]
        else:
            growth_delta = inflation_delta = liquidity_delta = risk_delta = 0.0

        # Raw values for vol surface and credit
        def _raw(col: str, default: float) -> float:
            if col in row_raw.index and not np.isnan(row_raw[col]):
                return float(row_raw[col])
            return default

        vix = _raw("VIX", 20.0)
        vix_term = _raw("VIX_TERM", 0.0)    # VIX3M - VIX, from derived series
        vvix = _raw("VVIX", 100.0)
        skew = _raw("SKEW", 130.0)           # CBOE SKEW index
        hy_oas = _raw("HY_OAS", 400.0)
        ig_oas = _raw("IG_OAS", 100.0)
        hy_ig_diff = _raw("HY_IG_DIFF", 300.0)
        move = _raw("MOVE_INDEX", 100.0)

        # Modifier flags
        liq = scores["LIQUIDITY"]
        rsk = scores["RISK"]

        # Regime label (display only)
        g, i_score = scores["GROWTH"], scores["INFLATION"]
        if g > 20 and i_score < 20:
            label = "GOLDILOCKS"
        elif g > 20 and i_score >= 20:
            label = "REFLATION"
        elif g <= -20 and i_score >= 20:
            label = "STAGFLATION"
        elif g <= -20 and i_score < -20:
            label = "DEFLATION"
        else:
            label = "MIXED"

        conviction = (abs(g) + abs(i_score)) / 2

        cv = ConditionVector(
            growth=scores["GROWTH"],
            inflation=scores["INFLATION"],
            liquidity=scores["LIQUIDITY"],
            risk=scores["RISK"],
            growth_delta=growth_delta,
            inflation_delta=inflation_delta,
            liquidity_delta=liquidity_delta,
            risk_delta=risk_delta,
            vix=vix,
            vix_term=vix_term,
            vvix=vvix,
            skew=skew,
            move_index=move,
            hy_oas=hy_oas,
            ig_oas=ig_oas,
            hy_ig_diff=hy_ig_diff,
            tight_liq=(liq < -30),
            loose_liq=(liq > 30),
            high_vol=(rsk < -30),
            low_vol=(rsk > 30),
            vol_crush_imminent=(vix_term < -3.0 and vvix > 120),
            regime_label=label,
            conviction=conviction,
        )
        condition_vectors[dt_date] = cv
        prev_scores = scores

    logger.info(f"Computed {len(condition_vectors)} daily ConditionVectors")

    # Save to disk (pickle for ConditionVector objects)
    MACRO_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONDITION_VECTORS_PATH, "wb") as f:
        pickle.dump(condition_vectors, f)
    logger.info(f"Saved condition vectors to {CONDITION_VECTORS_PATH}")

    return condition_vectors


def load_cached_condition_vectors() -> dict[date, ConditionVector] | None:
    """Load previously computed condition vectors from disk.

    Returns:
        Dict mapping date → ConditionVector, or None if not found.
    """
    if not CONDITION_VECTORS_PATH.exists():
        return None
    with open(CONDITION_VECTORS_PATH, "rb") as f:
        return pickle.load(f)


# ── CLI ───────────────────────────────────────────────────────────────────────


@click.command()
@click.option("--start", default="2005-01-01", help="History start date")
@click.option("--end", default=None, help="History end date (default: today)")
@click.option("--refresh", is_flag=True, help="Force re-download from FRED (ignore cache)")
@click.option("--debug", is_flag=True, help="Enable DEBUG logging")
def main(start: str, end: str | None, refresh: bool, debug: bool) -> None:
    """Download raw macro data (if needed) and compute condition vectors."""
    from utils.logging import setup_logging
    setup_logging("DEBUG" if debug else None)

    from dotenv import load_dotenv
    load_dotenv()

    # Stage A: Download (cached)
    raw_path = download_raw_indicators(start, end, force_refresh=refresh)

    # Stage B: Compute CVs (always recomputes)
    logger.info("Recomputing condition vectors with current engine config...")
    compute_condition_vectors(raw_path)


if __name__ == "__main__":
    main()
