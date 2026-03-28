"""Historical macro data backfill from FRED.

Pulls multi-year history for all macro indicators used by the regime engine,
computes daily z-scores and axis scores, and generates a daily ConditionVector
for every trading day in the backtest window.

Output: data/macro/macro_daily.parquet

Usage:
    python -m backtest.macro_backfill
    python -m backtest.macro_backfill --start 2010-01-01 --end 2023-12-31
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import click
import numpy as np
import pandas as pd
from loguru import logger

from engine.condition_vector import ConditionVector

MACRO_DIR = Path(__file__).parent.parent / "data" / "macro"
FRED_INDICATORS_FILE = MACRO_DIR / "macro_daily.parquet"


# Key FRED series for axis scoring
# Maps our indicator IDs to FRED series IDs
FRED_SERIES = {
    # Growth
    "ISM_MFG": "MANEMP",
    "CLAIMS_4WK": "IC4WSA",
    "GDP_NOWCAST": None,  # Not on FRED — filled separately
    # Inflation
    "CPI_YOY": "CPIAUCSL",
    "PCE_CORE_YOY": "PCEPILFE",
    "BREAKEVEN_5Y": "T5YIE",
    # Liquidity
    "FFR_LEVEL": "FEDFUNDS",
    "FED_BS_DELTA_3M": "WALCL",
    # Rates
    "UST_2Y": "DGS2",
    "UST_10Y": "DGS10",
    "UST_2S10S": None,  # Derived: DGS10 - DGS2
    "REAL_YIELD_10Y": "DFII10",
    # Volatility (from FRED where available)
    "MOVE_INDEX": "MOVE",
    # Credit
    "HY_OAS": "BAMLH0A0HYM2",
    "IG_OAS": "BAMLC0A4CBBB",
}


def fetch_fred_history(
    series_id: str,
    start_date: str = "2005-01-01",
    end_date: str | None = None,
) -> pd.Series:
    """Fetch historical daily data from FRED.

    Args:
        series_id: FRED series identifier.
        start_date: Start date string.
        end_date: End date string (defaults to today).

    Returns:
        pandas Series indexed by date.
    """
    try:
        from fredapi import Fred
    except ImportError:
        raise ImportError("fredapi required: pip install fredapi")

    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        raise ValueError("FRED_API_KEY environment variable required")

    fred = Fred(api_key=api_key)
    data = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
    return data.dropna()


def build_macro_history(
    start_date: str = "2005-01-01",
    end_date: str | None = None,
) -> pd.DataFrame:
    """Fetch all FRED indicators and build a daily DataFrame.

    Fetches 5+ years of history per indicator to enable z-score computation.
    Forward-fills missing values (FRED data has different frequencies).

    Args:
        start_date: Start of history pull (should be ~5 years before backtest start).
        end_date: End of history.

    Returns:
        DataFrame with date index and one column per indicator.
    """
    all_series = {}

    for ind_id, fred_id in FRED_SERIES.items():
        if fred_id is None:
            continue
        try:
            data = fetch_fred_history(fred_id, start_date, end_date)
            all_series[ind_id] = data
            logger.info(f"Fetched {ind_id} ({fred_id}): {len(data)} observations")
        except Exception as e:
            logger.warning(f"Failed to fetch {ind_id} ({fred_id}): {e}")

    if not all_series:
        raise RuntimeError("No FRED data fetched — check API key and network")

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

    # CPI YoY computation (CPIAUCSL is index level, need YoY pct change)
    if "CPI_YOY" in df.columns:
        df["CPI_YOY"] = df["CPI_YOY"].pct_change(periods=252) * 100  # ~252 trading days = 1 year

    # PCE Core YoY
    if "PCE_CORE_YOY" in df.columns:
        df["PCE_CORE_YOY"] = df["PCE_CORE_YOY"].pct_change(periods=252) * 100

    logger.info(f"Macro history: {len(df)} days, {len(df.columns)} indicators")
    return df


def compute_daily_condition_vectors(
    macro_df: pd.DataFrame,
    lookback: int = 252 * 5,
    trading_days: list[date] | None = None,
) -> dict[date, ConditionVector]:
    """Compute a ConditionVector for each trading day.

    Uses rolling z-scores over `lookback` days for normalization,
    then maps to axis scores using simplified weights.

    Args:
        macro_df: DataFrame from build_macro_history().
        lookback: Rolling window for z-score computation (default 5 years).
        trading_days: If provided, only compute for these dates.

    Returns:
        Dict mapping date → ConditionVector.
    """
    # Simplified axis mappings (indicator → axis, polarity)
    axis_map = {
        "ISM_MFG": ("GROWTH", 1),
        "CLAIMS_4WK": ("GROWTH", -1),
        "CPI_YOY": ("INFLATION", 1),
        "PCE_CORE_YOY": ("INFLATION", 1),
        "BREAKEVEN_5Y": ("INFLATION", 1),
        "FFR_LEVEL": ("LIQUIDITY", -1),
        "FED_BS_DELTA_3M": ("LIQUIDITY", 1),
        "UST_2S10S": ("LIQUIDITY", 1),
        "REAL_YIELD_10Y": ("LIQUIDITY", -1),
        "HY_OAS": ("RISK", -1),
        "IG_OAS": ("RISK", -1),
        "MOVE_INDEX": ("RISK", -1),
    }

    # Compute rolling z-scores
    rolling_mean = macro_df.rolling(window=lookback, min_periods=60).mean()
    rolling_std = macro_df.rolling(window=lookback, min_periods=60).std()
    z_scores = (macro_df - rolling_mean) / rolling_std.replace(0, np.nan)
    z_scores = z_scores.clip(-3, 3)

    # Compute axis scores
    condition_vectors: dict[date, ConditionVector] = {}

    dates_to_process = macro_df.index
    if trading_days is not None:
        trading_set = set(pd.Timestamp(d) for d in trading_days)
        dates_to_process = macro_df.index[macro_df.index.isin(trading_set)]

    prev_scores = None

    for dt in dates_to_process:
        row_z = z_scores.loc[dt]
        row_raw = macro_df.loc[dt]

        # Compute axis scores using tanh bounding
        axis_raw = {"GROWTH": 0.0, "INFLATION": 0.0, "LIQUIDITY": 0.0, "RISK": 0.0}
        axis_counts = {"GROWTH": 0, "INFLATION": 0, "LIQUIDITY": 0, "RISK": 0}

        for ind_id, (axis, polarity) in axis_map.items():
            if ind_id in row_z.index and not np.isnan(row_z[ind_id]):
                axis_raw[axis] += row_z[ind_id] * polarity
                axis_counts[axis] += 1

        # Normalize and apply tanh
        scores = {}
        for axis in ["GROWTH", "INFLATION", "LIQUIDITY", "RISK"]:
            if axis_counts[axis] > 0:
                raw = axis_raw[axis] / axis_counts[axis]
                scores[axis] = float(np.tanh(raw) * 100)
            else:
                scores[axis] = 0.0

        # Transition deltas (30-day lookback)
        dt_date = dt.date() if hasattr(dt, 'date') else dt
        if prev_scores is not None:
            growth_delta = scores["GROWTH"] - prev_scores["GROWTH"]
            inflation_delta = scores["INFLATION"] - prev_scores["INFLATION"]
            liquidity_delta = scores["LIQUIDITY"] - prev_scores["LIQUIDITY"]
            risk_delta = scores["RISK"] - prev_scores["RISK"]
        else:
            growth_delta = inflation_delta = liquidity_delta = risk_delta = 0.0

        # Raw values for vol surface and credit
        vix = 20.0  # VIX not in FRED; use placeholder or load separately
        hy_oas = float(row_raw.get("HY_OAS", 400)) if "HY_OAS" in row_raw.index else 400.0
        ig_oas = float(row_raw.get("IG_OAS", 100)) if "IG_OAS" in row_raw.index else 100.0
        hy_ig_diff = float(row_raw.get("HY_IG_DIFF", 300)) if "HY_IG_DIFF" in row_raw.index else 300.0
        move = float(row_raw.get("MOVE_INDEX", 100)) if "MOVE_INDEX" in row_raw.index else 100.0

        # Modifier flags
        liq = scores["LIQUIDITY"]
        rsk = scores["RISK"]

        # Regime label (display only)
        g, i = scores["GROWTH"], scores["INFLATION"]
        if g > 20 and i < 20:
            label = "GOLDILOCKS"
        elif g > 20 and i >= 20:
            label = "REFLATION"
        elif g <= -20 and i >= 20:
            label = "STAGFLATION"
        elif g <= -20 and i < -20:
            label = "DEFLATION"
        else:
            label = "MIXED"

        conviction = (abs(g) + abs(i)) / 2

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
            vix_term=0.0,
            vvix=100.0,
            skew=130.0,
            move_index=move,
            hy_oas=hy_oas,
            ig_oas=ig_oas,
            hy_ig_diff=hy_ig_diff,
            tight_liq=(liq < -30),
            loose_liq=(liq > 30),
            high_vol=(rsk < -30),
            low_vol=(rsk > 30),
            vol_crush_imminent=False,
            regime_label=label,
            conviction=conviction,
        )
        condition_vectors[dt_date] = cv
        prev_scores = scores

    logger.info(f"Computed {len(condition_vectors)} daily ConditionVectors")
    return condition_vectors


def save_macro_daily(macro_df: pd.DataFrame) -> Path:
    """Save macro history to Parquet.

    Args:
        macro_df: DataFrame from build_macro_history().

    Returns:
        Path to saved file.
    """
    MACRO_DIR.mkdir(parents=True, exist_ok=True)
    macro_df.to_parquet(FRED_INDICATORS_FILE, engine="pyarrow")
    logger.info(f"Saved macro history to {FRED_INDICATORS_FILE}")
    return FRED_INDICATORS_FILE


@click.command()
@click.option("--start", default="2005-01-01", help="History start date")
@click.option("--end", default=None, help="History end date (default: today)")
@click.option("--debug", is_flag=True, help="Enable DEBUG logging")
def main(start: str, end: str | None, debug: bool) -> None:
    """Pull historical FRED data and save macro history."""
    from utils.logging import setup_logging
    setup_logging("DEBUG" if debug else None)

    from dotenv import load_dotenv
    load_dotenv()

    macro_df = build_macro_history(start, end)
    save_macro_daily(macro_df)


if __name__ == "__main__":
    main()
