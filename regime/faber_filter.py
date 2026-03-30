"""Faber 10-month SMA trend filter."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SMA_MONTHS = 10


def compute_monthly_sma(daily_prices: pd.DataFrame, months: int = SMA_MONTHS) -> pd.DataFrame:
    """Compute 10-month SMA from daily prices resampled to monthly."""
    monthly = daily_prices.resample("MS").last()
    sma = monthly.rolling(months, min_periods=months).mean()
    return sma


def compute_trend_signals(daily_prices: pd.DataFrame, months: int = SMA_MONTHS) -> pd.DataFrame:
    """For each asset, True if current price > 10-month SMA.

    Returns monthly DataFrame of booleans indexed by month-start.
    """
    monthly_close = daily_prices.resample("MS").last()
    sma = monthly_close.rolling(months, min_periods=months).mean()
    above_sma = monthly_close > sma
    return above_sma


def apply_faber_to_baseline(baseline: dict, trend_signals: dict) -> dict:
    """Apply trend filter: assets below SMA go to cash.

    Args:
        baseline: dict of {asset: weight}
        trend_signals: dict of {asset: True/False} (True = above SMA)

    Returns:
        Filtered weights with trend-off assets moved to cash.
    """
    w = {}
    moved_to_cash = 0.0

    for asset, weight in baseline.items():
        if asset == "cash":
            w[asset] = weight
            continue

        in_trend = trend_signals.get(asset, True)  # default to in-trend if no signal
        if in_trend:
            w[asset] = weight
        else:
            w[asset] = 0.0
            moved_to_cash += weight

    w["cash"] = w.get("cash", 0) + moved_to_cash
    return w


def apply_faber_to_harvey(harvey_w: dict, trend_signals: dict) -> dict:
    """Apply Faber filter to Harvey allocation: remove assets below SMA.

    Assets below SMA → allocation to cash.
    Remaining non-cash assets renormalized to maintain original non-cash total.
    """
    w = dict(harvey_w)
    original_cash = w.get("cash", 0)
    original_noncash = sum(v for k, v in w.items() if k != "cash")

    # Filter out assets below SMA
    moved = 0.0
    for asset in list(w.keys()):
        if asset == "cash":
            continue
        in_trend = trend_signals.get(asset, True)
        if not in_trend:
            moved += w[asset]
            w[asset] = 0.0

    # Renormalize remaining non-cash to maintain original non-cash total
    remaining_noncash = sum(v for k, v in w.items() if k != "cash")
    if remaining_noncash > 0 and original_noncash > 0:
        scale = original_noncash / remaining_noncash
        for asset in w:
            if asset != "cash":
                w[asset] *= scale

    # But actually, moved capital goes to cash (Faber rule)
    # So we DON'T renormalize — we just move filtered assets to cash
    w = dict(harvey_w)
    moved = 0.0
    for asset in list(w.keys()):
        if asset == "cash":
            continue
        if not trend_signals.get(asset, True):
            moved += w[asset]
            w[asset] = 0.0
    w["cash"] = w.get("cash", 0) + moved

    return w
