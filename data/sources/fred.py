"""FRED API client for the Macro Regime Engine.

Covers ~20 indicators via the fredapi library.
Requires FRED_API_KEY in the environment (or .env file).
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from fredapi import Fred
from loguru import logger


class FREDSource:
    """Thin wrapper around fredapi.Fred with error handling and caching hooks."""

    def __init__(self) -> None:
        api_key = os.getenv("FRED_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "FRED_API_KEY is not set. Get a free key at "
                "https://fred.stlouisfed.org/docs/api/api_key.html"
            )
        self.client = Fred(api_key=api_key)
        logger.debug("FREDSource initialised")

    # ── Low-level helpers ──────────────────────────────────────────────────────

    def get_series(self, series_id: str, start_date: Optional[str] = None) -> pd.Series:
        """
        Fetch a FRED series as a pd.Series with DatetimeIndex.

        Args:
            series_id: FRED series identifier, e.g. "DFF".
            start_date: ISO date string "YYYY-MM-DD"; defaults to 5 years ago.

        Returns:
            pd.Series with DatetimeIndex, NaNs dropped.
        """
        if start_date is None:
            start_date = (datetime.today() - timedelta(days=365 * 5 + 30)).strftime("%Y-%m-%d")
        logger.debug(f"FRED fetch: {series_id} from {start_date}")
        series = self.client.get_series(series_id, observation_start=start_date)
        return series.dropna()

    def get_latest(self, series_id: str) -> float:
        """
        Return the most recent non-NaN observation for a FRED series.

        Args:
            series_id: FRED series identifier.

        Returns:
            Most recent float value.
        """
        series = self.client.get_series(series_id)
        return float(series.dropna().iloc[-1])

    # ── Derived computations ───────────────────────────────────────────────────

    def get_yoy_pct(self, series_id: str) -> float:
        """
        Compute year-over-year percent change using the two most recent
        non-NaN observations separated by ~12 months.

        Args:
            series_id: FRED series identifier for a monthly series.

        Returns:
            YoY % change as a float (e.g. 3.2 for 3.2%).
        """
        start = (datetime.today() - timedelta(days=400)).strftime("%Y-%m-%d")
        s = self.get_series(series_id, start_date=start)
        if len(s) < 13:
            raise ValueError(f"Not enough history for YoY on {series_id}")
        current = s.iloc[-1]
        year_ago = s.iloc[-13]
        return float((current / year_ago - 1) * 100)

    def get_mom_3m_annualized(self, series_id: str) -> float:
        """
        Compute 3-month annualized MoM % change: ((v[-1]/v[-4])^4 - 1) * 100.

        Args:
            series_id: FRED monthly series identifier.

        Returns:
            3-month annualized rate as a float.
        """
        start = (datetime.today() - timedelta(days=200)).strftime("%Y-%m-%d")
        s = self.get_series(series_id, start_date=start)
        if len(s) < 4:
            raise ValueError(f"Not enough history for 3m annualized on {series_id}")
        return float(((s.iloc[-1] / s.iloc[-4]) ** 4 - 1) * 100)

    def get_delta_12m(self, series_id: str) -> float:
        """
        Compute the change from 12 months ago to today in original units.
        Useful for FFR_DELTA_12M (returns bps if input is %).

        Args:
            series_id: FRED daily series identifier.

        Returns:
            Difference in basis points (×100) for rate series.
        """
        start = (datetime.today() - timedelta(days=400)).strftime("%Y-%m-%d")
        s = self.get_series(series_id, start_date=start)
        if len(s) < 2:
            raise ValueError(f"Not enough history for delta_12m on {series_id}")
        # Find observation closest to 365 days ago
        target_date = s.index[-1] - pd.Timedelta(days=365)
        idx = int(np.abs((s.index - target_date).total_seconds().values).argmin())
        delta = float(s.iloc[-1] - s.iloc[idx])
        return delta * 100  # convert % to bps

    def get_delta_3m_pct(self, series_id: str) -> float:
        """
        Compute percent change over the past ~3 months (63 trading days).

        Args:
            series_id: FRED series identifier.

        Returns:
            Percent change as float.
        """
        start = (datetime.today() - timedelta(days=120)).strftime("%Y-%m-%d")
        s = self.get_series(series_id, start_date=start)
        if len(s) < 2:
            raise ValueError(f"Not enough history for delta_3m on {series_id}")
        return float((s.iloc[-1] / s.iloc[0] - 1) * 100)

    def get_spread_bps(self, series_id_a: str, series_id_b: str) -> float:
        """
        Compute current spread between two FRED series in basis points.
        Returns (series_a_latest - series_b_latest) * 100.

        Args:
            series_id_a: Numerator series (e.g. DGS10).
            series_id_b: Subtracted series (e.g. DGS2).

        Returns:
            Spread in basis points.
        """
        a = self.get_latest(series_id_a)
        b = self.get_latest(series_id_b)
        return (a - b) * 100

    def get_mom_3ma_thousands(self, series_id: str) -> float:
        """
        Compute 3-month average of month-over-month absolute change (in thousands).
        Used for NFP: average of last 3 monthly net changes.

        Args:
            series_id: FRED monthly level series (e.g. PAYEMS in thousands).

        Returns:
            3-month average MoM change in thousands.
        """
        start = (datetime.today() - timedelta(days=180)).strftime("%Y-%m-%d")
        s = self.get_series(series_id, start_date=start)
        if len(s) < 4:
            raise ValueError(f"Not enough history for mom_3ma on {series_id}")
        diffs = s.diff().dropna().iloc[-3:]
        return float(diffs.mean())

    def get_mom_pct(self, series_id: str) -> float:
        """
        Compute MoM percent change between the two most recent observations.

        Args:
            series_id: FRED series identifier.

        Returns:
            MoM % change as float.
        """
        start = (datetime.today() - timedelta(days=90)).strftime("%Y-%m-%d")
        s = self.get_series(series_id, start_date=start)
        if len(s) < 2:
            raise ValueError(f"Not enough history for mom_pct on {series_id}")
        return float((s.iloc[-1] / s.iloc[-2] - 1) * 100)

    # ── Historical pull for normalization ──────────────────────────────────────

    def get_history_for_normalization(self, series_id: str, years: int = 5) -> pd.Series:
        """
        Pull multi-year history for z-score normalization baseline.

        Args:
            series_id: FRED series identifier.
            years: Number of years of history to pull.

        Returns:
            pd.Series with DatetimeIndex.
        """
        start = (datetime.today() - timedelta(days=365 * years + 30)).strftime("%Y-%m-%d")
        return self.get_series(series_id, start_date=start)
