"""CBOE scraper for the Macro Regime Engine.

Provides: equity put/call ratio.
No API key required. Falls back to VIX-derived proxy if scraping fails.
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Optional

import pandas as pd
import requests
from loguru import logger


class CBOESource:
    """Scraper for CBOE daily market statistics."""

    # CBOE publishes daily option volume CSV files
    DAILY_CSV_URL = "https://www.cboe.com/us/options/market_statistics/daily/"
    # Alternate direct CSV endpoint for total put/call
    TOTAL_PC_URL = "https://cdn.cboe.com/api/global/delayed_quotes/charts/historical/_total.json"
    TIMEOUT = 15

    def get_put_call_ratio(self) -> tuple[float, bool]:
        """
        Fetch the most recent total equity put/call ratio.

        Tries the CBOE daily data endpoint first. Falls back to a VIX-derived
        proxy (VIX / 20) if the scrape fails.

        Returns:
            Tuple of (put_call_ratio: float, is_proxy: bool).
        """
        try:
            return self._fetch_from_cboe_json(), False
        except Exception as e:
            logger.warning(f"CBOE put/call fetch failed ({e}); using VIX proxy")
            return self._vix_proxy(), True

    def get_historical_put_call(self, days: int = 252) -> Optional[pd.DataFrame]:
        """
        Attempt to fetch recent put/call ratio history from CBOE.

        Args:
            days: Number of recent trading days to retrieve.

        Returns:
            DataFrame with [date, put_call] columns, or None on failure.
        """
        try:
            resp = requests.get(self.TOTAL_PC_URL, timeout=self.TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            records = data.get("data", [])
            if not records:
                return None
            df = pd.DataFrame(records, columns=["datetime", "put_call"])
            df["date"] = pd.to_datetime(df["datetime"])
            df["put_call"] = pd.to_numeric(df["put_call"], errors="coerce")
            df = df[["date", "put_call"]].dropna().sort_values("date")
            return df.tail(days).reset_index(drop=True)
        except Exception as e:
            logger.warning(f"CBOE historical put/call unavailable: {e}")
            return None

    # ── Private helpers ────────────────────────────────────────────────────────

    def _fetch_from_cboe_json(self) -> float:
        """
        Parse the CBOE delayed quotes JSON for latest total put/call ratio.

        Returns:
            Latest put/call ratio as float.

        Raises:
            ValueError: If data cannot be parsed.
        """
        resp = requests.get(self.TOTAL_PC_URL, timeout=self.TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        records = data.get("data", [])
        if not records:
            raise ValueError("Empty CBOE put/call data response")
        # Records are [timestamp, value] pairs — take the last non-null
        for row in reversed(records):
            try:
                val = float(row[1])
                if val > 0:
                    logger.debug(f"CBOE put/call ratio: {val}")
                    return val
            except (TypeError, IndexError, ValueError):
                continue
        raise ValueError("No valid put/call ratio found in CBOE response")

    @staticmethod
    def _vix_proxy() -> float:
        """
        Compute a rough put/call proxy from VIX level.
        VIX / 20 normalizes around 1.0 at VIX=20.
        Not a true put/call ratio — clearly flagged as proxy.

        Returns:
            Float approximation of put/call ratio.
        """
        try:
            import yfinance as yf
            t = yf.Ticker("^VIX")
            hist = t.history(period="5d")
            vix = float(hist["Close"].dropna().iloc[-1])
            proxy = vix / 20.0
            logger.debug(f"VIX proxy put/call: {proxy:.3f} (VIX={vix:.2f})")
            return proxy
        except Exception as e:
            logger.warning(f"VIX proxy for put/call also failed: {e}; returning 1.0")
            return 1.0
