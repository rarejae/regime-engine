"""BLS (Bureau of Labor Statistics) API client for the Macro Regime Engine.

Used as a backup source for CPI and NFP when FRED data is stale or unavailable.
Free API key optional but recommended: https://registrationapi.bls.gov/registrationapi/
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import pandas as pd
import requests
from loguru import logger


class BLSSource:
    """Client for the BLS Public Data API v2."""

    BASE_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    TIMEOUT = 15  # seconds

    def __init__(self) -> None:
        self.api_key: Optional[str] = os.getenv("BLS_API_KEY", "")
        if self.api_key:
            logger.debug("BLSSource initialised with API key (500 req/day limit)")
        else:
            logger.debug("BLSSource initialised without API key (25 req/day limit)")

    def get_series(
        self,
        series_id: str,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch a BLS time series and return it as a tidy DataFrame.

        Args:
            series_id: BLS series identifier, e.g. "CUSR0000SA0" for CPI-U.
            start_year: First year to fetch (defaults to 5 years ago).
            end_year: Last year to fetch (defaults to current year).

        Returns:
            DataFrame with columns: date (datetime), value (float).
            Sorted ascending by date.
        """
        if end_year is None:
            end_year = datetime.today().year
        if start_year is None:
            start_year = end_year - 5

        logger.debug(f"BLS fetch: {series_id} {start_year}–{end_year}")

        payload: dict = {
            "seriesid": [series_id],
            "startyear": str(start_year),
            "endyear": str(end_year),
        }
        if self.api_key:
            payload["registrationkey"] = self.api_key

        response = requests.post(self.BASE_URL, json=payload, timeout=self.TIMEOUT)
        response.raise_for_status()

        body = response.json()
        if body.get("status") != "REQUEST_SUCCEEDED":
            messages = body.get("message", [])
            raise RuntimeError(f"BLS API error for {series_id}: {messages}")

        raw = body["Results"]["series"][0]["data"]
        return self._parse(raw)

    # ── Public derived helpers ─────────────────────────────────────────────────

    def get_cpi_yoy(self) -> float:
        """
        Compute CPI-U YoY % change using the BLS public API.

        Returns:
            Year-over-year CPI % change as float.
        """
        df = self.get_series("CUSR0000SA0")
        if len(df) < 13:
            raise ValueError("Not enough BLS CPI history for YoY calculation")
        current = df["value"].iloc[-1]
        year_ago = df["value"].iloc[-13]
        return float((current / year_ago - 1) * 100)

    def get_nfp_3ma(self) -> float:
        """
        Compute 3-month average of monthly NFP changes (in thousands).

        Returns:
            3-month average MoM change in thousands of jobs.
        """
        df = self.get_series("CES0000000001")
        if len(df) < 4:
            raise ValueError("Not enough BLS NFP history")
        diffs = df["value"].diff().dropna().iloc[-3:]
        return float(diffs.mean())

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _parse(raw: list[dict]) -> pd.DataFrame:
        """
        Parse raw BLS API response data into a sorted DataFrame.

        BLS returns data newest-first with period codes like "M01"–"M12".
        Annual entries ("M13") are excluded.

        Args:
            raw: List of dicts from BLS Results.series[0].data.

        Returns:
            DataFrame with columns [date, value], sorted ascending.
        """
        records = []
        for entry in raw:
            period = entry.get("period", "")
            if not period.startswith("M") or period == "M13":
                continue  # skip annual summaries
            month = int(period[1:])
            year = int(entry["year"])
            try:
                date = datetime(year, month, 1)
                value = float(entry["value"])
                records.append({"date": date, "value": value})
            except (ValueError, KeyError):
                continue

        df = pd.DataFrame(records, columns=["date", "value"])
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
