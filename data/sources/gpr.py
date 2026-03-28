"""Geopolitical Risk (GPR) Index downloader for the Macro Regime Engine.

Downloads the Caldara-Iacoviello GPR index directly from their published data file.
No API key required. Updated daily.
Reference: https://www.matteoiacoviello.com/gpr.htm
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Optional

import pandas as pd
import requests
from loguru import logger


class GPRSource:
    """Downloader for the Caldara & Iacoviello Geopolitical Risk Index."""

    # Primary: recent daily data (Excel format)
    GPR_DAILY_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"
    # Fallback: monthly historical data
    GPR_MONTHLY_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_monthly.xls"
    TIMEOUT = 30  # larger file, allow more time

    def get_gpr(self) -> tuple[float, datetime]:
        """
        Fetch the latest GPR index value.

        Tries the daily file first, then falls back to monthly.

        Returns:
            Tuple of (gpr_value: float, as_of: datetime).
        """
        try:
            return self._from_daily_file()
        except Exception as e:
            logger.warning(f"GPR daily file failed ({e}); trying monthly fallback")

        try:
            return self._from_monthly_file()
        except Exception as e:
            logger.warning(f"GPR monthly file also failed: {e}")
            raise RuntimeError("GPR index unavailable from all sources") from e

    def get_history(self, years: int = 5) -> pd.DataFrame:
        """
        Return historical GPR data for z-score normalization.

        Args:
            years: Number of years of history to return.

        Returns:
            DataFrame with columns [date, gpr].
        """
        try:
            df = self._load_daily_df()
        except Exception:
            df = self._load_monthly_df()

        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=365 * years)
        df_filtered = df[df["date"] >= cutoff].copy()
        return df_filtered.reset_index(drop=True)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _from_daily_file(self) -> tuple[float, datetime]:
        """Parse the daily GPR Excel file for the latest value."""
        df = self._load_daily_df()
        latest = df.iloc[-1]
        as_of = latest["date"]
        if hasattr(as_of, "to_pydatetime"):
            as_of = as_of.to_pydatetime()
        gpr_val = float(latest["gpr"])
        logger.debug(f"GPR (daily): {gpr_val:.1f} as of {as_of}")
        return gpr_val, as_of

    def _from_monthly_file(self) -> tuple[float, datetime]:
        """Parse the monthly GPR Excel file for the latest value."""
        df = self._load_monthly_df()
        latest = df.iloc[-1]
        as_of = latest["date"]
        if hasattr(as_of, "to_pydatetime"):
            as_of = as_of.to_pydatetime()
        gpr_val = float(latest["gpr"])
        logger.debug(f"GPR (monthly): {gpr_val:.1f} as of {as_of}")
        return gpr_val, as_of

    def _load_daily_df(self) -> pd.DataFrame:
        """Download and parse the daily GPR Excel file."""
        logger.debug("Downloading GPR daily Excel file")
        resp = requests.get(self.GPR_DAILY_URL, timeout=self.TIMEOUT)
        resp.raise_for_status()
        return self._parse_gpr_excel(io.BytesIO(resp.content))

    def _load_monthly_df(self) -> pd.DataFrame:
        """Download and parse the monthly GPR Excel file."""
        logger.debug("Downloading GPR monthly Excel file")
        resp = requests.get(self.GPR_MONTHLY_URL, timeout=self.TIMEOUT)
        resp.raise_for_status()
        return self._parse_gpr_excel(io.BytesIO(resp.content))

    @staticmethod
    def _parse_gpr_excel(file_obj: io.BytesIO) -> pd.DataFrame:
        """
        Parse a GPR Excel file and return a tidy DataFrame.

        The GPR files use varying column naming conventions across vintages.
        We search for the date column and the main GPR series column.

        Args:
            file_obj: Binary file-like object of the Excel file.

        Returns:
            DataFrame with columns [date, gpr], sorted ascending, NaNs dropped.
        """
        df = pd.read_excel(file_obj, sheet_name=0)

        # Identify the date column
        date_col: Optional[str] = None
        gpr_col: Optional[str] = None

        for col in df.columns:
            col_lower = str(col).lower().strip()
            if date_col is None and any(k in col_lower for k in ("date", "year", "month", "day")):
                date_col = col
            if gpr_col is None and col_lower in ("gpr", "gprd", "gpr_daily", "gpr_index", "gprc"):
                gpr_col = col

        # If no exact match, use heuristic: first column = date, 'GPR' anywhere
        if date_col is None:
            date_col = df.columns[0]
        if gpr_col is None:
            gpr_candidates = [c for c in df.columns if "gpr" in str(c).lower()]
            if not gpr_candidates:
                raise ValueError(f"Cannot identify GPR column in: {list(df.columns)}")
            gpr_col = gpr_candidates[0]

        result = df[[date_col, gpr_col]].copy()
        result.columns = ["date", "gpr"]
        result["date"] = pd.to_datetime(result["date"], errors="coerce")
        result["gpr"] = pd.to_numeric(result["gpr"], errors="coerce")
        result = result.dropna().sort_values("date").reset_index(drop=True)

        if result.empty:
            raise ValueError("GPR Excel file parsed but resulted in empty DataFrame")

        return result
