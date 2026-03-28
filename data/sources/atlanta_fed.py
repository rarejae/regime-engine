"""Atlanta Fed GDPNow scraper for the Macro Regime Engine.

Fetches the latest GDPNow real GDP growth estimate.
Updated approximately 6 times per month.
No API key required.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

import requests
from loguru import logger


class AtlantaFedSource:
    """Scraper for the Atlanta Fed GDPNow tracker."""

    # Atlanta Fed publishes a machine-readable XML/JSON alongside the HTML page
    GDPNOW_JSON_URL = (
        "https://www.atlantafed.org/cqer/research/gdpnow"
    )
    # The Atlanta Fed also provides a direct data file
    GDPNOW_EXCEL_URL = (
        "https://www.atlantafed.org/-/media/documents/cqer/researchcq/"
        "gdpnow/GDPNowCast.xlsx"
    )
    # Fallback: public-facing tracker page (scrape headline number)
    GDPNOW_PAGE_URL = "https://www.atlantafed.org/cqer/research/gdpnow"
    TIMEOUT = 20

    def get_gdpnow(self) -> tuple[float, datetime]:
        """
        Fetch the latest GDPNow GDP growth estimate and its as-of date.

        Tries the published Excel file first, then falls back to scraping
        the HTML page for the headline number.

        Returns:
            Tuple of (gdpnow_estimate: float, as_of: datetime).
            gdpnow_estimate is annualized quarterly GDP growth rate in %.
        """
        try:
            return self._from_excel()
        except Exception as e:
            logger.warning(f"GDPNow Excel fetch failed ({e}); trying HTML scrape")

        try:
            return self._from_html()
        except Exception as e:
            logger.warning(f"GDPNow HTML scrape also failed ({e})")
            raise RuntimeError("GDPNow unavailable from all sources") from e

    # ── Private helpers ────────────────────────────────────────────────────────

    def _from_excel(self) -> tuple[float, datetime]:
        """
        Parse the GDPNow Excel file for the latest estimate.

        Returns:
            Tuple of (estimate, as_of_date).
        """
        import pandas as pd

        logger.debug("Downloading GDPNow Excel file")
        resp = requests.get(self.GDPNOW_EXCEL_URL, timeout=self.TIMEOUT)
        resp.raise_for_status()

        import io
        df = pd.read_excel(io.BytesIO(resp.content), sheet_name=0)

        # Column names vary by vintage — search for 'GDPNow' pattern
        gdpnow_cols = [c for c in df.columns if "gdpnow" in str(c).lower() or "nowcast" in str(c).lower()]
        date_cols = [c for c in df.columns if "date" in str(c).lower()]

        if not gdpnow_cols:
            # Try second column which is typically the GDPNow series
            gdpnow_col = df.columns[1]
        else:
            gdpnow_col = gdpnow_cols[0]

        if not date_cols:
            date_col = df.columns[0]
        else:
            date_col = date_cols[0]

        # Drop rows with NaN in the GDPNow column
        df_clean = df[[date_col, gdpnow_col]].dropna()
        if df_clean.empty:
            raise ValueError("GDPNow Excel file parsed but no valid rows found")

        latest = df_clean.iloc[-1]
        estimate = float(latest[gdpnow_col])
        raw_date = latest[date_col]

        if isinstance(raw_date, str):
            as_of = datetime.strptime(raw_date.strip()[:10], "%Y-%m-%d")
        elif hasattr(raw_date, "to_pydatetime"):
            as_of = raw_date.to_pydatetime()
        else:
            as_of = datetime.utcnow()

        logger.debug(f"GDPNow (Excel): {estimate:.1f}% as of {as_of.date()}")
        return estimate, as_of

    def _from_html(self) -> tuple[float, datetime]:
        """
        Scrape the GDPNow headline estimate from the Atlanta Fed page.

        Returns:
            Tuple of (estimate, as_of_date).
        """
        from bs4 import BeautifulSoup

        logger.debug("Scraping GDPNow from HTML page")
        resp = requests.get(self.GDPNOW_PAGE_URL, timeout=self.TIMEOUT)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "lxml")

        # Look for patterns like "2.3 percent" or "-1.5%" near GDPNow text
        text = soup.get_text(separator=" ")

        # Pattern: latest estimate is typically stated as "X.X percent" near "GDPNow"
        pattern = r"GDPNow[^.]*?([-+]?\d+\.?\d*)\s*percent"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            estimate = float(match.group(1))
            logger.debug(f"GDPNow (HTML scrape): {estimate:.1f}%")
            return estimate, datetime.utcnow()

        # Fallback: look for any decimal near "latest estimate"
        pattern2 = r"latest estimate[^.]*?([-+]?\d+\.\d+)"
        match2 = re.search(pattern2, text, re.IGNORECASE)
        if match2:
            estimate = float(match2.group(1))
            logger.debug(f"GDPNow (HTML scrape fallback): {estimate:.1f}%")
            return estimate, datetime.utcnow()

        raise ValueError("Could not parse GDPNow estimate from HTML page")
