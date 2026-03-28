"""AAII (American Association of Individual Investors) sentiment scraper.

Fetches the weekly bull/bear sentiment survey results.
Published every Thursday. No API key required.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

import requests
from loguru import logger


class AAIISource:
    """Scraper for AAII Investor Sentiment Survey data."""

    # AAII publishes results at this URL (updated every Thursday)
    SENTIMENT_URL = "https://www.aaii.com/sentimentsurvey/sent_results"
    # AAII also provides a CSV download
    CSV_URL = "https://www.aaii.com/files/surveys/sentiment.xls"
    TIMEOUT = 15

    def get_sentiment(self) -> dict[str, float]:
        """
        Fetch the latest AAII bull/bear/neutral percentages and computed spread.

        Tries the CSV/XLS download first, then falls back to HTML scraping.

        Returns:
            Dict with keys: bullish, bearish, neutral, bull_bear_spread.
            Values are percentages (e.g. 38.2 for 38.2%).
        """
        try:
            return self._from_xls()
        except Exception as e:
            logger.warning(f"AAII XLS fetch failed ({e}); trying HTML scrape")

        try:
            return self._from_html()
        except Exception as e:
            logger.warning(f"AAII HTML scrape also failed ({e})")
            raise RuntimeError("AAII sentiment unavailable from all sources") from e

    def get_bull_bear_spread(self) -> tuple[float, bool]:
        """
        Return the bull-bear spread (bullish% - bearish%) and proxy flag.

        Returns:
            Tuple of (bull_bear_spread: float, is_proxy: bool).
        """
        try:
            data = self.get_sentiment()
            return data["bull_bear_spread"], False
        except Exception as e:
            logger.warning(f"AAII bull-bear spread unavailable: {e}; using 0.0 as neutral proxy")
            return 0.0, True

    # ── Private helpers ────────────────────────────────────────────────────────

    def _from_xls(self) -> dict[str, float]:
        """
        Parse the AAII historical sentiment XLS file.

        Returns:
            Dict with bullish, bearish, neutral, bull_bear_spread.
        """
        import io
        import pandas as pd

        logger.debug("Downloading AAII sentiment XLS")
        resp = requests.get(self.CSV_URL, timeout=self.TIMEOUT)
        resp.raise_for_status()

        df = pd.read_excel(io.BytesIO(resp.content), header=None)

        # AAII XLS: locate rows with % values
        # Find header row (contains "Bullish", "Bearish", "Neutral")
        header_row = None
        for i, row in df.iterrows():
            row_vals = [str(v).lower() for v in row.values]
            if any("bullish" in v for v in row_vals):
                header_row = i
                break

        if header_row is None:
            raise ValueError("Could not find header row in AAII XLS")

        df.columns = df.iloc[header_row]
        df = df.iloc[header_row + 1:].reset_index(drop=True)

        # Normalize column names
        col_map: dict[str, str] = {}
        for col in df.columns:
            cs = str(col).lower()
            if "bullish" in cs:
                col_map[col] = "bullish"
            elif "bearish" in cs:
                col_map[col] = "bearish"
            elif "neutral" in cs:
                col_map[col] = "neutral"
        df.rename(columns=col_map, inplace=True)

        # Drop rows with missing data and take the last valid row
        needed = [c for c in ["bullish", "bearish", "neutral"] if c in df.columns]
        df_clean = df[needed].apply(pd.to_numeric, errors="coerce").dropna()
        if df_clean.empty:
            raise ValueError("No valid rows in AAII XLS after cleaning")

        latest = df_clean.iloc[-1]

        # Values may be stored as fractions (0.382) or percentages (38.2)
        bullish = float(latest.get("bullish", 0))
        bearish = float(latest.get("bearish", 0))
        neutral = float(latest.get("neutral", 0))

        if bullish < 1.0:  # stored as fractions
            bullish *= 100
            bearish *= 100
            neutral *= 100

        result = {
            "bullish": round(bullish, 2),
            "bearish": round(bearish, 2),
            "neutral": round(neutral, 2),
            "bull_bear_spread": round(bullish - bearish, 2),
        }
        logger.debug(f"AAII sentiment: bull={bullish:.1f}% bear={bearish:.1f}% spread={result['bull_bear_spread']:.1f}%")
        return result

    def _from_html(self) -> dict[str, float]:
        """
        Scrape AAII sentiment percentages from the results page HTML.

        Returns:
            Dict with bullish, bearish, neutral, bull_bear_spread.
        """
        from bs4 import BeautifulSoup

        logger.debug("Scraping AAII sentiment from HTML")
        resp = requests.get(self.SENTIMENT_URL, timeout=self.TIMEOUT)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "lxml")
        text = soup.get_text(separator=" ")

        # Search for patterns like "Bullish 38.2%" or "38.2% Bullish"
        patterns = {
            "bullish": r"bullish[^\d]*([\d.]+)\s*%",
            "bearish": r"bearish[^\d]*([\d.]+)\s*%",
            "neutral": r"neutral[^\d]*([\d.]+)\s*%",
        }
        results: dict[str, float] = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                results[key] = float(match.group(1))

        if len(results) < 2:
            raise ValueError(f"Could not parse AAII sentiment from HTML (got: {results})")

        results.setdefault("neutral", 100.0 - results.get("bullish", 0) - results.get("bearish", 0))
        results["bull_bear_spread"] = round(
            results.get("bullish", 0) - results.get("bearish", 0), 2
        )
        return results
