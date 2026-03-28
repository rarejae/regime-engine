"""Yahoo Finance client via yfinance for the Macro Regime Engine.

Covers: VIX, VIX3M, VVIX, SKEW, MOVE, WTI, Gold, Copper, SPY, RSP.
No API key required.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd
import yfinance as yf
from loguru import logger


class YahooSource:
    """Wrapper around yfinance with error handling for regime engine indicators."""

    # ── Core price getters ─────────────────────────────────────────────────────

    def get_current_price(self, ticker: str) -> float:
        """
        Get the latest close price for a ticker (15-min delayed for indices).

        Args:
            ticker: Yahoo Finance ticker symbol, e.g. "^VIX".

        Returns:
            Most recent close price as float.
        """
        logger.debug(f"Yahoo fetch: {ticker}")
        t = yf.Ticker(ticker)
        hist = t.history(period="5d")
        if hist.empty:
            raise ValueError(f"No price data returned for {ticker}")
        return float(hist["Close"].dropna().iloc[-1])

    def get_history(self, ticker: str, period: str = "5y") -> pd.Series:
        """
        Get historical daily close prices.

        Args:
            ticker: Yahoo Finance ticker symbol.
            period: yfinance period string ("1y", "5y", etc.).

        Returns:
            pd.Series of close prices with DatetimeIndex.
        """
        logger.debug(f"Yahoo history: {ticker} ({period})")
        t = yf.Ticker(ticker)
        hist = t.history(period=period)
        if hist.empty:
            raise ValueError(f"No history returned for {ticker}")
        return hist["Close"].dropna()

    # ── Derived indicators ─────────────────────────────────────────────────────

    def get_vix_term_structure(self) -> float:
        """
        Compute VIX3M minus VIX (positive = contango = calm).

        Returns:
            Float: VIX3M level minus VIX level.
        """
        vix = self.get_current_price("^VIX")
        try:
            vix3m = self.get_current_price("^VIX3M")
        except Exception:
            logger.warning("^VIX3M unavailable; using VIX9D as proxy for VIX term structure")
            vix3m = self.get_current_price("^VIX9D")
        spread = vix3m - vix
        logger.debug(f"VIX term structure: {vix3m:.2f} - {vix:.2f} = {spread:.2f}")
        return spread

    def get_return_3m_pct(self, ticker: str) -> float:
        """
        Compute trailing 3-month (≈63 trading day) total return.

        Args:
            ticker: Yahoo Finance ticker symbol.

        Returns:
            3-month return as % (e.g. 5.3 for 5.3%).
        """
        hist = self.get_history(ticker, period="6mo")
        if len(hist) < 60:
            raise ValueError(f"Insufficient history for 3m return on {ticker}")
        ret = (hist.iloc[-1] / hist.iloc[-63] - 1) * 100
        return float(ret)

    def get_yoy_pct(self, ticker: str) -> float:
        """
        Compute year-over-year percent change.

        Args:
            ticker: Yahoo Finance ticker symbol.

        Returns:
            YoY return as %.
        """
        hist = self.get_history(ticker, period="2y")
        if len(hist) < 252:
            raise ValueError(f"Insufficient history for YoY on {ticker}")
        ret = (hist.iloc[-1] / hist.iloc[-252] - 1) * 100
        return float(ret)

    def get_up_down_ratio_10d(self, ticker: str) -> float:
        """
        Compute 10-day up/down day ratio as an A/D breadth proxy.

        Args:
            ticker: Yahoo Finance ticker symbol (e.g. "RSP" for equal-weight S&P).

        Returns:
            Ratio of up days to down days over last 10 trading sessions.
            Returns 1.0 (neutral) if insufficient data.
        """
        hist = self.get_history(ticker, period="1mo")
        if len(hist) < 10:
            return 1.0
        recent = hist.iloc[-10:]
        daily_ret = recent.pct_change().dropna()
        up_days = int((daily_ret > 0).sum())
        down_days = int((daily_ret <= 0).sum())
        if down_days == 0:
            return float(up_days)
        return float(up_days / down_days)

    def get_trailing_pe(self, ticker: str = "SPY") -> Optional[float]:
        """
        Retrieve trailing P/E ratio from yfinance ticker info.

        Args:
            ticker: Yahoo Finance ticker, default "SPY".

        Returns:
            Trailing P/E as float, or None if unavailable.
        """
        t = yf.Ticker(ticker)
        info = t.info
        pe = info.get("trailingPE") or info.get("forwardPE")
        if pe is None:
            logger.warning(f"No P/E data in yfinance info for {ticker}")
        return float(pe) if pe else None

    def get_options_chain(self, ticker: str = "SPY") -> tuple[list[str], yf.Ticker]:
        """
        Return available expiration dates and the Ticker object for options access.

        Args:
            ticker: Underlying ticker, default "SPY".

        Returns:
            Tuple of (list of expiry date strings, yf.Ticker instance).
        """
        t = yf.Ticker(ticker)
        expirations = list(t.options)
        logger.debug(f"SPY options expirations available: {len(expirations)}")
        return expirations, t

    def get_chain_for_expiry(self, expiry: str, ticker: str = "SPY") -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch calls and puts for a specific expiry date.

        Args:
            expiry: Expiry date string "YYYY-MM-DD".
            ticker: Underlying ticker.

        Returns:
            Tuple of (calls_df, puts_df) DataFrames.
        """
        t = yf.Ticker(ticker)
        chain = t.option_chain(expiry)
        return chain.calls, chain.puts

    # ── History for normalization ──────────────────────────────────────────────

    def get_history_for_normalization(self, ticker: str, years: int = 5) -> pd.Series:
        """
        Pull multi-year price history for z-score normalization.

        Args:
            ticker: Yahoo Finance ticker symbol.
            years: Number of years.

        Returns:
            pd.Series of close prices.
        """
        period = f"{years}y"
        return self.get_history(ticker, period=period)
