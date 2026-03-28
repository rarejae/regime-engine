"""Utility helpers for the Macro Regime Engine.

Includes:
  - Black-Scholes delta calculation
  - Market calendar utilities (next/nearest expiry dates)
  - Delta-to-strike conversion from an options chain
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import norm


# ── Black-Scholes ─────────────────────────────────────────────────────────────

def bs_delta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    """
    Calculate Black-Scholes delta for a European option.

    Args:
        S: Current spot price of the underlying.
        K: Strike price.
        T: Time to expiry in years (e.g. 21/365 for 21 DTE).
        r: Risk-free rate as decimal (e.g. 0.045 for 4.5%).
        sigma: Implied volatility as decimal (e.g. 0.18 for 18%).
        option_type: "call" or "put".

    Returns:
        Delta as float: calls in [0, 1], puts in [-1, 0].
    """
    if T <= 0:
        # At expiry: delta is 0 or ±1 based on moneyness
        if option_type == "call":
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0

    if sigma <= 0:
        raise ValueError(f"Implied volatility must be positive, got {sigma}")

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

    if option_type == "call":
        return float(norm.cdf(d1))
    else:
        return float(norm.cdf(d1) - 1)


def bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    """
    Calculate Black-Scholes theoretical option price.

    Args:
        S: Spot price.
        K: Strike price.
        T: Time to expiry in years.
        r: Risk-free rate as decimal.
        sigma: Implied volatility as decimal.
        option_type: "call" or "put".

    Returns:
        Theoretical option price (per share).
    """
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == "call":
        return float(S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2))
    else:
        return float(K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


# ── Options chain utilities ───────────────────────────────────────────────────

def filter_chain(
    chain: pd.DataFrame,
    min_open_interest: int = 100,
    max_bid_ask_spread_pct: float = 0.20,
    min_volume: int = 10,
) -> pd.DataFrame:
    """
    Filter an options chain DataFrame for liquid, tradeable strikes.

    Args:
        chain: yfinance options chain DataFrame (calls or puts).
        min_open_interest: Minimum open interest required.
        max_bid_ask_spread_pct: Max (ask - bid) / mid as a fraction.
        min_volume: Minimum volume.

    Returns:
        Filtered DataFrame with added 'mid' column.
    """
    df = chain.copy()

    # Compute mid price
    df["mid"] = (df["bid"] + df["ask"]) / 2

    # Filter zero or negative mid prices
    df = df[df["mid"] > 0]

    # Open interest filter
    if "openInterest" in df.columns:
        df = df[df["openInterest"] >= min_open_interest]

    # Volume filter
    if "volume" in df.columns:
        df = df[df["volume"].fillna(0) >= min_volume]

    # Bid-ask spread filter
    with np.errstate(divide="ignore", invalid="ignore"):
        ba_spread_pct = (df["ask"] - df["bid"]) / df["mid"].replace(0, np.nan)
    df = df[ba_spread_pct <= max_bid_ask_spread_pct]

    return df.reset_index(drop=True)


def find_strike_by_delta(
    chain: pd.DataFrame,
    target_delta: float,
    S: float,
    T: float,
    r: float,
    option_type: str = "call",
) -> Optional[pd.Series]:
    """
    Find the chain row whose computed delta is closest to target_delta.

    yfinance does not provide Greeks; we compute delta from implied volatility.

    Args:
        chain: Filtered options chain DataFrame with 'impliedVolatility' column.
        target_delta: Target delta magnitude (positive, e.g. 0.30).
        S: Spot price.
        T: Time to expiry in years.
        r: Risk-free rate as decimal.
        option_type: "call" or "put".

    Returns:
        The row from the chain closest to the target delta, or None if empty.
    """
    if chain.empty:
        return None

    deltas = []
    for _, row in chain.iterrows():
        iv = row.get("impliedVolatility", 0)
        if iv is None or iv <= 0 or pd.isna(iv):
            deltas.append(np.nan)
            continue
        try:
            delta = bs_delta(S, float(row["strike"]), T, r, float(iv), option_type)
            deltas.append(abs(delta))
        except Exception:
            deltas.append(np.nan)

    chain = chain.copy()
    chain["computed_delta"] = deltas
    chain = chain.dropna(subset=["computed_delta"])

    if chain.empty:
        return None

    # Find closest delta to target
    idx = (chain["computed_delta"] - target_delta).abs().idxmin()
    return chain.loc[idx]


# ── Date / market calendar utilities ─────────────────────────────────────────

def target_expiry_date(target_dte: int, available_expiries: list[str]) -> str:
    """
    Select the available expiry date closest to target_dte from today.

    Args:
        target_dte: Desired days-to-expiry.
        available_expiries: List of "YYYY-MM-DD" strings from yfinance.

    Returns:
        The best-matching expiry date string.

    Raises:
        ValueError: If available_expiries is empty.
    """
    if not available_expiries:
        raise ValueError("No available expiry dates provided")

    today = date.today()
    target_date = today + timedelta(days=target_dte)

    best = min(
        available_expiries,
        key=lambda d: abs((datetime.strptime(d, "%Y-%m-%d").date() - target_date).days),
    )
    return best


def dte_for_expiry(expiry: str) -> int:
    """
    Compute days-to-expiry from today for a given expiry date string.

    Args:
        expiry: Expiry date string "YYYY-MM-DD".

    Returns:
        Integer DTE (can be negative for expired options).
    """
    expiry_date = datetime.strptime(expiry, "%Y-%m-%d").date()
    return (expiry_date - date.today()).days


def years_to_expiry(expiry: str) -> float:
    """
    Convert an expiry date string to time-to-expiry in years.

    Args:
        expiry: Expiry date string "YYYY-MM-DD".

    Returns:
        Time to expiry as float (e.g. 21/365 ≈ 0.0575).
    """
    dte = max(dte_for_expiry(expiry), 1)  # avoid zero/negative T
    return dte / 365.0


def mid_price(row: pd.Series) -> float:
    """
    Compute the mid price of a bid/ask spread.

    Args:
        row: DataFrame row with 'bid' and 'ask' columns.

    Returns:
        Mid price as float. Returns last_price if bid/ask unavailable.
    """
    bid = row.get("bid", 0) or 0
    ask = row.get("ask", 0) or 0
    if bid > 0 and ask > 0:
        return (bid + ask) / 2
    return float(row.get("lastPrice", 0) or 0)
