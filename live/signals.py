"""V19d signal engine — same rules as experiments/v19d_final."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SMA_PERIODS = [126, 200, 252]


@dataclass
class SleeveState:
    asset: str  # QQQ, IVV, IAU
    score: int
    mode: str
    levered: bool
    cb_breach: bool


def compute_smas(prices: pd.DataFrame) -> dict[int, pd.DataFrame]:
    return {p: prices.rolling(p, min_periods=p).mean() for p in SMA_PERIODS}


def asset_score(day: pd.Timestamp, asset: str, prices: pd.DataFrame, smas: dict) -> int:
    if asset not in prices.columns:
        return 0
    hist = prices.loc[:day, asset].dropna()
    if hist.empty:
        return 0
    px = float(hist.iloc[-1])
    sc = 0
    for per in SMA_PERIODS:
        s = smas[per].loc[:day, asset].dropna()
        if len(s) and px > float(s.iloc[-1]):
            sc += 1
    return sc


def check_breach(day: pd.Timestamp, asset: str, prices: pd.DataFrame, smas: dict) -> bool:
    if asset not in prices.columns:
        return False
    hist = prices.loc[:day, asset].dropna()
    if hist.empty:
        return False
    px = float(hist.iloc[-1])
    b = 0
    for per in SMA_PERIODS:
        s = smas[per].loc[:day, asset].dropna()
        if len(s) and px < float(s.iloc[-1]):
            b += 1
    return b >= 3


def pod1_mode(qqq_score: int, ivv_score: int) -> tuple[str, bool]:
    """Returns (mode, levered)."""
    if qqq_score >= 3:
        if ivv_score <= 1:
            return "qqq", False
        return "qld", True
    if qqq_score == 2:
        return "qqq_partial", False
    return "cash", False


def pod2_mode(ivv_score: int) -> tuple[str, bool]:
    if ivv_score >= 3:
        return "sso", True
    if ivv_score == 2:
        return "ivv_partial", False
    return "cash", False


def gold_mode(iau_score: int) -> str:
    return "iau" if iau_score >= 3 else "cash"


def evaluate_day(day: pd.Timestamp, prices: pd.DataFrame, smas: dict | None = None) -> dict:
    smas = smas or compute_smas(prices)
    sc = {a: asset_score(day, a, prices, smas) for a in ("QQQ", "IVV", "IAU")}
    p1, p1_lev = pod1_mode(sc["QQQ"], sc["IVV"])
    p2, p2_lev = pod2_mode(sc["IVV"])
    g = gold_mode(sc["IAU"])
    breaches = {a: check_breach(day, a, prices, smas) for a in ("QQQ", "IVV", "IAU")}
    return {
        "day": day.strftime("%Y-%m-%d"),
        "scores": sc,
        "p1_mode": p1,
        "p1_lev": p1_lev,
        "p2_mode": p2,
        "p2_lev": p2_lev,
        "gold_mode": g,
        "breaches": breaches,
        "smas_ok": all(
            not np.isnan(smas[252].loc[:day, a].iloc[-1])
            if a in prices.columns and len(smas[252].loc[:day, a].dropna())
            else False
            for a in ("QQQ", "IVV")
        ),
    }


def ticker_for_mode(mode: str) -> str | None:
    return {
        "qld": "QLD",
        "qqq": "QQQ",
        "qqq_partial": "QQQ",
        "sso": "SSO",
        "ivv": "IVV",
        "ivv_partial": "IVV",
        "iau": "IAU",
        "cash": None,
    }.get(mode)
