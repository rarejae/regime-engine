"""Tests for backtest/spread_finder.py — spread discovery in historical chains."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from backtest.spread_finder import SpreadCandidate, find_best_spread
from engine.condition_vector import ConditionVector


def _make_cv(**overrides) -> ConditionVector:
    defaults = dict(
        growth=0.0, inflation=0.0, liquidity=0.0, risk=0.0,
        growth_delta=0.0, inflation_delta=0.0, liquidity_delta=0.0, risk_delta=0.0,
        vix=20.0, vix_term=2.0, vvix=100.0, skew=130.0, move_index=100.0,
        hy_oas=400.0, ig_oas=100.0, hy_ig_diff=300.0,
        tight_liq=False, loose_liq=False, high_vol=False, low_vol=False,
        vol_crush_imminent=False,
        regime_label="MIXED", conviction=50.0,
    )
    defaults.update(overrides)
    return ConditionVector(**defaults)


def _make_chain(
    underlying: float = 400.0,
    strikes: list[float] | None = None,
    dte: int = 30,
    opt_type: str = "P",
) -> pd.DataFrame:
    """Create a synthetic option chain DataFrame."""
    if strikes is None:
        strikes = [380, 385, 390, 395, 400, 405, 410, 415, 420]

    rows = []
    for s in strikes:
        moneyness = (s - underlying) / underlying
        if opt_type == "P":
            iv = 0.18 + abs(moneyness) * 0.3  # skew
            delta = -max(0.01, min(0.99, 0.5 + moneyness * 3))
        else:
            iv = 0.18 + abs(moneyness) * 0.2
            delta = max(0.01, min(0.99, 0.5 - moneyness * 3))

        mid = max(0.10, abs(moneyness) * underlying * 0.1)
        bid = mid * 0.95
        ask = mid * 1.05

        rows.append({
            "trade_date": pd.Timestamp(f"2020-01-15"),
            "expiry": pd.Timestamp(f"2020-01-15") + pd.Timedelta(days=dte),
            "dte": dte,
            "strike": s,
            "option_type": opt_type,
            "bid": round(bid, 2),
            "ask": round(ask, 2),
            "mid": round(mid, 2),
            "last_price": round(mid, 2),
            "volume": 500,
            "open_interest": 1000,
            "iv": round(iv, 4),
            "delta": round(delta, 4),
            "gamma": 0.02,
            "theta": -0.05,
            "vega": 0.15,
            "rho": 0.01,
            "underlying_close": underlying,
            "bid_ask_pct": round((ask - bid) / mid, 4) if mid > 0 else 0,
            "source": "optionsdx",
        })

    return pd.DataFrame(rows)


class TestSpreadCandidate:
    """Tests for SpreadCandidate dataclass and filters."""

    def test_meets_filters_valid_spread(self):
        spread = SpreadCandidate(
            strategy="bull_put_spread",
            trade_date=date(2020, 1, 15),
            expiry=date(2020, 2, 14),
            dte=30,
            short_strike=395.0,
            long_strike=390.0,
            short_delta=0.25,
            long_delta=0.15,
            short_mid=2.50,
            long_mid=1.00,
            width=5.0,
            entry_credit=1.50,
            max_risk=3.50,
            breakeven=393.50,
            option_type="P",
            underlying_close=400.0,
            source="optionsdx",
            short_iv=0.20,
            long_iv=0.22,
        )

        assert spread.meets_filters() is True

    def test_fails_min_width(self):
        spread = SpreadCandidate(
            strategy="bull_put_spread",
            trade_date=date(2020, 1, 15),
            expiry=date(2020, 2, 14),
            dte=30,
            short_strike=395.0,
            long_strike=395.0,
            short_delta=0.25,
            long_delta=0.25,
            short_mid=2.50,
            long_mid=2.50,
            width=0.0,
            entry_credit=0.0,
            max_risk=0.0,
            breakeven=395.0,
            option_type="P",
            underlying_close=400.0,
            source="optionsdx",
            short_iv=0.20,
            long_iv=0.20,
        )
        assert spread.meets_filters() is False

    def test_fails_credit_to_width(self):
        spread = SpreadCandidate(
            strategy="bull_put_spread",
            trade_date=date(2020, 1, 15),
            expiry=date(2020, 2, 14),
            dte=30,
            short_strike=395.0,
            long_strike=390.0,
            short_delta=0.25,
            long_delta=0.15,
            short_mid=1.00,
            long_mid=0.60,
            width=5.0,
            entry_credit=0.40,  # 0.40/5.0 = 8% < 15%
            max_risk=4.60,
            breakeven=394.60,
            option_type="P",
            underlying_close=400.0,
            source="optionsdx",
            short_iv=0.20,
            long_iv=0.22,
        )
        assert spread.meets_filters(min_credit_to_width=0.15) is False


class TestFindBestSpread:
    """Tests for find_best_spread with mocked data loader."""

    def test_returns_none_on_empty_chain(self):
        loader = MagicMock()
        loader.get_chain.return_value = pd.DataFrame()

        cv = _make_cv(growth=-50, vix=25)
        result = find_best_spread(
            loader=loader,
            trade_date=date(2020, 1, 15),
            strategy="bear_call_spread",
            cv=cv,
            target_delta=0.25,
            target_dte=30,
        )

        assert result is None

    def test_finds_spread_in_valid_chain(self):
        chain = _make_chain(underlying=400.0, dte=30, opt_type="P")
        loader = MagicMock()
        loader.get_chain.return_value = chain

        cv = _make_cv(growth=40, vix=20)
        result = find_best_spread(
            loader=loader,
            trade_date=date(2020, 1, 15),
            strategy="bull_put_spread",
            cv=cv,
            target_delta=0.25,
            target_dte=30,
        )

        assert result is not None
        assert result.strategy == "bull_put_spread"
        assert result.short_strike != result.long_strike
        assert result.width > 0
        assert result.trade_date == date(2020, 1, 15)
