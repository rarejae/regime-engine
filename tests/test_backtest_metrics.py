"""Tests for backtest/metrics.py — all 4 metric layers."""

from __future__ import annotations

from datetime import date

import pytest

from backtest.metrics import (
    portfolio_metrics,
    realized_pnl_metrics,
    regime_accuracy,
    strategy_hit_rate,
)
from backtest.trade_simulator import BacktestTrade


def _make_trade(
    pnl: float = 1.0,
    win: bool = True,
    strategy: str = "bull_put_spread",
    regime: str = "GOLDILOCKS",
    source: str = "optionsdx",
    underlying_entry: float = 400.0,
    underlying_exit: float = 410.0,
    entry_date: date | None = None,
    exit_date: date | None = None,
    exit_reason: str = "take_profit_50pct",
    days_held: int = 20,
) -> BacktestTrade:
    """Create a minimal BacktestTrade for testing."""
    return BacktestTrade(
        trade_id="test-0001",
        run_id="test-run",
        window="pre_covid",
        strategy=strategy,
        strategy_display="Bull Put Spread",
        strategy_type="credit",
        entry_date=entry_date or date(2020, 1, 15),
        expiry=date(2020, 2, 14),
        dte_at_entry=30,
        short_strike=395.0,
        long_strike=390.0,
        short_delta=0.25,
        long_delta=0.15,
        width=5.0,
        entry_credit=1.50,
        option_type="P",
        underlying_at_entry=underlying_entry,
        source=source,
        regime_label=regime,
        conviction=60.0,
        growth=40.0,
        inflation=-10.0,
        liquidity=20.0,
        risk=10.0,
        exit_date=exit_date or date(2020, 2, 4),
        exit_value=0.75,
        exit_reason=exit_reason,
        underlying_at_exit=underlying_exit,
        pnl=pnl,
        win=win,
        days_held=days_held,
    )


class TestRegimeAccuracy:
    """Layer 1: Regime classification accuracy."""

    def test_correct_bullish_prediction(self):
        trades = [_make_trade(strategy="bull_put_spread", underlying_entry=400, underlying_exit=410)]
        result = regime_accuracy(trades)
        assert result["accuracy"] == 1.0
        assert result["n"] == 1

    def test_incorrect_prediction(self):
        trades = [_make_trade(strategy="bull_put_spread", underlying_entry=400, underlying_exit=380)]
        result = regime_accuracy(trades)
        assert result["accuracy"] == 0.0

    def test_empty_trades(self):
        result = regime_accuracy([])
        assert result["accuracy"] == 0.0
        assert result["n"] == 0

    def test_mixed_accuracy(self):
        trades = [
            _make_trade(strategy="bull_put_spread", underlying_entry=400, underlying_exit=410),
            _make_trade(strategy="bull_put_spread", underlying_entry=400, underlying_exit=380),
        ]
        result = regime_accuracy(trades)
        assert result["accuracy"] == 0.5
        assert result["n"] == 2


class TestRealizedPnl:
    """Layer 3: Realized P&L metrics."""

    def test_basic_pnl(self):
        trades = [
            _make_trade(pnl=2.0, win=True),
            _make_trade(pnl=-1.0, win=False),
            _make_trade(pnl=1.5, win=True),
        ]
        result = realized_pnl_metrics(trades)

        assert result["n_trades"] == 3
        assert result["total_pnl"] == 2.5
        assert result["win_rate"] == pytest.approx(2 / 3, abs=0.01)
        assert result["avg_win"] == pytest.approx(1.75, abs=0.01)
        assert result["avg_loss"] == pytest.approx(-1.0, abs=0.01)

    def test_profit_factor(self):
        trades = [
            _make_trade(pnl=3.0, win=True),
            _make_trade(pnl=-1.0, win=False),
        ]
        result = realized_pnl_metrics(trades)
        assert result["profit_factor"] == pytest.approx(3.0, abs=0.01)

    def test_by_strategy_grouping(self):
        trades = [
            _make_trade(pnl=2.0, win=True, strategy="bull_put_spread"),
            _make_trade(pnl=-1.0, win=False, strategy="bear_call_spread"),
        ]
        result = realized_pnl_metrics(trades)
        assert "bull_put_spread" in result["by_strategy"]
        assert "bear_call_spread" in result["by_strategy"]

    def test_by_source_grouping(self):
        trades = [
            _make_trade(pnl=2.0, win=True, source="optionsdx"),
            _make_trade(pnl=1.0, win=True, source="dubach"),
        ]
        result = realized_pnl_metrics(trades)
        assert "optionsdx" in result["by_source"]
        assert "dubach" in result["by_source"]

    def test_empty_trades(self):
        result = realized_pnl_metrics([])
        assert result["n_trades"] == 0
        assert result["total_pnl"] == 0.0

    def test_max_consecutive_losses(self):
        trades = [
            _make_trade(pnl=1.0, win=True, entry_date=date(2020, 1, 1)),
            _make_trade(pnl=-1.0, win=False, entry_date=date(2020, 1, 5)),
            _make_trade(pnl=-0.5, win=False, entry_date=date(2020, 1, 10)),
            _make_trade(pnl=-0.3, win=False, entry_date=date(2020, 1, 15)),
            _make_trade(pnl=2.0, win=True, entry_date=date(2020, 1, 20)),
        ]
        result = realized_pnl_metrics(trades)
        assert result["max_consecutive_losses"] == 3


class TestPortfolioMetrics:
    """Layer 4: Sharpe ratio and drawdown."""

    def test_basic_equity_curve(self):
        trades = [
            _make_trade(pnl=1.0, win=True, entry_date=date(2020, 1, 1), exit_date=date(2020, 1, 21)),
            _make_trade(pnl=0.5, win=True, entry_date=date(2020, 2, 1), exit_date=date(2020, 2, 21)),
        ]
        result = portfolio_metrics(trades, initial_capital=10000)

        assert result["equity_curve"][0] == 10000
        assert result["equity_curve"][-1] == 10150  # +$100 + $50 per contract
        assert result["total_return_pct"] == pytest.approx(1.5, abs=0.1)

    def test_max_drawdown(self):
        trades = [
            _make_trade(pnl=2.0, win=True, entry_date=date(2020, 1, 1), exit_date=date(2020, 1, 21)),
            _make_trade(pnl=-3.0, win=False, entry_date=date(2020, 2, 1), exit_date=date(2020, 2, 21)),
            _make_trade(pnl=1.0, win=True, entry_date=date(2020, 3, 1), exit_date=date(2020, 3, 21)),
        ]
        result = portfolio_metrics(trades, initial_capital=10000)

        assert result["max_drawdown"] < 0  # should be negative

    def test_empty_trades(self):
        result = portfolio_metrics([], initial_capital=10000)
        assert result["sharpe_ratio"] == 0.0
        assert result["max_drawdown"] == 0.0
        assert result["equity_curve"] == [10000]
