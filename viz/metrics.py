"""Shared metrics for experiment packages — match backtest conventions.

Daily series: annualize with 252.
Monthly series: annualize with 12.
DCA applies contributions at month-end AFTER that month's return
(matches experiments/v11_beta_scaled.dca_terminal).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ann_factor(freq: str) -> int:
    return 252 if freq == "daily" else 12


def cagr(returns: pd.Series, freq: str = "daily") -> float:
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    n = _ann_factor(freq)
    return float((1 + r).prod() ** (n / len(r)) - 1)


def volatility(returns: pd.Series, freq: str = "daily") -> float:
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    return float(r.std() * np.sqrt(_ann_factor(freq)))


def max_drawdown(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) == 0:
        return float("nan")
    cum = (1 + r).cumprod()
    return float(((cum - cum.expanding().max()) / cum.expanding().max()).min())


def sharpe(returns: pd.Series, freq: str = "daily", rf: float = 0.0) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    n = _ann_factor(freq)
    ar = r.mean() * n
    av = r.std() * np.sqrt(n)
    return float((ar - rf) / av) if av > 0 else 0.0


def sortino(returns: pd.Series, freq: str = "daily") -> float:
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    n = _ann_factor(freq)
    ar = r.mean() * n
    neg = r[r < 0]
    ds = neg.std() * np.sqrt(n) if len(neg) > 10 else r.std() * np.sqrt(n)
    return float(ar / ds) if ds > 0 else 0.0


def calmar(returns: pd.Series, freq: str = "daily") -> float:
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    ar = r.mean() * _ann_factor(freq)
    dd = max_drawdown(r)
    return float(ar / abs(dd)) if dd != 0 and not np.isnan(dd) else 0.0


def terminal_wealth(returns: pd.Series, start: float = 1.0) -> float:
    r = returns.dropna()
    if len(r) == 0:
        return float("nan")
    return float(start * (1 + r).prod())


def equity_curve(returns: pd.Series, start: float = 1.0) -> pd.Series:
    r = returns.dropna()
    return start * (1 + r).cumprod()


def drawdown_curve(returns: pd.Series) -> pd.Series:
    eq = equity_curve(returns)
    peak = eq.expanding().max()
    return (eq - peak) / peak


def dca_path(
    monthly_returns: pd.Series,
    start: float = 21_000,
    contrib: float = 700,
    convention: str = "earn_then_contribute",
) -> pd.Series:
    """Build a DCA wealth path from monthly returns.

    Conventions:
      earn_then_contribute (default, interactive):
          each month: val = val * (1+r) + contrib
      vault (matches experiments/v11 dca_terminal used in published notes):
          month 0: val = start + contrib  (return[0] skipped)
          month i>0: val = val * (1+r) + contrib
    """
    r = monthly_returns.dropna()
    vals = []
    val = float(start)
    for i, (_, ret) in enumerate(r.items()):
        ret = float(ret)
        if convention == "vault":
            if i > 0:
                val = val * (1 + ret) + contrib
            else:
                val = val + contrib
        else:  # earn_then_contribute
            val = val * (1 + ret) + contrib
        vals.append(val)
    return pd.Series(vals, index=r.index, name="dca")


def dca_terminal(
    monthly_returns: pd.Series,
    start: float = 21_000,
    contrib: float = 700,
    convention: str = "earn_then_contribute",
) -> float:
    path = dca_path(monthly_returns, start=start, contrib=contrib, convention=convention)
    return float(path.iloc[-1]) if len(path) else float("nan")


def metrics_from_daily(daily: pd.Series) -> dict:
    return {
        "CAGR": cagr(daily, "daily"),
        "Vol": volatility(daily, "daily"),
        "Sharpe": sharpe(daily, "daily"),
        "Sortino": sortino(daily, "daily"),
        "MaxDD": max_drawdown(daily),
        "Calmar": calmar(daily, "daily"),
        "Terminal_$1": terminal_wealth(daily, 1.0),
    }


def crisis_drawdowns(daily: pd.Series, windows: dict[str, tuple[str, str]]) -> dict[str, float]:
    out = {}
    for name, (start, end) in windows.items():
        w = daily.loc[start:end]
        out[name] = max_drawdown(w) if len(w) > 5 else float("nan")
    return out


CRISIS_WINDOWS = {
    "Dot-com 2000-02": ("2000-03-01", "2002-12-31"),
    "GFC 07-09": ("2007-11-01", "2009-03-31"),
    "COVID 2020": ("2020-02-01", "2020-04-30"),
    "2022 bear": ("2022-01-01", "2022-12-31"),
}
