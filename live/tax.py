"""Tax-drag model for taxable-account compounding sensitivity.

Not tax advice / not 1099-accurate. Used to answer: after short-term turnover,
does V19d still clear a hurdle vs benchmarks?
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def effective_gain_rate(
    ordinary: float = 0.32,
    ltcg: float = 0.15,
    state: float = 0.05,
    stcg_fraction: float = 0.80,
) -> float:
    """Blended marginal rate on a dollar of realized gain."""
    fed = stcg_fraction * ordinary + (1.0 - stcg_fraction) * ltcg
    # State roughly stacks on ordinary; simplify: full state on all gains
    return fed + state


def estimate_annual_turnover(monthly_returns: pd.Series, state: pd.DataFrame | None) -> pd.Series:
    """Proxy realized-gain fraction of NAV per year from allocation changes.

    If monthly_state exists with mode columns, count sleeve flips as turnover.
    Else use a conservative default turnover rate series (constant).
    """
    years = sorted(set(monthly_returns.dropna().index.year))
    out = {}
    if state is not None and {"p1_mode", "p2_mode", "gold_mode"}.issubset(state.columns):
        s = state.reindex(monthly_returns.index).ffill()
        for y in years:
            sy = s[s.index.year == y]
            if len(sy) < 2:
                out[y] = 0.5
                continue
            flips = 0
            for col in ("p1_mode", "p2_mode", "gold_mode"):
                flips += int((sy[col] != sy[col].shift(1)).sum())
            # Each flip ≈ turning over ~1/3 of portfolio once
            out[y] = min(2.5, 0.15 + flips * 0.08)
    else:
        for y in years:
            out[y] = 0.6  # default: moderately high turnover strategy
    return pd.Series(out)


def after_tax_monthly_returns(
    monthly_returns: pd.Series,
    state: pd.DataFrame | None = None,
    ordinary: float = 0.32,
    ltcg: float = 0.15,
    state_tax: float = 0.05,
    stcg_fraction: float = 0.80,
    gain_fraction_of_positive: float = 0.70,
) -> pd.Series:
    """Haircut positive months by an estimated tax accrual.

    Model: in each calendar year, estimate turnover τ. On positive months,
    accrue tax ≈ rate * gain_fraction * r * (τ / n_pos_months_scale).
    Negative months unchanged (losses accrue; simplified — no carry detail).
    """
    r = monthly_returns.dropna().copy()
    if r.empty:
        return r
    rate = effective_gain_rate(ordinary, ltcg, state_tax, stcg_fraction)
    turnover = estimate_annual_turnover(r, state)
    out = []
    for ts, ret in r.items():
        y = ts.year
        tau = float(turnover.get(y, 0.6))
        if ret > 0:
            # Tax accrual proportional to gain and turnover intensity
            tax = rate * gain_fraction_of_positive * min(1.0, tau) * ret
            out.append(ret - tax)
        else:
            out.append(ret)
    return pd.Series(out, index=r.index, name=r.name)


def after_tax_metrics_table(
    monthly: pd.DataFrame,
    daily: pd.DataFrame,
    state: pd.DataFrame | None,
    ordinary: float,
    ltcg: float,
    state_tax: float,
    stcg_fraction: float,
) -> pd.DataFrame:
    """Compare pre-tax terminal from daily vs after-tax DCA-style path from monthly."""
    from viz.metrics import metrics_from_daily, dca_terminal, cagr, max_drawdown, sharpe

    rows = []
    for col in monthly.columns:
        pre = metrics_from_daily(daily[col].dropna())
        at = after_tax_monthly_returns(
            monthly[col], state if col == "v19d" else None,
            ordinary, ltcg, state_tax, stcg_fraction,
        )
        # After-tax "daily-equivalent" metrics from monthly haircut series
        rows.append({
            "Strategy": col,
            "CAGR_pre": pre["CAGR"],
            "CAGR_after_tax_approx": cagr(at, "monthly"),
            "MaxDD_pre": pre["MaxDD"],
            "MaxDD_after_tax_approx": max_drawdown(at),
            "Sharpe_pre": pre["Sharpe"],
            "Terminal_$1_pre": pre["Terminal_$1"],
            "Terminal_$1_after_tax_approx": float((1 + at).prod()) if len(at) else np.nan,
            "DCA_pre_vault": dca_terminal(monthly[col], 21000, 700, "vault"),
            "DCA_after_tax_vault": dca_terminal(at, 21000, 700, "vault"),
            "blended_tax_rate": effective_gain_rate(ordinary, ltcg, state_tax, stcg_fraction),
        })
    return pd.DataFrame(rows)
