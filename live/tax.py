"""Tax-drag model for taxable-account compounding sensitivity.

Not tax advice / not 1099-accurate. Answers: after realizing gains at some
turnover, how much of V19d's edge survives in a taxable account?

Single canonical model, shared by both dashboards (viz/app.py DCA paths and
viz/pages/1_Growth_simulation.py CAGR windows via viz.growth_sim.after_tax_cagr):

    each calendar year, a `realized_fraction` of that year's gain is sold and
    taxed at the blended rate; the rest defers and compounds untaxed. Losses are
    not credited, and no tax is modeled at final liquidation.

`realized_fraction` is the main lever. V19d holds sleeves between CB/mode exits,
so a value well under 1.0 is realistic; 1.0 == realize-everything-annually
(worst case). The blended rate mixes short- vs long-term treatment via
`stcg_fraction`.
"""

from __future__ import annotations

import pandas as pd


def effective_gain_rate(
    ordinary: float = 0.32,
    ltcg: float = 0.15,
    state: float = 0.05,
    stcg_fraction: float = 0.50,
) -> float:
    """Blended marginal rate on a realized dollar of gain.

    Federal short-/long-term mix plus a state rate that stacks on both. Mirrored
    by viz.growth_sim.blended_gain_rate so the two dashboards agree.
    """
    fed = stcg_fraction * ordinary + (1.0 - stcg_fraction) * ltcg
    return fed + state


def after_tax_monthly_annual(
    monthly_returns: pd.Series,
    ordinary: float = 0.32,
    ltcg: float = 0.15,
    state: float = 0.05,
    stcg_fraction: float = 0.50,
    realized_fraction: float = 0.65,
) -> pd.Series:
    """After-tax monthly return series under the annual realized-gain model.

    For each calendar year with gross return g, tax = rate * realized_fraction *
    max(g, 0); the year then compounds to (1 + g - tax) instead of (1 + g). The
    within-year month-to-month shape is preserved by scaling every month in the
    year uniformly, so drawdown timing is unchanged and only the level is taxed.

    This is the monthly-path analogue of viz.growth_sim.after_tax_cagr, so the
    DCA view and the CAGR-window view rest on the same assumption.
    """
    r = monthly_returns.dropna().astype(float)
    if r.empty:
        return r
    rate = effective_gain_rate(ordinary, ltcg, state, stcg_fraction)
    out = r.copy()
    for _year, idx in r.groupby(r.index.year).groups.items():
        seg = r.loc[idx]
        g = float((1.0 + seg).prod() - 1.0)
        if g <= 0.0:
            continue  # losses untaxed; leave the year as-is
        tax = rate * realized_fraction * g
        g_at = g - tax
        n = len(seg)
        # scale each (1+r) by k so the year compounds to (1 + g_at)
        k = ((1.0 + g_at) / (1.0 + g)) ** (1.0 / n)
        out.loc[idx] = k * (1.0 + seg) - 1.0
    return out
