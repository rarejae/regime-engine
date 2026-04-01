"""Parameterized backtest harness for validation suite.

Wraps the TAA production backtest (run_weekly_threshold.py logic)
with configurable parameters so we can run sensitivity, bootstrap,
and walk-forward tests programmatically.
"""

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv; load_dotenv()

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from taa.data import (load_daily_etf_returns, load_monthly_prices,
                       load_realized_vols, load_monthly_macro, load_monthly_asset_returns)
from taa.faber import compute_trend_scores, apply_faber_filter
from taa.harvey import compute_zscore_variables, find_similar_months, compute_expected_returns
from taa.allocation import BASELINE, ASSETS, EQUITY, direct_capital, normalize


@dataclass
class BacktestConfig:
    """All tunable parameters for the TAA system."""
    # Baseline weights
    baseline: dict = field(default_factory=lambda: dict(BASELINE))

    # Faber SMA periods
    sma_periods: list = field(default_factory=lambda: [6, 10, 12])

    # Faber filter thresholds: score -> weight multiplier
    faber_full_threshold: int = 3       # score >= this -> full weight
    faber_partial_mult: float = 0.70    # score == (full-1) -> this multiplier

    # Harvey parameters
    harvey_change_lookback: int = 12
    harvey_zscore_window: int = 120
    harvey_exclude_months: int = 36
    harvey_similar_pctl: float = 0.15

    # Allocation constraints
    max_single: float = 0.60
    max_equity: float = 0.85
    min_cash: float = 0.03

    # Leverage tiers: {tier: substitution_fraction}
    tier_subs: dict = field(default_factory=lambda: {0: 0.0, 1: 0.30, 2: 0.65})

    # Leverage parameters
    leverage_ratio: float = 2.0
    sso_expense: float = 0.0091
    qld_expense: float = 0.0089

    # Weekly de-lever
    weekly_delever_threshold: int = 3   # n-of-3 SMA breach to trigger

    # Median seed for Tier 2 determination
    seed: dict = field(default_factory=lambda: {"IVV": 0.0028, "QQQ": 0.0080})

    # Backtest window
    bt_start: str = "2002-01-01"
    bt_end: str = None  # None = use all available data


@dataclass
class BacktestResult:
    """Output from a single backtest run."""
    daily_returns: pd.Series
    ann_return: float
    ann_vol: float
    sharpe: float
    sortino: float
    max_dd: float
    calmar: float
    terminal_value: float
    n_delever_events: int
    config: BacktestConfig


# ── Shared data cache ────────────────────────────────────────────────
_DATA_CACHE = {}

def _load_data():
    """Load and cache all data needed for backtesting."""
    if _DATA_CACHE:
        return _DATA_CACHE

    daily_ret = load_daily_etf_returns()
    daily_ret = daily_ret[[c for c in daily_ret.columns if c in ASSETS]]
    monthly_prices = load_monthly_prices()
    rvol_monthly = load_realized_vols()
    asset_ret = load_monthly_asset_returns()
    asset_ret_fwd = asset_ret.shift(-1)
    macro = load_monthly_macro()

    # Risk-free rate
    from fredapi import Fred
    rfr_daily = pd.Series(0.0, index=daily_ret.index)
    key = os.environ.get("FRED_API_KEY")
    if key:
        tb = Fred(api_key=key).get_series("DTB3", observation_start="1998-01-01")
        tb.index = pd.to_datetime(tb.index)
        rfr_daily = (tb / 100 / 252).reindex(daily_ret.index, method="ffill").fillna(0)

    # Daily prices for SMA breach checks
    import yfinance as yf
    daily_prices = {}
    for our, ticker in [("IVV", "SPY"), ("QQQ", "QQQ")]:
        d = yf.download(ticker, start="1998-01-01", progress=False)
        if d is not None and not d.empty:
            p = d["Close"]
            if hasattr(p, "columns"): p = p.iloc[:, 0]
            p.index = pd.to_datetime(p.index).tz_localize(None)
            daily_prices[our] = p
    dpdf = pd.DataFrame(daily_prices)

    _DATA_CACHE.update({
        "daily_ret": daily_ret,
        "monthly_prices": monthly_prices,
        "rvol_monthly": rvol_monthly,
        "asset_ret": asset_ret,
        "asset_ret_fwd": asset_ret_fwd,
        "macro": macro,
        "rfr_daily": rfr_daily,
        "dpdf": dpdf,
    })
    return _DATA_CACHE


def _compute_trend_scores_custom(monthly_prices, sma_periods):
    """Compute trend scores with custom SMA periods."""
    scores = pd.DataFrame(0, index=monthly_prices.index, columns=monthly_prices.columns)
    for period in sma_periods:
        sma = monthly_prices.rolling(period, min_periods=period).mean()
        scores += (monthly_prices > sma).astype(int)
    return scores.shift(1)


def _apply_faber_custom(trend_scores, baseline, full_thresh, partial_mult):
    """Apply Faber filter with custom thresholds."""
    w = {}
    pool = 0.0
    for a, base_w in baseline.items():
        if a == "cash":
            w[a] = base_w
            continue
        score = trend_scores.get(a, 0)
        if score >= full_thresh:
            w[a] = base_w
        elif score == full_thresh - 1:
            w[a] = base_w * partial_mult
            pool += base_w * (1 - partial_mult)
        else:
            w[a] = 0.0
            pool += base_w
    return w, pool


def _count_sma_breaches(etf, day, dpdf, sma_dfs):
    """Count how many of the SMA DataFrames the Friday price is below."""
    if etf not in dpdf.columns:
        return 0
    prices_up_to = dpdf.loc[:day, etf]
    if len(prices_up_to) == 0:
        return 0
    price = prices_up_to.iloc[-1]
    ms = pd.Timestamp(f"{day.year}-{day.month:02d}-01")
    breaches = 0
    for sma_df in sma_dfs:
        sma_dates = sma_df.index[sma_df.index <= ms]
        if len(sma_dates) == 0:
            continue
        val = sma_df.loc[sma_dates[-1], etf]
        if pd.notna(val) and price < val:
            breaches += 1
    return breaches


def compute_metrics(daily_returns: pd.Series) -> dict:
    """Compute standard performance metrics from daily returns."""
    s = daily_returns.dropna()
    if len(s) < 20:
        return {"ann_return": 0, "ann_vol": 0, "sharpe": 0, "sortino": 0,
                "max_dd": 0, "calmar": 0, "terminal_value": 1.0}

    ar = s.mean() * 252
    av = s.std() * np.sqrt(252)
    sh = ar / av if av > 0 else 0
    neg = s[s < 0]
    ds = neg.std() * np.sqrt(252) if len(neg) > 10 else av
    so = ar / ds if ds > 0 else 0
    cum = (1 + s).cumprod()
    dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
    cal = ar / abs(dd) if dd != 0 else 0
    final = cum.iloc[-1]
    return {"ann_return": ar, "ann_vol": av, "sharpe": sh, "sortino": so,
            "max_dd": dd, "calmar": cal, "terminal_value": final}


def run_backtest(config: BacktestConfig = None, data: dict = None,
                 verbose: bool = False) -> BacktestResult:
    """Run full TAA backtest with given configuration.

    Args:
        config: BacktestConfig with all parameters. Uses defaults if None.
        data: Pre-loaded data dict (from _load_data). Loaded if None.
        verbose: Print progress.

    Returns:
        BacktestResult with daily returns and metrics.
    """
    if config is None:
        config = BacktestConfig()
    if data is None:
        data = _load_data()

    daily_ret = data["daily_ret"]
    monthly_prices = data["monthly_prices"]
    rvol_monthly = data["rvol_monthly"]
    asset_ret_fwd = data["asset_ret_fwd"]
    rfr_daily = data["rfr_daily"]
    dpdf = data["dpdf"]

    # Compute z-scores with Harvey params
    z_data = compute_zscore_variables(
        data["macro"],
        change_lookback=config.harvey_change_lookback,
        zscore_window=config.harvey_zscore_window,
    )
    z_clean = z_data[[c for c in z_data.columns if c.endswith("_z")]].dropna()

    # Compute trend scores with custom SMA periods
    n_smas = len(config.sma_periods)
    trend_df = _compute_trend_scores_custom(monthly_prices, config.sma_periods)

    # Monthly SMAs for breach checks (always use the config's SMA periods)
    mc = dpdf.resample("MS").last()
    sma_dfs = [mc.rolling(p, min_periods=p).mean() for p in config.sma_periods]

    # Pre-backtest seeds for median
    bt_start = pd.Timestamp(config.bt_start)
    pre_ers = {"IVV": [], "QQQ": []}
    for z_dt in z_clean.index[z_clean.index < bt_start]:
        try:
            sim, _ = find_similar_months(
                z_clean, z_dt,
                exclude_months=config.harvey_exclude_months,
                similar_pctl=config.harvey_similar_pctl,
            )
            er = compute_expected_returns(sim, asset_ret_fwd, ["IVV", "QQQ"])
            for a in ["IVV", "QQQ"]:
                if not np.isnan(er.get(a, np.nan)):
                    pre_ers[a].append(er[a])
        except Exception:
            pass

    # State
    tier = 0
    delevered = False
    hist = {"IVV": list(pre_ers["IVV"]), "QQQ": list(pre_ers["QQQ"])}
    med = dict(config.seed)
    events = []

    ct = {a: n_smas for a in ASSETS if a != "cash"}
    he = {a: 0.0 for a in ASSETS}
    wp = dict(config.baseline)

    strat_returns = {}

    common_start = max(daily_ret.dropna(how="all").index.min(), bt_start)
    bt_end = pd.Timestamp(config.bt_end) if config.bt_end else daily_ret.index.max()
    trading_days = daily_ret.loc[common_start:bt_end].index

    if verbose:
        print(f"Backtest: {len(trading_days)} days ({common_start.date()} to {trading_days.max().date()})")

    for day in trading_days:
        if day not in daily_ret.index:
            continue
        dr = daily_ret.loc[day]
        avail = [a for a in ASSETS if a in dr.index and pd.notna(dr[a])]
        if len(avail) < 3:
            continue
        actual = {a: float(dr[a]) for a in avail}
        rfr = float(rfr_daily.get(day, 0))

        is_ms = (day == trading_days[0] or
                 day.month != trading_days[trading_days.get_loc(day) - 1].month)
        is_friday = day.weekday() == 4

        if is_ms:
            # Trend scores
            ts_c = trend_df.index[trend_df.index <= day]
            if len(ts_c) > 0:
                for a in ASSETS:
                    if a == "cash":
                        continue
                    if a in trend_df.columns:
                        v = trend_df.loc[ts_c[-1], a]
                        ct[a] = int(v) if pd.notna(v) else 0

            # Harvey expected returns
            z_c = z_clean.index[z_clean.index < day]
            if len(z_c) > 0:
                try:
                    sim, _ = find_similar_months(
                        z_clean, z_c[-1],
                        exclude_months=config.harvey_exclude_months,
                        similar_pctl=config.harvey_similar_pctl,
                    )
                    he = compute_expected_returns(sim, asset_ret_fwd, avail)
                except Exception:
                    pass

            # Realized vols
            rv_c = rvol_monthly.index[rvol_monthly.index <= day]
            rvols = {}
            if len(rv_c) > 0:
                for a in ASSETS:
                    if a == "cash":
                        continue
                    if a in rvol_monthly.columns:
                        v = rvol_monthly.loc[rv_c[-1], a]
                        rvols[a] = float(v) if pd.notna(v) and v > 0 else 0.15

            # Two-step allocation
            w1, pool = _apply_faber_custom(ct, config.baseline,
                                            config.faber_full_threshold, config.faber_partial_mult)
            w2, _ = direct_capital(dict(w1), pool, he, rvols)
            wp = normalize(w2)

            # Conviction
            f_c = ct.get("IVV", 0) >= n_smas and ct.get("QQQ", 0) >= n_smas
            h_c = he.get("IVV", 0) > 0 and he.get("QQQ", 0) > 0
            conv = f_c and h_c

            delevered = False
            if conv:
                above = all(he.get(a, 0) > med.get(a, 0) for a in ["IVV", "QQQ"])
                tier = 2 if above else 1
                for a in ["IVV", "QQQ"]:
                    hist[a].append(he.get(a, 0))
                    med[a] = float(np.median(hist[a]))
            else:
                tier = 0

        # Weekly de-lever check
        if is_friday and config.weekly_delever_threshold > 0:
            ivv_breaches = _count_sma_breaches("IVV", day, dpdf, sma_dfs)
            qqq_breaches = _count_sma_breaches("QQQ", day, dpdf, sma_dfs)

            if tier > 0 and not delevered:
                if (ivv_breaches >= config.weekly_delever_threshold or
                        qqq_breaches >= config.weekly_delever_threshold):
                    events.append({"date": day, "tier": tier})
                    tier = 0
                    delevered = True

        # Daily return
        iw = wp.get("IVV", 0)
        qw = wp.get("QQQ", 0)
        ir = actual.get("IVV", 0)
        qr = actual.get("QQQ", 0)
        sr = config.leverage_ratio * ir - (config.leverage_ratio - 1) * rfr - config.sso_expense / 252
        ql = config.leverage_ratio * qr - (config.leverage_ratio - 1) * rfr - config.qld_expense / 252
        base = sum(wp.get(a, 0) * actual.get(a, 0) for a in avail if a not in ["IVV", "QQQ"])

        sub = config.tier_subs.get(tier, 0)
        if sub > 0:
            strat_returns[day] = (iw * (1 - sub) * ir + iw * sub * sr +
                                   qw * (1 - sub) * qr + qw * sub * ql + base)
        else:
            strat_returns[day] = iw * ir + qw * qr + base

    daily_series = pd.Series(strat_returns).sort_index()
    met = compute_metrics(daily_series)

    return BacktestResult(
        daily_returns=daily_series,
        ann_return=met["ann_return"],
        ann_vol=met["ann_vol"],
        sharpe=met["sharpe"],
        sortino=met["sortino"],
        max_dd=met["max_dd"],
        calmar=met["calmar"],
        terminal_value=met["terminal_value"],
        n_delever_events=len(events),
        config=config,
    )


def run_benchmark(data: dict = None, bt_start: str = "2002-01-01",
                  bt_end: str = None) -> BacktestResult:
    """Run unlevered P1 1x benchmark (no leverage tiers)."""
    cfg = BacktestConfig(
        tier_subs={0: 0.0, 1: 0.0, 2: 0.0},
        weekly_delever_threshold=0,
        bt_start=bt_start,
        bt_end=bt_end,
    )
    return run_backtest(cfg, data)


def run_ivv_buyhold(data: dict = None, bt_start: str = "2002-01-01",
                    bt_end: str = None) -> BacktestResult:
    """Run IVV buy-and-hold benchmark."""
    if data is None:
        data = _load_data()

    daily_ret = data["daily_ret"]
    common_start = max(daily_ret.dropna(how="all").index.min(), pd.Timestamp(bt_start))
    bt_end_ts = pd.Timestamp(bt_end) if bt_end else daily_ret.index.max()
    trading_days = daily_ret.loc[common_start:bt_end_ts].index

    ivv_rets = {}
    for day in trading_days:
        if day in daily_ret.index and "IVV" in daily_ret.columns:
            v = daily_ret.loc[day, "IVV"]
            if pd.notna(v):
                ivv_rets[day] = float(v)

    daily_series = pd.Series(ivv_rets).sort_index()
    met = compute_metrics(daily_series)

    return BacktestResult(
        daily_returns=daily_series,
        ann_return=met["ann_return"],
        ann_vol=met["ann_vol"],
        sharpe=met["sharpe"],
        sortino=met["sortino"],
        max_dd=met["max_dd"],
        calmar=met["calmar"],
        terminal_value=met["terminal_value"],
        n_delever_events=0,
        config=BacktestConfig(bt_start=bt_start, bt_end=bt_end),
    )
