"""Harvey-conditional vertical spread backtest during Faber cash periods.

Deploys a fraction of Faber's idle cash as margin on vertical credit spreads.
Harvey macro similarity engine selects direction (put vs call spreads).
Uses real SPY options chain data 2010-2023.

Four strategies:
1. FABER-ONLY: production system, cash earns T-bill
2. FABER-TBILL: same, explicitly tracks T-bill on cash pool
3. FABER-UNCONDITIONAL: spreads when VIX>18 + Faber cash, always sell puts
4. FABER-HARVEY: full conditional — Harvey selects put vs call direction
"""

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from dotenv import load_dotenv; load_dotenv()

import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

from taa.data import (load_daily_etf_returns, load_monthly_prices,
                       load_monthly_macro, load_monthly_asset_returns)
from taa.faber import compute_trend_scores, apply_faber_filter
from taa.harvey import compute_zscore_variables, find_similar_months, compute_expected_returns

BASELINE = {"IVV": 0.45, "QQQ": 0.25, "VGLT": 0.05, "IAU": 0.10, "DBC": 0.05, "cash": 0.10}
ASSETS = list(BASELINE.keys())
DAILY_SMA_PERIODS = [126, 200, 252]

# Options params
VIX_THRESHOLD = 18
HARVEY_THRESHOLD = 0.005  # |ER| >= 0.5% monthly
MAX_CONTRACTS = 20
MARGIN_PCT = 0.30  # deploy 30% of cash pool as margin
SLIPPAGE = 0.10    # 10% worse than mid
COMMISSION = 2.60  # per spread (4 legs × $0.65)
TARGET_DTE_MIN = 21
TARGET_DTE_MAX = 30
TARGET_DTE_IDEAL = 25
SHORT_DELTA_PUT = -0.10
LONG_DELTA_PUT = -0.05
SHORT_DELTA_CALL = 0.10
LONG_DELTA_CALL = 0.05
MIN_OI = 100
MAX_BID_ASK = 0.05
MIN_BID = 0.10


# ── Data loading ─────────────────────────────────────────────────────────────

def load_all_data():
    import yfinance as yf

    daily_ret = load_daily_etf_returns()
    daily_ret = daily_ret[[c for c in daily_ret.columns if c in ASSETS]]
    monthly_prices = load_monthly_prices()

    # Daily prices + SMAs
    ticker_map = {"IVV": "SPY", "QQQ": "QQQ", "VGLT": "TLT", "IAU": "GLD", "DBC": "DBC"}
    dp = {}
    for our, ticker in ticker_map.items():
        d = yf.download(ticker, start="1998-01-01", progress=False)
        if d is not None and not d.empty:
            p = d["Close"]
            if hasattr(p, "columns"): p = p.iloc[:, 0]
            p.index = pd.to_datetime(p.index).tz_localize(None)
            dp[our] = p
    dpdf = pd.DataFrame(dp).sort_index()
    daily_smas = {p: dpdf.rolling(p, min_periods=p).mean() for p in DAILY_SMA_PERIODS}

    # Harvey data
    macro = load_monthly_macro()
    z_data = compute_zscore_variables(macro)
    z_cols = [c for c in z_data.columns if c.endswith("_z")]
    z_clean = z_data[z_cols].dropna()
    asset_ret = load_monthly_asset_returns()
    asset_ret_fwd = asset_ret.shift(-1)

    # VIX
    from fredapi import Fred
    key = os.environ.get("FRED_API_KEY")
    vix_monthly = None
    tbill_monthly = None
    if key:
        f = Fred(api_key=key)
        try:
            vix = f.get_series("VIXCLS", observation_start="2009-01-01")
            vix.index = pd.to_datetime(vix.index)
            vix_monthly = vix.resample("MS").last().dropna()
        except Exception:
            pass
        try:
            tb = f.get_series("DTB3", observation_start="2009-01-01")
            tb.index = pd.to_datetime(tb.index)
            tbill_monthly = tb.resample("MS").last().dropna() / 100 / 12
        except Exception:
            pass

    # Fall back to yfinance VIX if FRED failed
    if vix_monthly is None:
        vd = yf.download("^VIX", start="2009-01-01", progress=False)
        if vd is not None and not vd.empty:
            p = vd["Close"]
            if hasattr(p, "columns"): p = p.iloc[:, 0]
            p.index = pd.to_datetime(p.index).tz_localize(None)
            vix_monthly = p.resample("MS").last().dropna()

    return (daily_ret, dpdf, daily_smas, z_clean, asset_ret_fwd,
            vix_monthly, tbill_monthly)


def load_options_chain(year):
    """Load SPY options for a given year."""
    path = Path(f"data/processed/spy_options_{year}.parquet")
    if not path.exists():
        return None
    return pd.read_parquet(path)


# ── Faber signals ────────────────────────────────────────────────────────────

def sma_scores(day, dpdf, smas):
    scores = {}
    for a in ["IVV", "QQQ", "VGLT", "IAU", "DBC"]:
        if a not in dpdf.columns: scores[a] = 0; continue
        p = dpdf.loc[:day, a]
        if len(p) == 0 or pd.isna(p.iloc[-1]): scores[a] = 0; continue
        price = p.iloc[-1]; sc = 0
        for per in DAILY_SMA_PERIODS:
            s = smas[per].loc[:day, a]
            if len(s) > 0 and pd.notna(s.iloc[-1]) and price > s.iloc[-1]: sc += 1
        scores[a] = sc
    return scores


def compute_faber_cash_pool(scores):
    """Compute what fraction of the portfolio is in cash due to Faber filter."""
    pool = 0.0
    for a, base_w in BASELINE.items():
        if a == "cash": continue
        sc = scores.get(a, 0)
        if sc >= 3: pass
        elif sc == 2: pool += base_w * 0.30
        else: pool += base_w
    return pool


# ── Options chain filtering and spread selection ─────────────────────────────

def find_best_contract(chain, option_type, target_delta, dte_min, dte_max, dte_ideal):
    """Find contract closest to target delta within DTE window."""
    filt = chain[
        (chain["option_type"] == option_type) &
        (chain["dte"] >= dte_min) &
        (chain["dte"] <= dte_max) &
        # OI filter disabled: OptionsDX data has OI=0 for all contracts
        # (chain["open_interest"] >= MIN_OI) &
        (chain["bid_ask_pct"] <= MAX_BID_ASK) &
        (chain["bid"] >= MIN_BID) &
        (chain["delta"].notna())
    ].copy()

    if len(filt) == 0:
        return None

    # Prefer expiry closest to ideal DTE
    filt["dte_dist"] = abs(filt["dte"] - dte_ideal)
    best_dte = filt["dte_dist"].min()
    filt = filt[filt["dte_dist"] <= best_dte + 3]  # within 3 days of ideal

    # Find closest delta
    filt["delta_dist"] = abs(filt["delta"] - target_delta)
    best = filt.loc[filt["delta_dist"].idxmin()]
    return best


def select_spread(chain, direction, trade_date):
    """Select a vertical spread from the chain.

    direction: 'PUT' or 'CALL'
    Returns: dict with short/long strikes, mid prices, expiry, or None.
    """
    day_chain = chain[chain["trade_date"] == trade_date]
    if len(day_chain) == 0:
        return None

    if direction == "PUT":
        short = find_best_contract(day_chain, "P", SHORT_DELTA_PUT,
                                    TARGET_DTE_MIN, TARGET_DTE_MAX, TARGET_DTE_IDEAL)
        if short is None:
            return None
        # Long put: same expiry, lower strike, closer to 0 delta
        same_expiry = day_chain[
            (day_chain["option_type"] == "P") &
            (day_chain["expiry"] == short["expiry"]) &
            (day_chain["strike"] < short["strike"]) &
            (day_chain["delta"].notna())
        ]
        if len(same_expiry) == 0:
            return None
        same_expiry = same_expiry.copy()
        same_expiry["delta_dist"] = abs(same_expiry["delta"] - LONG_DELTA_PUT)
        long = same_expiry.loc[same_expiry["delta_dist"].idxmin()]

    else:  # CALL
        short = find_best_contract(day_chain, "C", SHORT_DELTA_CALL,
                                    TARGET_DTE_MIN, TARGET_DTE_MAX, TARGET_DTE_IDEAL)
        if short is None:
            return None
        same_expiry = day_chain[
            (day_chain["option_type"] == "C") &
            (day_chain["expiry"] == short["expiry"]) &
            (day_chain["strike"] > short["strike"]) &
            (day_chain["delta"].notna())
        ]
        if len(same_expiry) == 0:
            return None
        same_expiry = same_expiry.copy()
        same_expiry["delta_dist"] = abs(same_expiry["delta"] - LONG_DELTA_CALL)
        long = same_expiry.loc[same_expiry["delta_dist"].idxmin()]

    net_credit = float(short["mid"] - long["mid"])
    if net_credit <= 0:
        return None

    spread_width = abs(float(short["strike"]) - float(long["strike"]))
    if spread_width <= 0:
        return None

    return {
        "short_strike": float(short["strike"]),
        "long_strike": float(long["strike"]),
        "short_mid": float(short["mid"]),
        "long_mid": float(long["mid"]),
        "net_credit": net_credit,
        "spread_width": spread_width,
        "expiry": pd.Timestamp(short["expiry"]),
        "dte": int(short["dte"]),
        "short_delta": float(short["delta"]),
        "long_delta": float(long["delta"]),
        "underlying": float(short["underlying_close"]),
    }


def simulate_spread_outcome(spread, direction, expiry_price):
    """Compute P&L for a spread held to expiry."""
    actual_credit = spread["net_credit"] * (1 - SLIPPAGE)
    K_short = spread["short_strike"]
    K_long = spread["long_strike"]
    w = spread["spread_width"]

    if direction == "PUT":
        if expiry_price >= K_short:
            profit_per_share = actual_credit
        elif expiry_price <= K_long:
            profit_per_share = actual_credit - w
        else:
            profit_per_share = actual_credit - (K_short - expiry_price)
    else:  # CALL
        if expiry_price <= K_short:
            profit_per_share = actual_credit
        elif expiry_price >= K_long:
            profit_per_share = actual_credit - w
        else:
            profit_per_share = actual_credit - (expiry_price - K_short)

    return profit_per_share


# ── Main backtest ────────────────────────────────────────────────────────────

def run_backtest(daily_ret, dpdf, daily_smas, z_clean, asset_ret_fwd,
                 vix_monthly, tbill_monthly):
    print(f"\n{'='*120}")
    print(f"  BACKTEST: Harvey-Conditional Vertical Spreads (2010-2023)")
    print(f"{'='*120}")

    bt_start = pd.Timestamp("2010-01-01")
    bt_end = pd.Timestamp("2023-12-31")

    # Pre-load all options data
    options_by_year = {}
    for yr in range(2010, 2024):
        df = load_options_chain(yr)
        if df is not None:
            options_by_year[yr] = df
            print(f"  Loaded {yr}: {len(df):,} rows, {df['trade_date'].nunique()} dates")

    # Get month-end dates in range
    trading_days = dpdf.index[(dpdf.index >= bt_start) & (dpdf.index <= bt_end)]
    month_ends = []
    for i, day in enumerate(trading_days):
        if i == len(trading_days) - 1 or day.month != trading_days[i + 1].month:
            month_ends.append(day)

    print(f"  Month-ends to evaluate: {len(month_ends)}")

    # Track outcomes
    all_trades = []
    activation_log = {
        "total": 0, "no_cash": 0, "vix_low": 0, "harvey_ambiguous": 0,
        "no_contracts": 0, "opened": 0, "puts": 0, "calls": 0,
    }

    # Simulated portfolio value (start $100K for round numbers)
    portfolio_value = 100000.0

    for me_date in month_ends:
        activation_log["total"] += 1
        year = me_date.year

        # ── Condition 1: Faber cash pool ─────────────────────────────────
        prior = trading_days[trading_days < me_date]
        sd = prior[-1] if len(prior) > 0 else me_date
        scores = sma_scores(sd, dpdf, daily_smas)
        cash_pool = compute_faber_cash_pool(scores)

        if cash_pool < 0.20:
            activation_log["no_cash"] += 1
            continue

        # ── Condition 2: VIX elevated ────────────────────────────────────
        vix_val = None
        if vix_monthly is not None:
            vix_dates = vix_monthly.index[vix_monthly.index <= me_date]
            if len(vix_dates) > 0:
                vix_val = float(vix_monthly.loc[vix_dates[-1]])

        if vix_val is None or vix_val < VIX_THRESHOLD:
            activation_log["vix_low"] += 1
            continue

        # ── Condition 3: Harvey directional signal ───────────────────────
        z_prior = z_clean.index[z_clean.index < me_date]
        harvey_er = 0.0
        if len(z_prior) > 0:
            try:
                sim, _ = find_similar_months(z_clean, z_prior[-1])
                er = compute_expected_returns(sim, asset_ret_fwd, ["IVV"])
                harvey_er = er.get("IVV", 0.0)
            except ValueError:
                pass

        if abs(harvey_er) < HARVEY_THRESHOLD:
            activation_log["harvey_ambiguous"] += 1
            # For UNCONDITIONAL strategy, we'd still open here — tracked separately
            harvey_direction = None
        elif harvey_er > 0:
            harvey_direction = "PUT"  # recovery → sell puts
        else:
            harvey_direction = "CALL"  # bear → sell calls

        # ── Condition 4: Find qualifying contracts ───────────────────────
        chain = options_by_year.get(year)
        if chain is None:
            activation_log["no_contracts"] += 1
            continue

        # Find closest trade date in chain to month-end
        chain_dates = chain["trade_date"].unique()
        valid_dates = chain_dates[chain_dates <= me_date]
        if len(valid_dates) == 0:
            activation_log["no_contracts"] += 1
            continue
        trade_date = valid_dates[-1]

        # Try to select spread for UNCONDITIONAL (always puts)
        uncond_spread = select_spread(chain, "PUT", trade_date)

        # Try to select spread for HARVEY direction
        harvey_spread = None
        if harvey_direction is not None:
            harvey_spread = select_spread(chain, harvey_direction, trade_date)

        if uncond_spread is None and harvey_spread is None:
            activation_log["no_contracts"] += 1
            continue

        # ── Position sizing ──────────────────────────────────────────────
        available_margin = portfolio_value * cash_pool * MARGIN_PCT

        # ── Record trades ────────────────────────────────────────────────
        # Find expiry price
        for spread, direction, strat_label in [
            (uncond_spread, "PUT", "UNCONDITIONAL"),
            (harvey_spread, harvey_direction, "HARVEY"),
        ]:
            if spread is None:
                continue

            max_loss_per = (spread["spread_width"] - spread["net_credit"]) * 100
            if max_loss_per <= 0:
                continue
            contracts = min(int(available_margin / max_loss_per), MAX_CONTRACTS)
            contracts = max(contracts, 1)

            # Find expiry price from options chain or daily prices
            expiry_dt = spread["expiry"]
            expiry_price = None

            # Check options chain for underlying at expiry
            exp_year = expiry_dt.year
            exp_chain = options_by_year.get(exp_year)
            if exp_chain is not None:
                exp_rows = exp_chain[exp_chain["trade_date"] == expiry_dt]
                if len(exp_rows) > 0:
                    expiry_price = float(exp_rows.iloc[0]["underlying_close"])

            # Fall back to daily prices
            if expiry_price is None:
                exp_prices = dpdf.loc[:expiry_dt, "IVV"]
                if len(exp_prices) > 0:
                    expiry_price = float(exp_prices.iloc[-1])

            if expiry_price is None:
                continue

            profit_per_share = simulate_spread_outcome(spread, direction, expiry_price)
            net_pnl = (profit_per_share * contracts * 100) - (COMMISSION * contracts)

            all_trades.append({
                "date": me_date,
                "trade_date": trade_date,
                "strategy": strat_label,
                "direction": direction,
                "short_strike": spread["short_strike"],
                "long_strike": spread["long_strike"],
                "spread_width": spread["spread_width"],
                "net_credit": spread["net_credit"],
                "expiry": expiry_dt,
                "dte": spread["dte"],
                "underlying_entry": spread["underlying"],
                "underlying_expiry": expiry_price,
                "contracts": contracts,
                "profit_per_share": profit_per_share,
                "net_pnl": net_pnl,
                "margin_used": max_loss_per * contracts,
                "cash_pool": cash_pool,
                "vix": vix_val,
                "harvey_er": harvey_er,
                "harvey_direction": harvey_direction,
                "market_moved": (expiry_price / spread["underlying"]) - 1,
            })

        if harvey_spread is not None:
            activation_log["opened"] += 1
            if harvey_direction == "PUT":
                activation_log["puts"] += 1
            else:
                activation_log["calls"] += 1

    trades_df = pd.DataFrame(all_trades)
    return trades_df, activation_log, month_ends


# ── Report ───────────────────────────────────────────────────────────────────

def report(trades_df, activation_log, month_ends, tbill_monthly):
    total = activation_log["total"]

    # ── Activation frequency ─────────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  ACTIVATION FREQUENCY")
    print(f"{'='*120}")

    print(f"\n  Total month-ends evaluated:              {total}")
    print(f"  Months Faber had <20% cash pool:         {activation_log['no_cash']} ({activation_log['no_cash']/total*100:.0f}%) — spreads impossible")
    print(f"  Months VIX below {VIX_THRESHOLD}:                    {activation_log['vix_low']} ({activation_log['vix_low']/total*100:.0f}%) — thin premium")
    print(f"  Months Harvey ambiguous:                 {activation_log['harvey_ambiguous']} ({activation_log['harvey_ambiguous']/total*100:.0f}%) — no clear direction")
    print(f"  Months no qualifying contracts:          {activation_log['no_contracts']} ({activation_log['no_contracts']/total*100:.0f}%) — chain filter failed")
    print(f"  Months spread opened (HARVEY):           {activation_log['opened']} ({activation_log['opened']/total*100:.0f}%)")

    if activation_log["opened"] > 0:
        print(f"\n  Of months opened:")
        print(f"    PUT spreads (recovery regime):         {activation_log['puts']} ({activation_log['puts']/activation_log['opened']*100:.0f}%)")
        print(f"    CALL spreads (bear regime):            {activation_log['calls']} ({activation_log['calls']/activation_log['opened']*100:.0f}%)")

    if len(trades_df) == 0:
        print(f"\n  NO TRADES EXECUTED — cannot compute performance.")
        return

    # ── Harvey signal accuracy ───────────────────────────────────────────
    harvey_trades = trades_df[trades_df["strategy"] == "HARVEY"]
    uncond_trades = trades_df[trades_df["strategy"] == "UNCONDITIONAL"]

    if len(harvey_trades) > 0:
        put_trades_h = harvey_trades[harvey_trades["direction"] == "PUT"]
        call_trades_h = harvey_trades[harvey_trades["direction"] == "CALL"]

        put_correct = (put_trades_h["market_moved"] > 0).sum() if len(put_trades_h) > 0 else 0
        call_correct = (call_trades_h["market_moved"] < 0).sum() if len(call_trades_h) > 0 else 0

        total_correct = put_correct + call_correct
        total_harvey = len(harvey_trades)

        print(f"\n  Harvey signal accuracy:")
        if len(put_trades_h) > 0:
            print(f"    PUT spread opened, market went up:     {put_correct}/{len(put_trades_h)} ({put_correct/len(put_trades_h)*100:.0f}%)")
        if len(call_trades_h) > 0:
            print(f"    CALL spread opened, market went down:  {call_correct}/{len(call_trades_h)} ({call_correct/len(call_trades_h)*100:.0f}%)")
        print(f"    Overall directional accuracy:           {total_correct}/{total_harvey} ({total_correct/total_harvey*100:.0f}%)")

    # ── Performance by strategy ──────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  PERFORMANCE BY STRATEGY")
    print(f"{'='*120}")

    for strat in ["UNCONDITIONAL", "HARVEY"]:
        st = trades_df[trades_df["strategy"] == strat]
        if len(st) == 0:
            print(f"\n  {strat}: No trades")
            continue

        wins = st[st["net_pnl"] > 0]
        losses = st[st["net_pnl"] <= 0]
        win_rate = len(wins) / len(st) * 100

        print(f"\n  {strat} ({len(st)} trades):")
        print(f"    Win rate:              {win_rate:.0f}%")
        print(f"    Avg profit per win:    ${wins['net_pnl'].mean():,.0f}" if len(wins) > 0 else "    Avg profit per win:    N/A")
        print(f"    Avg loss per loss:     ${losses['net_pnl'].mean():,.0f}" if len(losses) > 0 else "    Avg loss per loss:     N/A")
        print(f"    Total net P&L:         ${st['net_pnl'].sum():,.0f}")
        print(f"    Avg net P&L per trade: ${st['net_pnl'].mean():,.0f}")
        print(f"    Avg contracts:         {st['contracts'].mean():.1f}")
        print(f"    Avg margin per trade:  ${st['margin_used'].mean():,.0f}")
        print(f"    Avg premium collected: ${(st['net_credit'] * st['contracts'] * 100).mean():,.0f}")

        # Annualize
        years = 14.0
        ann_pnl = st["net_pnl"].sum() / years
        print(f"    Annual P&L:            ${ann_pnl:,.0f}")
        print(f"    As % of $100K port:    {ann_pnl/100000*100:.2f}%")

    # ── Harvey vs Unconditional comparison ───────────────────────────────
    print(f"\n{'='*120}")
    print(f"  HARVEY DIRECTIONAL VALUE-ADD")
    print(f"{'='*120}")

    if len(uncond_trades) > 0 and len(harvey_trades) > 0:
        u_wr = (uncond_trades["net_pnl"] > 0).mean() * 100
        h_wr = (harvey_trades["net_pnl"] > 0).mean() * 100
        u_avg = uncond_trades["net_pnl"].mean()
        h_avg = harvey_trades["net_pnl"].mean()
        u_total = uncond_trades["net_pnl"].sum()
        h_total = harvey_trades["net_pnl"].sum()

        print(f"\n  {'':>25} {'UNCONDITIONAL':>15} {'HARVEY':>15}")
        print(f"  {'-'*25} {'-'*15} {'-'*15}")
        print(f"  {'Trades':>25} {len(uncond_trades):>15} {len(harvey_trades):>15}")
        print(f"  {'Win rate':>25} {u_wr:>14.0f}% {h_wr:>14.0f}%")
        print(f"  {'Avg P&L per trade':>25} ${u_avg:>13,.0f} ${h_avg:>13,.0f}")
        print(f"  {'Total P&L':>25} ${u_total:>13,.0f} ${h_total:>13,.0f}")
        print(f"  {'Annual P&L':>25} ${u_total/14:>13,.0f} ${h_total/14:>13,.0f}")
        print(f"  {'Ann % of $100K':>25} {u_total/14/100000*100:>14.2f}% {h_total/14/100000*100:>14.2f}%")

        print(f"\n  Does Harvey's directional signal add value? ", end="")
        if h_total > u_total:
            print(f"YES — +${h_total - u_total:,.0f} over 14 years")
        else:
            print(f"NO — -${u_total - h_total:,.0f} vs unconditional")

    # ── Crisis period analysis ───────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  CRISIS PERIOD ANALYSIS")
    print(f"{'='*120}")

    crises = [
        ("2011 correction", "2011-07", "2011-10"),
        ("2015-2016", "2015-08", "2016-02"),
        ("2018 Q4", "2018-10", "2018-12"),
        ("COVID Feb-Apr 2020", "2020-02", "2020-04"),
        ("2022 Bear", "2022-01", "2022-10"),
    ]

    for cname, cs, ce in crises:
        for strat in ["HARVEY", "UNCONDITIONAL"]:
            ct = trades_df[(trades_df["strategy"] == strat) &
                           (trades_df["date"] >= pd.Timestamp(cs)) &
                           (trades_df["date"] <= pd.Timestamp(ce))]
            if len(ct) > 0:
                total_pnl = ct["net_pnl"].sum()
                dirs = ct["direction"].value_counts().to_dict()
                wins = (ct["net_pnl"] > 0).sum()
                print(f"\n  {cname} — {strat}:")
                print(f"    Trades: {len(ct)}, Directions: {dirs}, Wins: {wins}/{len(ct)}, Net: ${total_pnl:,.0f}")
                for _, t in ct.iterrows():
                    print(f"      {t['date'].strftime('%Y-%m')}: {t['direction']} {t['short_strike']:.0f}/{t['long_strike']:.0f} "
                          f"cr=${t['net_credit']:.2f} → ${t['net_pnl']:+,.0f} "
                          f"(SPY {t['underlying_entry']:.0f}→{t['underlying_expiry']:.0f})")
            else:
                # Check if Faber was in cash but VIX/Harvey filtered out
                crisis_months = [d for d in month_ends
                                 if pd.Timestamp(cs) <= d <= pd.Timestamp(ce)]
                if crisis_months and strat == "HARVEY":
                    print(f"\n  {cname} — {strat}: No trades (filtered out by conditions)")

    # ── Position sizing reality check ────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  POSITION SIZING REALITY CHECK ($100K portfolio)")
    print(f"{'='*120}")

    for strat in ["UNCONDITIONAL", "HARVEY"]:
        st = trades_df[trades_df["strategy"] == strat]
        if len(st) == 0: continue
        years = 14.0
        print(f"\n  {strat}:")
        print(f"    Avg contracts per trade:    {st['contracts'].mean():.1f}")
        print(f"    Avg margin per trade:       ${st['margin_used'].mean():,.0f}")
        print(f"    Avg premium collected:      ${(st['net_credit'] * st['contracts'] * 100).mean():,.0f}")
        print(f"    Avg net profit per trade:   ${st['net_pnl'].mean():,.0f}")
        print(f"    Annual trades:              {len(st)/years:.1f}")
        print(f"    Annual premium income:      ${(st['net_credit'] * st['contracts'] * 100).sum()/years:,.0f}")
        print(f"    Annual net income:          ${st['net_pnl'].sum()/years:,.0f}")
        print(f"    As % of portfolio:          {st['net_pnl'].sum()/years/100000*100:.2f}%")

    # ── Key questions ────────────────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  KEY QUESTIONS")
    print(f"{'='*120}")

    print(f"\n  Q1. Activation rate: {activation_log['opened']}/{total} months ({activation_log['opened']/total*100:.0f}%)")
    print(f"      Biggest filter: {'VIX<18' if activation_log['vix_low'] > activation_log['no_cash'] else 'No Faber cash'}")

    if len(harvey_trades) > 0 and len(uncond_trades) > 0:
        h_wr = (harvey_trades["net_pnl"] > 0).mean()
        u_wr = (uncond_trades["net_pnl"] > 0).mean()
        print(f"  Q2. Harvey win rate: {h_wr:.0%} vs unconditional: {u_wr:.0%}")

        # Q3 COVID
        covid_h = harvey_trades[(harvey_trades["date"] >= "2020-02-01") & (harvey_trades["date"] <= "2020-05-01")]
        if len(covid_h) == 0:
            print(f"  Q3. COVID: No trades — filters kept us out")
        else:
            print(f"  Q3. COVID: {len(covid_h)} trades, net ${covid_h['net_pnl'].sum():+,.0f}")

        h_ann = harvey_trades["net_pnl"].sum() / 14
        print(f"  Q5. Annual premium (HARVEY): ${h_ann:,.0f} ({h_ann/100000:.2%} of $100K)")

        print(f"\n  Q6. Final verdict: ", end="")
        if h_ann > 0 and harvey_trades["net_pnl"].sum() > uncond_trades["net_pnl"].sum():
            print(f"Harvey-conditional spreads ADD value over unconditional")
        elif h_ann > 0:
            print(f"Spreads add value but Harvey direction doesn't improve on always-puts")
        else:
            print(f"Options pod does NOT add value — negative P&L")

    print()


if __name__ == "__main__":
    print("=" * 120)
    print("  HARVEY-CONDITIONAL VERTICAL SPREAD BACKTEST (2010-2023)")
    print("=" * 120)

    print(f"\n  Loading data...")
    (daily_ret, dpdf, daily_smas, z_clean, asset_ret_fwd,
     vix_monthly, tbill_monthly) = load_all_data()

    trades_df, activation_log, month_ends = run_backtest(
        daily_ret, dpdf, daily_smas, z_clean, asset_ret_fwd,
        vix_monthly, tbill_monthly)

    report(trades_df, activation_log, month_ends, tbill_monthly)
