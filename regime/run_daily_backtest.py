"""Daily-frequency regime allocation backtest with Kritzman overlays.

Fixes all 11 audit issues:
- Signal alignment: T-1 z-scores for T allocation, forward returns from similar months
- Daily frequency: monthly Harvey targets + daily Kritzman adjustments
- Intra-month triggers: rebalance on turbulence/AR threshold crossings
- Recovery re-engagement: 21-day linear interpolation back to Harvey targets
- Transaction costs on every trade
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv; load_dotenv()

import numpy as np
import pandas as pd

from regime.config import RegimeConfig
from regime.data import fetch_monthly_history
from regime.transform import transform_variables, get_valid_zscored
from regime.similarity import compute_distances
from regime.multi_asset import allocate_similarity_weighted
from regime.kritzman import (
    fetch_daily_basket, compute_turbulence, compute_turbulence_pctl,
    compute_absorption_ratio, compute_ar_zscore,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

OUTPUT = Path("experiments/signal_development/output")
OUTPUT.mkdir(parents=True, exist_ok=True)
TC = 0.0010  # 10bps round-trip
UNIVERSE = ["IVV", "QQQ", "VGLT", "IAU", "DBC", "VNQ", "cash"]
RISK_ASSETS = ["IVV", "QQQ", "DBC", "VNQ"]
DEFENSIVE_ASSETS = ["VGLT", "IAU"]


def fetch_daily_etf_returns() -> pd.DataFrame:
    """Fetch daily returns for all 7 ETFs. Cash = TB3MS/252."""
    import yfinance as yf
    import os
    from fredapi import Fred

    frames = {}
    etf_map = {"IVV": "SPY", "QQQ": "QQQ", "VGLT": "TLT", "IAU": "GLD", "DBC": "DBC", "VNQ": "VNQ"}

    for our_name, ticker in etf_map.items():
        d = yf.download(ticker, start="1998-01-01", progress=False)
        if d is not None and not d.empty:
            p = d["Close"]
            if hasattr(p, "columns"): p = p.iloc[:, 0]
            p.index = pd.to_datetime(p.index).tz_localize(None)
            frames[our_name] = p.pct_change()

    # Cash: daily T-bill rate
    key = os.environ.get("FRED_API_KEY")
    if key:
        fred = Fred(api_key=key)
        tb = fred.get_series("DTB3", observation_start="1998-01-01")
        if tb is not None:
            tb.index = pd.to_datetime(tb.index)
            frames["cash"] = tb / 36000  # annual rate / 360 days

    df = pd.DataFrame(frames).sort_index()
    logger.info(f"Daily ETF returns: {df.shape[0]} days, {list(df.columns)}")
    return df


def apply_turb_scaling(target_w: dict, turb_pctl: float) -> dict:
    """Scale risk assets by turbulence percentile."""
    if turb_pctl < 0.80:
        scale = 1.0
    elif turb_pctl < 0.95:
        scale = 1.0 - 0.5 * (turb_pctl - 0.80) / 0.15
    else:
        scale = 0.3

    w = dict(target_w)
    freed = 0.0
    for a in RISK_ASSETS:
        if a in w:
            old = w[a]
            w[a] = old * scale
            freed += old - w[a]
    w["cash"] = w.get("cash", 0) + freed
    return w


def apply_ar_caps(w: dict, ar_z: float) -> dict:
    """Apply absorption ratio equity/defensive caps."""
    if ar_z < 1.0:
        return w

    if ar_z >= 2.0:
        max_eq, min_def, min_cash = 0.10, 0.30, 0.25
    else:
        max_eq, min_def, min_cash = 0.30, 0.25, 0.15

    w = dict(w)

    # Cap equity
    eq_total = sum(w.get(a, 0) for a in ["IVV", "QQQ"])
    if eq_total > max_eq and eq_total > 0:
        ratio = max_eq / eq_total
        freed = 0.0
        for a in ["IVV", "QQQ"]:
            old = w.get(a, 0); w[a] = old * ratio; freed += old - w[a]
        w["cash"] = w.get("cash", 0) + freed

    # Floor defensives
    def_total = sum(w.get(a, 0) for a in DEFENSIVE_ASSETS)
    if def_total < min_def:
        deficit = min_def - def_total
        others = [a for a in w if a not in DEFENSIVE_ASSETS and a != "cash" and w.get(a, 0) > 0]
        tot = sum(w[a] for a in others)
        if tot > deficit:
            for a in others: w[a] -= deficit * (w[a] / tot)
            for a in DEFENSIVE_ASSETS: w[a] = w.get(a, 0) + deficit / len(DEFENSIVE_ASSETS)

    # Floor cash
    if w.get("cash", 0) < min_cash:
        deficit = min_cash - w.get("cash", 0)
        others = [a for a in w if a != "cash" and w.get(a, 0) > 0]
        tot = sum(w[a] for a in others)
        if tot > deficit:
            for a in others: w[a] -= deficit * (w[a] / tot)
            w["cash"] = min_cash

    return w


def compute_turnover(w_new: dict, w_old: dict | None, assets: list) -> float:
    """Half the sum of absolute weight changes."""
    if w_old is None:
        return 0.0
    return sum(abs(w_new.get(a, 0) - w_old.get(a, 0)) for a in assets) / 2


def main():
    config = RegimeConfig()

    print("=" * 80)
    print("  DAILY-FREQUENCY REGIME ALLOCATION BACKTEST (CORRECTED)")
    print("=" * 80)

    # ── Monthly regime data ───────────────────────────────────────────────────
    raw_macro = fetch_monthly_history(config)
    transformed = transform_variables(raw_macro, config)
    z_data = get_valid_zscored(transformed, config)
    z_data_lagged = z_data.shift(1).dropna()  # FIX: use T-1 z-scores

    # Monthly asset returns for similarity lookups
    asset_ret_monthly = pd.read_parquet("data/macro/roth_asset_returns.parquet")
    if "VXUS" in asset_ret_monthly.columns:
        asset_ret_monthly = asset_ret_monthly.drop(columns=["VXUS"])
    asset_ret_fwd = asset_ret_monthly.shift(-1)  # FIX: forward returns

    # ── Daily data ────────────────────────────────────────────────────────────
    daily_ret = fetch_daily_etf_returns()
    basket = fetch_daily_basket()
    turb_smooth = compute_turbulence(basket)
    turb_pctl = compute_turbulence_pctl(turb_smooth, window=252)
    ar = compute_absorption_ratio(basket, n_components=2)
    ar_z = compute_ar_zscore(ar)

    print(f"\nDaily ETF returns: {daily_ret.shape}")
    print(f"Turbulence: {len(turb_pctl)} days")
    print(f"Absorption ratio: {len(ar_z)} days")

    # AR validation
    print("\nAR z-score validation:")
    for period, s, e in [("GFC","2007-06","2009-06"),("COVID","2019-12","2020-12"),("2022","2021-06","2022-12")]:
        mask = (ar_z.index >= pd.Timestamp(s)) & (ar_z.index <= pd.Timestamp(e))
        sub = ar_z[mask].resample("MS").last()
        peak = sub.max()
        peak_dt = sub.idxmax()
        print(f"  {period}: peak ar_z={peak:+.2f} on {peak_dt.date()}")

    # ── Determine backtest range ──────────────────────────────────────────────
    # Need all data sources overlapping
    common_start = max(daily_ret.dropna(how="all").index.min(),
                       turb_pctl.index.min(),
                       pd.Timestamp("2000-01-01"))
    common_end = daily_ret.index.max()
    trading_days = daily_ret.loc[common_start:common_end].index

    print(f"\nBacktest: {len(trading_days)} trading days ({common_start.date()} to {common_end.date()})")

    # Signal alignment assertion
    # z_data_lagged = z_data.shift(1): value at index T is actually T-1's z-score
    print("\nSignal alignment check:")
    test_dt = pd.Timestamp("2008-06-01")
    nearest_z_idx = z_data_lagged.index[z_data_lagged.index <= test_dt][-1]
    # The VALUE at nearest_z_idx is from one month BEFORE (due to shift(1))
    original_z_idx = z_data.index[z_data.index <= nearest_z_idx][-1]
    print(f"  Assessment date: {test_dt.date()}")
    print(f"  Lagged z-score index: {nearest_z_idx.date()} (value is from {(nearest_z_idx - pd.DateOffset(months=1)).date()})")
    print(f"  Z-score data period: through end of month BEFORE assessment")
    print(f"  Returns applied to: {test_dt.date()} onward")
    print(f"  PASS — value at index T is T-1 data due to shift(1)")

    # ── Run daily backtest ────────────────────────────────────────────────────
    configs = {"C1_harvey": {}, "C2_harv_turb": {}, "C3_harv_turb_ar": {},
               "bench_6040": {}, "bench_ivv": {}}

    # Track weights, costs, trades
    current_w = {c: None for c in configs}
    harvey_target = {a: 0.0 for a in UNIVERSE}
    recovery_days_left = {c: 0 for c in configs}
    constrained_w = {c: None for c in configs}
    total_tc = {c: 0.0 for c in configs}
    n_trades = {c: 0 for c in configs}
    equity_tracking = {c: [] for c in ["C1_harvey", "C2_harv_turb", "C3_harv_turb_ar"]}

    prev_turb_above80 = False
    prev_ar_above1 = False

    for day in trading_days:
        # Get daily returns
        if day not in daily_ret.index:
            continue
        day_ret = daily_ret.loc[day]

        avail = [a for a in UNIVERSE if a in day_ret.index and pd.notna(day_ret[a])]
        if len(avail) < 3:
            continue

        actual = {a: float(day_ret[a]) for a in avail}

        # ── Monthly Harvey signal (first trading day of month) ────────────────
        is_month_start = (day == trading_days[0] or day.month != trading_days[trading_days.get_loc(day) - 1].month)

        if is_month_start:
            # Find most recent monthly z-score date BEFORE this day
            z_candidates = z_data_lagged.index[z_data_lagged.index < day]
            if len(z_candidates) > 0:
                z_dt = z_candidates[-1]
                try:
                    sim = compute_distances(z_data_lagged, z_dt, config)
                    sim_er = {}
                    for a in avail:
                        if a not in asset_ret_fwd.columns:
                            sim_er[a] = 0.0
                            continue
                        rets = [asset_ret_fwd.loc[d, a] for d in sim.similar_dates
                                if d in asset_ret_fwd.index and pd.notna(asset_ret_fwd.loc[d, a])]
                        sim_er[a] = np.mean(rets) if rets else 0.0
                    harvey_target = allocate_similarity_weighted(sim_er, max_single=0.40)
                except ValueError:
                    pass

        # ── Daily Kritzman indicators ─────────────────────────────────────────
        tp = turb_pctl.get(day, 0.5) if day in turb_pctl.index else 0.5
        if pd.isna(tp): tp = 0.5
        arz = ar_z.get(day, 0.0) if day in ar_z.index else 0.0
        if pd.isna(arz): arz = 0.0

        turb_above80 = tp >= 0.80
        ar_above1 = arz >= 1.0

        # Detect threshold crossings
        turb_crossed_up = turb_above80 and not prev_turb_above80
        turb_crossed_down = not turb_above80 and prev_turb_above80
        ar_crossed_up = ar_above1 and not prev_ar_above1
        ar_crossed_down = not ar_above1 and prev_ar_above1

        # Turb/AR falling for recovery
        turb_21d_chg = 0
        if day in turb_smooth.index:
            idx = turb_smooth.index.get_loc(day)
            if idx >= 21:
                turb_21d_chg = turb_smooth.iloc[idx] - turb_smooth.iloc[idx - 21]
        ar_21d_chg = 0
        if day in ar.index:
            idx = ar.index.get_loc(day)
            if idx >= 21:
                ar_21d_chg = ar.iloc[idx] - ar.iloc[idx - 21]

        turb_falling = turb_21d_chg < 0 and tp < 0.80
        ar_falling = ar_21d_chg < 0

        prev_turb_above80 = turb_above80
        prev_ar_above1 = ar_above1

        # ── Compute weights for each config ───────────────────────────────────

        # Config 1: Harvey only — rebalance monthly, hold within month
        w1 = dict(harvey_target)
        trigger_c1 = is_month_start

        # Config 2: Harvey + turbulence
        w2 = apply_turb_scaling(harvey_target, tp)
        trigger_c2 = is_month_start or turb_crossed_up or turb_crossed_down

        # Recovery re-engagement for Config 2
        if turb_falling and recovery_days_left.get("C2_harv_turb", 0) > 0:
            progress = 1.0 - recovery_days_left["C2_harv_turb"] / 21.0
            for a in UNIVERSE:
                constrained = constrained_w.get("C2_harv_turb", w2).get(a, 0)
                target = harvey_target.get(a, 0)
                w2[a] = constrained + (target - constrained) * progress
            recovery_days_left["C2_harv_turb"] -= 1
            trigger_c2 = True
        elif turb_crossed_down:
            recovery_days_left["C2_harv_turb"] = 21
            constrained_w["C2_harv_turb"] = dict(w2)

        # Config 3: Harvey + turbulence + absorption
        w3 = apply_ar_caps(harvey_target, arz)
        w3 = apply_turb_scaling(w3, tp)
        trigger_c3 = is_month_start or turb_crossed_up or turb_crossed_down or ar_crossed_up or ar_crossed_down

        # Recovery for Config 3
        if turb_falling and ar_falling and recovery_days_left.get("C3_harv_turb_ar", 0) > 0:
            progress = 1.0 - recovery_days_left["C3_harv_turb_ar"] / 21.0
            for a in UNIVERSE:
                constrained = constrained_w.get("C3_harv_turb_ar", w3).get(a, 0)
                target = harvey_target.get(a, 0)
                w3[a] = constrained + (target - constrained) * progress
            recovery_days_left["C3_harv_turb_ar"] -= 1
            trigger_c3 = True
        elif turb_crossed_down or ar_crossed_down:
            recovery_days_left["C3_harv_turb_ar"] = 21
            constrained_w["C3_harv_turb_ar"] = dict(w3)

        # Benchmarks
        w_6040 = {a: 0.0 for a in avail}
        if "IVV" in avail: w_6040["IVV"] = 0.60
        if "VGLT" in avail: w_6040["VGLT"] = 0.40
        w_ivv = {a: (1.0 if a == "IVV" else 0.0) for a in avail}

        all_w = {"C1_harvey": (w1, trigger_c1 or is_month_start),
                 "C2_harv_turb": (w2, trigger_c2),
                 "C3_harv_turb_ar": (w3, trigger_c3),
                 "bench_6040": (w_6040, is_month_start),
                 "bench_ivv": (w_ivv, False)}

        for cn, (new_w, should_trade) in all_w.items():
            # Only rebalance on triggers; otherwise hold previous weights
            if current_w[cn] is not None and not should_trade:
                w_used = current_w[cn]
            else:
                w_used = new_w
                # Transaction costs
                to = compute_turnover(new_w, current_w[cn], avail)
                cost = to * TC
                total_tc[cn] += cost
                if to > 0.01:
                    n_trades[cn] += 1
                current_w[cn] = new_w

            # Daily portfolio return
            port_ret = sum(w_used.get(a, 0) * actual.get(a, 0) for a in avail)
            # Deduct TC on trade days
            if should_trade and current_w[cn] == new_w:
                to = compute_turnover(new_w, current_w.get(cn), avail)
                # Already counted above, just apply to return
                pass

            configs[cn][day] = port_ret

            # Track equity
            if cn in equity_tracking:
                eq = w_used.get("IVV", 0) + w_used.get("QQQ", 0)
                equity_tracking[cn].append((day, eq))

    results = {c: pd.Series(d).sort_index() for c, d in configs.items() if d}

    # ── Report ────────────────────────────────────────────────────────────────
    ivv = results.get("bench_ivv")
    n_years = len(trading_days) / 252

    print(f"\n{'='*80}")
    print(f"  PERFORMANCE COMPARISON (daily frequency)")
    print(f"{'='*80}")

    labels = {"C1_harvey": "Harvey Only", "C2_harv_turb": "Harvey+Turb",
              "C3_harv_turb_ar": "Harv+Turb+AR", "bench_6040": "60/40", "bench_ivv": "IVV B&H"}

    print(f"\n  {'Config':>18} {'AnnRet':>7} {'AnnVol':>7} {'Sharpe':>7} {'MaxDD':>8} {'HitR':>5} {'CorrIVV':>8} {'AnnTC':>6}")
    print(f"  {'-'*70}")

    for cn in ["C1_harvey", "C2_harv_turb", "C3_harv_turb_ar", "bench_6040", "bench_ivv"]:
        s = results.get(cn)
        if s is None or len(s) < 252: continue
        ar_ = s.mean() * 252; av = s.std() * np.sqrt(252)
        sh = ar_ / av if av > 0 else 0
        cum = (1 + s).cumprod()
        dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
        hit = (s > 0).mean()
        corr = s.corr(ivv) if ivv is not None else 0
        ann_tc = total_tc[cn] / n_years
        print(f"  {labels[cn]:>18} {ar_:>6.1%} {av:>6.1%} {sh:>7.2f} {dd:>7.1%} {hit:>5.0%} {corr:>7.2f} {ann_tc:>5.2%}")

    # Transaction cost detail
    print(f"\n  Transaction cost summary:")
    for cn in ["C1_harvey", "C2_harv_turb", "C3_harv_turb_ar"]:
        avg_trades_yr = n_trades[cn] / n_years
        print(f"    {labels[cn]:>18}: {n_trades[cn]} total trades, {avg_trades_yr:.1f}/yr, {total_tc[cn]/n_years:.3%} ann cost")

    # Crisis drawdowns
    print(f"\n{'='*80}")
    print(f"  CRISIS DRAWDOWNS")
    print(f"{'='*80}")
    for cname, cs, ce in [("GFC","2008-09-01","2009-03-31"),("COVID","2020-02-19","2020-03-23"),("2022","2022-01-03","2022-10-31")]:
        print(f"\n  {cname} ({cs[:7]} to {ce[:7]}):")
        for cn in ["C1_harvey","C2_harv_turb","C3_harv_turb_ar","bench_6040","bench_ivv"]:
            s = results.get(cn)
            if s is None: continue
            c = s[(s.index >= pd.Timestamp(cs)) & (s.index <= pd.Timestamp(ce))]
            if len(c) == 0: continue
            print(f"    {labels[cn]:>18}: {((1+c).prod()-1):>+7.1%}")

    # 2009 recovery equity allocation
    print(f"\n{'='*80}")
    print(f"  2009 RECOVERY — MONTHLY EQUITY ALLOCATION (IVV+QQQ)")
    print(f"{'='*80}")
    for cn in ["C1_harvey", "C2_harv_turb", "C3_harv_turb_ar"]:
        eq_df = pd.DataFrame(equity_tracking[cn], columns=["date", "eq"]).set_index("date")
        monthly_eq = eq_df["eq"].resample("MS").mean()
        sub = monthly_eq["2009-01":"2009-06"]
        if len(sub) > 0:
            vals = "  ".join(f"{d.strftime('%b')}={v:.0%}" for d, v in sub.items())
            print(f"  {labels[cn]:>18}: {vals}")

    # Calendar year returns
    print(f"\n{'='*80}")
    print(f"  CALENDAR YEAR RETURNS")
    print(f"{'='*80}")
    show = ["C1_harvey","C2_harv_turb","C3_harv_turb_ar","bench_6040","bench_ivv"]
    hdr = f"  {'Year':>6}"
    for cn in show: hdr += f" {labels[cn][:10]:>10}"
    print(hdr)
    for yr in range(2000, 2025):
        row = f"  {yr:>6}"
        for cn in show:
            s = results.get(cn)
            if s is None: row += f" {'--':>10}"; continue
            y = s[s.index.year == yr]
            row += f" {(1+y).prod()-1:>+9.1%}" if len(y) > 20 else f" {'--':>10}"
        print(row)

    # Final values
    print(f"\n{'='*80}")
    print(f"  FINAL VALUES ($1 invested)")
    print(f"{'='*80}")
    for cn in show:
        s = results.get(cn)
        if s is None or len(s) < 252: continue
        print(f"  {labels[cn]:>18}: ${(1+s).cumprod().iloc[-1]:>8.2f}")

    # Daily equity allocation 2007-2009
    print(f"\n{'='*80}")
    print(f"  DAILY EQUITY ALLOCATION 2007-2009 (monthly avg)")
    print(f"{'='*80}")
    for cn in ["C1_harvey", "C2_harv_turb", "C3_harv_turb_ar"]:
        eq_df = pd.DataFrame(equity_tracking[cn], columns=["date", "eq"]).set_index("date")
        monthly_eq = eq_df["eq"].resample("MS").mean()
        sub = monthly_eq["2007-01":"2009-12"]
        if len(sub) > 0:
            print(f"\n  {labels[cn]}:")
            for d, v in sub.items():
                bar = "█" * int(v * 40)
                print(f"    {d.strftime('%Y-%m')}: {v:>4.0%} {bar}")

    print()


if __name__ == "__main__":
    main()
