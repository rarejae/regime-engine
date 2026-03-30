"""Daily backtest: strategy layer with Harvey tilts + Kritzman band adjustment."""

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
from regime.strategy_layer import (
    ASSETS, PARAMS, get_kritzman_state, get_active_bands,
    compute_tilts, construct_weights, get_baseline_weights,
)
from regime.run_daily_backtest import fetch_daily_etf_returns

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

OUTPUT = Path("regime/output")
OUTPUT.mkdir(parents=True, exist_ok=True)
TC = 0.0010


def main():
    config = RegimeConfig()

    print("=" * 80)
    print("  STRATEGY LAYER BACKTEST — HARVEY TILTS + KRITZMAN BANDS")
    print("=" * 80)

    # Load data
    raw_macro = fetch_monthly_history(config)
    transformed = transform_variables(raw_macro, config)
    z_data = get_valid_zscored(transformed, config)
    z_data_lagged = z_data.shift(1).dropna()

    asset_ret_monthly = pd.read_parquet("data/macro/roth_asset_returns.parquet")
    if "VXUS" in asset_ret_monthly.columns:
        asset_ret_monthly = asset_ret_monthly.drop(columns=["VXUS"])
    asset_ret_fwd = asset_ret_monthly.shift(-1)

    daily_ret = fetch_daily_etf_returns()
    basket = fetch_daily_basket()
    turb_smooth = compute_turbulence(basket)
    turb_pctl = compute_turbulence_pctl(turb_smooth, window=252)
    ar = compute_absorption_ratio(basket, n_components=2)
    ar_z = compute_ar_zscore(ar)

    # Precompute unconditional means and signal stds
    uncond_means = {}
    signal_history = {a: [] for a in ASSETS if a in asset_ret_fwd.columns}

    # Distance distribution for confidence
    all_min_dists = []

    # Backtest range
    common_start = max(daily_ret.dropna(how="all").index.min(),
                       turb_pctl.index.min(), pd.Timestamp("2002-01-01"))
    trading_days = daily_ret.loc[common_start:].index
    print(f"\nBacktest: {len(trading_days)} days ({common_start.date()} to {trading_days.max().date()})")

    # Configs
    configs = {"A_full": {}, "B_no_kritz": {}, "C_baseline": {},
               "D_unconstrained": {}, "bench_6040": {}, "bench_ivv": {}}
    current_w = {c: None for c in configs}
    total_tc = {c: 0.0 for c in configs}
    n_trades = {c: 0 for c in configs}
    weight_rows = []

    # Kritzman state tracking
    prev_state = "normal"
    recovery_days = 0
    recovery_blend = 0.0

    # Harvey signal vars
    harvey_er = {a: 0.0 for a in ASSETS}
    harvey_min_dist = 5.0

    baseline_w = get_baseline_weights()

    for day in trading_days:
        if day not in daily_ret.index:
            continue
        day_ret = daily_ret.loc[day]
        avail = [a for a in ASSETS if a in day_ret.index and pd.notna(day_ret[a])]
        if len(avail) < 3:
            continue
        actual = {a: float(day_ret[a]) for a in avail}

        # Monthly Harvey signal
        is_month_start = (day == trading_days[0] or
                         day.month != trading_days[trading_days.get_loc(day) - 1].month)

        if is_month_start:
            z_cands = z_data_lagged.index[z_data_lagged.index < day]
            if len(z_cands) > 0:
                z_dt = z_cands[-1]
                try:
                    sim = compute_distances(z_data_lagged, z_dt, config)
                    harvey_min_dist = sim.min_distance
                    all_min_dists.append(sim.min_distance)

                    for a in avail:
                        if a not in asset_ret_fwd.columns:
                            harvey_er[a] = 0.0
                            continue
                        rets = [asset_ret_fwd.loc[d, a] for d in sim.similar_dates
                                if d in asset_ret_fwd.index and pd.notna(asset_ret_fwd.loc[d, a])]
                        harvey_er[a] = np.mean(rets) if rets else 0.0

                    # Update unconditional means (expanding)
                    for a in avail:
                        if a in asset_ret_monthly.columns:
                            hist = asset_ret_monthly.loc[:z_dt, a].dropna()
                            uncond_means[a] = hist.mean() if len(hist) > 12 else 0.0

                    # Update signal stds
                    for a in avail:
                        sig = harvey_er.get(a, 0) - uncond_means.get(a, 0)
                        if a in signal_history:
                            signal_history[a].append(sig)
                except ValueError:
                    pass

        # Signal stds (trailing)
        signal_stds = {}
        for a in ASSETS:
            hist = signal_history.get(a, [])
            if len(hist) >= 12:
                signal_stds[a] = max(np.std(hist[-120:]), 1e-6)
            else:
                signal_stds[a] = 0.01

        # Distance percentiles for confidence
        if len(all_min_dists) >= 20:
            d25 = np.percentile(all_min_dists, 25)
            d75 = np.percentile(all_min_dists, 75)
        else:
            d25, d75 = 2.0, 5.0

        # Daily Kritzman indicators
        tp = turb_pctl.get(day, 0.5) if day in turb_pctl.index else 0.5
        if pd.isna(tp): tp = 0.5
        arz = ar_z.get(day, 0.0) if day in ar_z.index else 0.0
        if pd.isna(arz): arz = 0.0

        state = get_kritzman_state(tp, arz)

        # Recovery tracking
        if state == "normal" and prev_state in ("elevated", "crisis"):
            recovery_days = 21
        if recovery_days > 0:
            recovery_blend = 1.0 - recovery_days / 21.0
            recovery_days -= 1
            effective_state = prev_state  # Keep constrained state during recovery
        else:
            recovery_blend = 0.0
            effective_state = state
            prev_state = state

        # Compute tilts
        tilts = compute_tilts(harvey_er, uncond_means, signal_stds,
                              harvey_min_dist, d25, d75)

        # Config A: Full system (Harvey tilts + Kritzman bands)
        bands_a = get_active_bands(effective_state, recovery_blend)
        w_a = construct_weights(tilts, bands_a)

        # Config B: Harvey tilts, normal bands only (no Kritzman)
        bands_b = get_active_bands("normal")
        w_b = construct_weights(tilts, bands_b)

        # Config C: Baseline only
        w_c = dict(baseline_w)

        # Config D: Unconstrained Harvey+Turb+AR (from previous backtest)
        from regime.run_daily_backtest import apply_turb_scaling, apply_ar_caps
        w_d_raw = allocate_similarity_weighted(harvey_er, max_single=0.40)
        w_d = apply_ar_caps(w_d_raw, arz)
        w_d = apply_turb_scaling(w_d, tp)

        # Benchmarks
        w_6040 = {a: 0.0 for a in avail}
        if "IVV" in avail: w_6040["IVV"] = 0.60
        if "VGLT" in avail: w_6040["VGLT"] = 0.40
        w_ivv = {a: (1.0 if a == "IVV" else 0.0) for a in avail}

        is_trigger = is_month_start or state != prev_state
        all_configs = {
            "A_full": (w_a, is_trigger), "B_no_kritz": (w_b, is_month_start),
            "C_baseline": (w_c, is_month_start), "D_unconstrained": (w_d, is_trigger),
            "bench_6040": (w_6040, is_month_start), "bench_ivv": (w_ivv, False),
        }

        if is_month_start:
            row = {"date": day, "state": state}
            for a in ASSETS: row[a] = w_a.get(a, 0)
            weight_rows.append(row)

        for cn, (new_w, should_trade) in all_configs.items():
            if current_w[cn] is not None and not should_trade:
                w_used = current_w[cn]
            else:
                w_used = new_w
                to = sum(abs(new_w.get(a, 0) - (current_w[cn] or {}).get(a, 0))
                        for a in avail) / 2 if current_w[cn] else 0
                total_tc[cn] += to * TC
                if to > 0.01: n_trades[cn] += 1
                current_w[cn] = new_w

            ret = sum(w_used.get(a, 0) * actual.get(a, 0) for a in avail)
            configs[cn][day] = ret

        if state != "normal":
            prev_state = state

    results = {c: pd.Series(d).sort_index() for c, d in configs.items() if d}
    weights_df = pd.DataFrame(weight_rows).set_index("date")
    weights_df.to_parquet(OUTPUT / "strategy_weights.parquet")

    n_years = len(trading_days) / 252
    ivv = results.get("bench_ivv")

    # ── Report ────────────────────────────────────────────────────────────────

    labels = {"A_full": "A: Full System", "B_no_kritz": "B: No Kritzman",
              "C_baseline": "C: Baseline", "D_unconstrained": "D: Unconstrained",
              "bench_6040": "60/40", "bench_ivv": "IVV B&H"}

    print(f"\n{'='*80}")
    print(f"  PERFORMANCE COMPARISON")
    print(f"{'='*80}")
    print(f"\n  {'Config':>18} {'AnnRet':>7} {'AnnVol':>7} {'Sharpe':>7} {'MaxDD':>8} {'CorrIVV':>8} {'AnnTC':>6}")
    print(f"  {'-'*62}")

    for cn in ["A_full","B_no_kritz","C_baseline","D_unconstrained","bench_6040","bench_ivv"]:
        s = results.get(cn)
        if s is None or len(s) < 252: continue
        ar_ = s.mean() * 252; av = s.std() * np.sqrt(252)
        sh = ar_ / av if av > 0 else 0
        cum = (1 + s).cumprod()
        dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
        corr = s.corr(ivv) if ivv is not None else 0
        tc = total_tc[cn] / n_years
        print(f"  {labels[cn]:>18} {ar_:>6.1%} {av:>6.1%} {sh:>7.2f} {dd:>7.1%} {corr:>7.2f} {tc:>5.2%}")

    # Crisis drawdowns
    print(f"\n{'='*80}")
    print(f"  CRISIS DRAWDOWNS")
    print(f"{'='*80}")
    for cname, cs, ce in [("GFC","2008-09-01","2009-03-31"),("COVID","2020-02-19","2020-03-23"),("2022","2022-01-03","2022-10-31")]:
        print(f"\n  {cname}:")
        for cn in ["A_full","B_no_kritz","C_baseline","D_unconstrained","bench_6040","bench_ivv"]:
            s = results.get(cn)
            if s is None: continue
            c = s[(s.index >= pd.Timestamp(cs)) & (s.index <= pd.Timestamp(ce))]
            if len(c) == 0: continue
            print(f"    {labels[cn]:>18}: {((1+c).prod()-1):>+7.1%}")

    # Calendar years
    print(f"\n{'='*80}")
    print(f"  CALENDAR YEAR RETURNS")
    print(f"{'='*80}")
    show = ["A_full","B_no_kritz","C_baseline","D_unconstrained","bench_ivv"]
    hdr = f"  {'Year':>6}"
    for cn in show: hdr += f" {labels[cn][:10]:>10}"
    print(hdr)
    for yr in range(2002, 2025):
        row = f"  {yr:>6}"
        for cn in show:
            s = results.get(cn)
            if s is None: row += f" {'--':>10}"; continue
            y = s[s.index.year == yr]
            row += f" {(1+y).prod()-1:>+9.1%}" if len(y) > 20 else f" {'--':>10}"
        print(row)

    # Final values
    print(f"\n{'='*80}")
    print(f"  FINAL VALUES ($1)")
    print(f"{'='*80}")
    for cn in ["A_full","B_no_kritz","C_baseline","D_unconstrained","bench_6040","bench_ivv"]:
        s = results.get(cn)
        if s is None or len(s) < 252: continue
        print(f"  {labels[cn]:>18}: ${(1+s).cumprod().iloc[-1]:>8.2f}")

    # 2013 test
    print(f"\n{'='*80}")
    print(f"  2013 TEST (floor prevents zero equity?)")
    print(f"{'='*80}")
    for cn in ["A_full","B_no_kritz","D_unconstrained"]:
        s = results.get(cn)
        if s is None: continue
        y = s[s.index.year == 2013]
        if len(y) > 0:
            print(f"  {labels[cn]:>18}: {(1+y).prod()-1:>+7.1%}")
    w_2013 = weights_df[weights_df.index.year == 2013]
    if len(w_2013) > 0:
        print(f"\n  Config A avg weights in 2013:")
        for a in ASSETS:
            if a in w_2013.columns:
                print(f"    {a:>6}: {w_2013[a].mean():.0%}")

    # 2008-2009 weights
    print(f"\n{'='*80}")
    print(f"  2008-2009 CONFIG A MONTHLY WEIGHTS")
    print(f"{'='*80}")
    w_gfc = weights_df[(weights_df.index >= "2007-07") & (weights_df.index <= "2009-06")]
    if len(w_gfc) > 0:
        hdr = f"  {'Date':>7} {'State':>8}"
        for a in ASSETS: hdr += f" {a[:4]:>5}"
        print(hdr)
        for dt, row in w_gfc.iterrows():
            line = f"  {dt.strftime('%Y-%m'):>7} {row.get('state','?'):>8}"
            for a in ASSETS:
                v = row.get(a, 0)
                line += f" {v:>4.0%}" if v > 0.005 else f" {'--':>4}"
            print(line)

    # Band state 2007-2009 and 2020-2022
    print(f"\n  Kritzman state by month:")
    for period, s, e in [("2007-2009","2007-01","2009-12"),("2020-2022","2020-01","2022-12")]:
        print(f"\n    {period}:")
        sub = weights_df[(weights_df.index >= s) & (weights_df.index <= e)]
        for dt, row in sub.iterrows():
            print(f"      {dt.strftime('%Y-%m')}: {row.get('state','?')}")

    # Avg allocation
    print(f"\n{'='*80}")
    print(f"  CONFIG A AVERAGE ALLOCATION")
    print(f"{'='*80}")
    for a in ASSETS:
        if a in weights_df.columns:
            print(f"  {a:>6}: {weights_df[a].mean():.1%} (min={weights_df[a].min():.0%}, max={weights_df[a].max():.0%})")

    # Investigation: July 2008 and Feb 2020 state classification
    print(f"\n{'='*80}")
    print(f"  STATE CLASSIFICATION INVESTIGATION")
    print(f"{'='*80}")
    for period, s, e in [("Jun-Aug 2008","2008-06-01","2008-08-31"),
                          ("Jan-Mar 2020","2020-01-01","2020-03-31")]:
        print(f"\n  {period} — daily turb_pctl and ar_z:")
        print(f"  {'Date':>10} {'turb_pctl':>10} {'ar_z':>8} {'state':>10}")
        for day in pd.date_range(s, e, freq="B"):
            tp_val = turb_pctl.get(day, np.nan) if day in turb_pctl.index else np.nan
            arz_val = ar_z.get(day, np.nan) if day in ar_z.index else np.nan
            if pd.isna(tp_val) and pd.isna(arz_val):
                continue
            st = get_kritzman_state(tp_val if not pd.isna(tp_val) else 0.5,
                                    arz_val if not pd.isna(arz_val) else 0.0)
            # Only print first and last of each week + any state changes
            if day.day in (1, 15) or day == pd.Timestamp(s) or day == pd.Timestamp(e):
                tp_s = f"{tp_val:.3f}" if not pd.isna(tp_val) else "N/A"
                arz_s = f"{arz_val:+.2f}" if not pd.isna(arz_val) else "N/A"
                print(f"  {day.date()!s:>10} {tp_s:>10} {arz_s:>8} {st:>10}")

    print()


if __name__ == "__main__":
    main()
