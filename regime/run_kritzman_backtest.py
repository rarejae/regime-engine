"""3-way comparison: Harvey only vs Harvey+turbulence vs Harvey+turb+absorption."""

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
    apply_turbulence_scaling, apply_absorption_constraints,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

OUTPUT = Path("experiments/signal_development/output")
OUTPUT.mkdir(parents=True, exist_ok=True)
TC = 0.0010
UNIVERSE = ["IVV", "QQQ", "VGLT", "IAU", "DBC", "VNQ", "cash"]

config = RegimeConfig()


def main():
    print("=" * 80)
    print("  KRITZMAN OVERLAY COMPARISON — 7-ETF ROTH IRA")
    print("=" * 80)

    # Load regime data
    raw_macro = fetch_monthly_history(config)
    transformed = transform_variables(raw_macro, config)
    z_data = get_valid_zscored(transformed, config)
    asset_ret = pd.read_parquet("data/macro/roth_asset_returns.parquet")
    if "VXUS" in asset_ret.columns:
        asset_ret = asset_ret.drop(columns=["VXUS"])

    # Compute Kritzman indicators
    print("\nComputing Kritzman indicators...")
    basket = fetch_daily_basket()
    print(f"  Basket: {basket.shape[0]} days, {list(basket.columns)}")

    turb = compute_turbulence(basket)
    turb_pctl = compute_turbulence_pctl(turb)
    ar = compute_absorption_ratio(basket, n_components=2)  # 2 components for 5 assets
    ar_z = compute_ar_zscore(ar)
    ar_21d_chg = ar.diff(21)

    print(f"  Turbulence: {len(turb)} days, mean={turb.mean():.2f}")
    print(f"  Absorption: {len(ar)} days, mean={ar.mean():.3f}")

    # Save daily scores
    daily = pd.DataFrame({
        "turbulence": turb, "turb_pctl": turb_pctl,
        "absorption_ratio": ar, "ar_zscore": ar_z,
    })
    daily.to_parquet(OUTPUT / "kritzman_daily_scores.parquet")

    # Validate spikes
    print("\n  Turbulence spikes:")
    for ds, label in [("2008-10-10","GFC"),("2020-03-16","COVID"),("2022-06-15","2022")]:
        dt = pd.Timestamp(ds)
        for off in range(-3, 4):
            alt = dt + pd.Timedelta(days=off)
            if alt in turb_pctl.index:
                tp = turb_pctl.loc[alt]
                arz = ar_z.loc[alt] if alt in ar_z.index else float('nan')
                print(f"    {ds} ({label}): turb_pctl={tp:.2f}, ar_z={arz:+.2f}")
                break

    # Resample to monthly for overlay — SHIFT by 1 month for PIT safety
    # End-of-month turb/AR values are used for NEXT month's allocation
    turb_monthly = turb_pctl.resample("MS").last().shift(1)
    ar_z_monthly = ar_z.resample("MS").last().shift(1)
    ar_chg_monthly = ar_21d_chg.resample("MS").last().shift(1)
    turb_chg_monthly = turb.diff(21).resample("MS").last().shift(1)

    # FIX: Build forward return lookup — return AFTER a given month
    # asset_ret_fwd[d] = return of month d+1 (what happened after conditions at d)
    asset_ret_fwd = asset_ret.shift(-1)

    # FIX: Shift z-scores by 1 month — use T-1 z-scores for T allocation
    z_data_lagged = z_data.shift(1).dropna()

    # Backtest dates
    start = pd.Timestamp(config.backtest_start)
    end = pd.Timestamp(config.backtest_end)
    bt_dates = z_data_lagged.index[(z_data_lagged.index >= start) & (z_data_lagged.index <= end)]
    bt_dates = bt_dates[bt_dates.isin(asset_ret.index)]

    configs = {
        "C1_harvey": {},
        "C2_harv_turb": {},
        "C3_harv_turb_ar": {},
        "bench_6040": {},
        "bench_ivv": {},
    }
    prev_w = {c: None for c in configs}
    recovery_counter = 0

    for dt in bt_dates:
        avail = [a for a in asset_ret.columns if a in UNIVERSE and pd.notna(asset_ret.loc[dt, a])]
        if len(avail) < 3:
            continue
        actual = {a: asset_ret.loc[dt, a] for a in avail}

        # FIX: Use lagged z-scores (T-1 data for T allocation)
        try:
            sim = compute_distances(z_data_lagged, dt, config)
        except ValueError:
            continue

        # FIX: Use FORWARD returns from similar months (return after similar conditions)
        sim_er = {}
        for a in avail:
            rets = [asset_ret_fwd.loc[d, a] for d in sim.similar_dates
                    if d in asset_ret_fwd.index and pd.notna(asset_ret_fwd.loc[d, a])]
            sim_er[a] = np.mean(rets) if rets else 0.0

        w_harvey = allocate_similarity_weighted(sim_er, max_single=0.40)

        # Config 1: Harvey only
        w1 = dict(w_harvey)

        # Config 2: Harvey + turbulence
        tp = turb_monthly.get(dt, 0.5) if dt in turb_monthly.index else 0.5
        if np.isnan(tp): tp = 0.5
        w2 = apply_turbulence_scaling(w_harvey, tp)

        # Config 3: Harvey + turbulence + absorption
        arz = ar_z_monthly.get(dt, 0) if dt in ar_z_monthly.index else 0
        if np.isnan(arz): arz = 0
        ar_fall = (ar_chg_monthly.get(dt, 0) if dt in ar_chg_monthly.index else 0) < 0
        turb_fall = (turb_chg_monthly.get(dt, 0) if dt in turb_chg_monthly.index else 0) < 0

        if ar_fall and turb_fall and arz > 0.5:
            recovery_counter = min(recovery_counter + 1, 21)
        else:
            recovery_counter = 0
        recovery_pct = recovery_counter / 21.0

        w3 = apply_absorption_constraints(w_harvey, arz, ar_fall, turb_fall, recovery_pct)
        w3 = apply_turbulence_scaling(w3, tp)

        # Benchmarks
        w_6040 = {a: 0.0 for a in avail}
        if "IVV" in avail: w_6040["IVV"] = 0.60
        if "VGLT" in avail: w_6040["VGLT"] = 0.40
        w_ivv = {a: (1.0 if a == "IVV" else 0.0) for a in avail}

        all_w = {"C1_harvey": w1, "C2_harv_turb": w2, "C3_harv_turb_ar": w3,
                 "bench_6040": w_6040, "bench_ivv": w_ivv}

        for cn, cw in all_w.items():
            ret = sum(cw.get(a, 0) * actual.get(a, 0) for a in avail)
            if prev_w[cn] is not None:
                to = sum(abs(cw.get(a, 0) - prev_w[cn].get(a, 0)) for a in avail) / 2
                ret -= to * TC
            configs[cn][dt] = ret
            prev_w[cn] = cw

    results = {c: pd.Series(d).sort_index() for c, d in configs.items() if d}

    # ── Report ────────────────────────────────────────────────────────────────

    ivv = results.get("bench_ivv")

    print(f"\n{'='*80}")
    print(f"  PERFORMANCE COMPARISON")
    print(f"{'='*80}")
    print(f"\n  {'Config':>18} {'AnnRet':>7} {'AnnVol':>7} {'Sharpe':>7} {'MaxDD':>8} {'HitR':>5} {'CorrIVV':>8}")
    print(f"  {'-'*62}")

    labels = {"C1_harvey": "Harvey Only", "C2_harv_turb": "Harvey+Turb",
              "C3_harv_turb_ar": "Harv+Turb+AR", "bench_6040": "60/40", "bench_ivv": "IVV B&H"}

    for cn in ["C1_harvey", "C2_harv_turb", "C3_harv_turb_ar", "bench_6040", "bench_ivv"]:
        s = results.get(cn)
        if s is None or len(s) < 12: continue
        ar_ = s.mean() * 12; av = s.std() * np.sqrt(12)
        sh = ar_ / av if av > 0 else 0
        cum = (1 + s).cumprod()
        dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
        hit = (s > 0).mean()
        corr = s.corr(ivv) if ivv is not None else 0
        print(f"  {labels[cn]:>18} {ar_:>6.1%} {av:>6.1%} {sh:>7.2f} {dd:>7.1%} {hit:>5.0%} {corr:>7.2f}")

    # Crisis drawdowns
    print(f"\n{'='*80}")
    print(f"  CRISIS DRAWDOWNS")
    print(f"{'='*80}")
    for cname, cs, ce in [("GFC","2008-09","2009-03"),("COVID","2020-02","2020-03"),("2022","2022-01","2022-10")]:
        print(f"\n  {cname} ({cs} to {ce}):")
        for cn in ["C1_harvey","C2_harv_turb","C3_harv_turb_ar","bench_6040","bench_ivv"]:
            s = results.get(cn)
            if s is None: continue
            c = s[(s.index >= pd.Timestamp(cs)) & (s.index <= pd.Timestamp(ce))]
            if len(c) == 0: continue
            print(f"    {labels[cn]:>18}: {((1+c).prod()-1):>+7.1%}")

    # 2009 recovery
    print(f"\n{'='*80}")
    print(f"  2009 RECOVERY TEST")
    print(f"{'='*80}")
    for cn in ["C1_harvey","C2_harv_turb","C3_harv_turb_ar"]:
        s = results.get(cn)
        if s is None: continue
        y = s[s.index.year == 2009]
        print(f"  {labels[cn]:>18} 2009 return: {((1+y).prod()-1):>+7.1%}" if len(y) else f"  {cn}: no data")

    # Calendar years
    print(f"\n{'='*80}")
    print(f"  CALENDAR YEAR RETURNS")
    print(f"{'='*80}")
    show = ["C1_harvey","C2_harv_turb","C3_harv_turb_ar"]
    hdr = f"  {'Year':>6}"
    for cn in show: hdr += f" {labels[cn][:12]:>12}"
    hdr += f" {'60/40':>8} {'IVV':>8}"
    print(hdr)
    for yr in range(2000, 2025):
        row = f"  {yr:>6}"
        for cn in show:
            s = results.get(cn)
            if s is None: row += f" {'--':>12}"; continue
            y = s[s.index.year == yr]
            row += f" {(1+y).prod()-1:>+11.1%}" if len(y) else f" {'--':>12}"
        for cn in ["bench_6040","bench_ivv"]:
            s = results.get(cn)
            if s is None: row += f" {'--':>8}"; continue
            y = s[s.index.year == yr]
            row += f" {(1+y).prod()-1:>+7.1%}" if len(y) else f" {'--':>8}"
        print(row)

    # Cost of false positives
    print(f"\n{'='*80}")
    print(f"  COST OF FALSE POSITIVES (calm bull years)")
    print(f"{'='*80}")
    for yr in [2005, 2013, 2014, 2015, 2016, 2017, 2019]:
        row = f"  {yr}:"
        for cn in ["C1_harvey","C2_harv_turb","C3_harv_turb_ar"]:
            s = results.get(cn)
            if s is None: continue
            y = s[s.index.year == yr]
            row += f"  {labels[cn][:8]:>8}={((1+y).prod()-1):>+5.1%}" if len(y) else ""
        print(row)

    # Final values
    print(f"\n{'='*80}")
    print(f"  FINAL VALUES ($1 invested)")
    print(f"{'='*80}")
    for cn in ["C1_harvey","C2_harv_turb","C3_harv_turb_ar","bench_6040","bench_ivv"]:
        s = results.get(cn)
        if s is None or len(s) < 12: continue
        print(f"  {labels[cn]:>18}: ${(1+s).cumprod().iloc[-1]:>8.2f}")

    print()


if __name__ == "__main__":
    main()
