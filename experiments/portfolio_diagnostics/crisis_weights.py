"""4-asset portfolio weights around crisis periods."""

import sys, os, warnings, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from dotenv import load_dotenv; load_dotenv()
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

import numpy as np, pandas as pd

from regime.config import RegimeConfig
from regime.data import fetch_monthly_history
from regime.transform import transform_variables, get_valid_zscored
from regime.similarity import compute_distances
from regime.hmm_trend import fetch_daily_data, compute_features, fit_and_predict_rolling, apply_persistence_filter
from regime.kritzman import (fetch_daily_basket, compute_turbulence, compute_turbulence_pctl,
                              compute_absorption_ratio, compute_ar_zscore)
from regime.carry_value import compute_carry, compute_value
from regime.run_daily_backtest import fetch_daily_etf_returns

OUTPUT = Path(__file__).resolve().parent / "output"
SMA_PERIODS = [6, 10, 12]
TIER_SUBS = {0: 0.0, 1: 0.30, 2: 0.65}
BASE = {"IVV": 0.45, "QQQ": 0.25, "IAU": 0.15, "cash": 0.15}
ASSETS = list(BASE.keys())
EQUITY = ["IVV", "QQQ"]

CRISES = [
    ("GFC CRISIS",   "2008-01", "2009-12", {"2008-09": "Lehman", "2008-10": "Crash", "2009-03": "Bottom"}),
    ("COVID CRISIS",  "2019-10", "2020-09", {"2020-02": "Crash starts", "2020-03": "Bottom", "2020-04": "Recovery"}),
    ("2022 BEAR",     "2021-07", "2023-03", {"2021-12": "Peak", "2022-01": "Decline starts", "2022-10": "Bottom"}),
]


def compute_multi_sma(prices_df):
    monthly = prices_df.resample("MS").last()
    result = pd.DataFrame(0, index=monthly.index, columns=monthly.columns)
    for period in SMA_PERIODS:
        sma = monthly.rolling(period, min_periods=period).mean()
        result += (monthly > sma).astype(int)
    return result

def step1_faber(strengths):
    w = {}; pool = 0.0
    for a in ASSETS:
        if a == "cash": w[a] = BASE[a]; continue
        s = strengths.get(a, 0)
        if s >= 3: w[a] = BASE[a]
        elif s == 2: w[a] = BASE[a] * 0.70; pool += BASE[a] * 0.30
        else: w[a] = 0.0; pool += BASE[a]
    return w, pool

def step2_harvey_cv(weights, pool, harvey_er, rvols, carry, value):
    if pool <= 0.001: return weights
    candidates = {}
    for a in ASSETS:
        if a == "cash" or weights.get(a, 0) <= 0: continue
        adj = harvey_er.get(a, 0) + carry.get(a, 0)
        if adj <= 0: continue
        score = adj / max(rvols.get(a, 0.15), 0.01) + 0.01 * value.get(a, 0)
        if score > 0: candidates[a] = score
    if not candidates:
        weights["cash"] = weights.get("cash", 0) + pool; return weights
    if len(candidates) == 1:
        a = list(candidates.keys())[0]
        alloc = max(min(pool, 0.40 - weights.get(a, 0)), 0)
        weights[a] = weights.get(a, 0) + alloc
        weights["cash"] = weights.get("cash", 0) + (pool - alloc); return weights
    ts = sum(candidates.values())
    for a, sc in candidates.items(): weights[a] = weights.get(a, 0) + pool * (sc / ts)
    return weights

def step3_hmm(weights, bp):
    if bp > 0.7:
        for a in EQUITY:
            if weights.get(a, 0) > 0:
                b = weights[a] * 0.15; weights[a] += b; weights["cash"] = max(weights.get("cash", 0) - b, 0)
    elif bp < 0.3:
        for a in EQUITY:
            if weights.get(a, 0) > 0:
                r = weights[a] * 0.15; weights[a] -= r; weights["cash"] = weights.get("cash", 0) + r
    return weights

def step4_kritzman(weights, tp, arz):
    if tp > 0.95 and arz > 2.0:
        for a in EQUITY:
            if weights.get(a, 0) > 0:
                f = weights[a] * 0.50; weights[a] -= f; weights["cash"] = weights.get("cash", 0) + f
    return weights

def step5_normalize(weights):
    w = dict(weights)
    risky = [a for a in w if a != "cash" and w.get(a, 0) > 0.01]
    if len(risky) == 1 and w[risky[0]] > 0.40:
        w["cash"] = w.get("cash", 0) + w[risky[0]] - 0.40; w[risky[0]] = 0.40
    for _ in range(10):
        total = sum(max(v, 0) for v in w.values())
        if total > 0 and abs(total - 1.0) > 1e-6: w = {a: max(v, 0) / total for a, v in w.items()}
        changed = False
        for a in w:
            if w[a] > 0.60: w["cash"] = w.get("cash", 0) + w[a] - 0.60; w[a] = 0.60; changed = True
        eq = sum(w.get(a, 0) for a in EQUITY)
        if eq > 0.85:
            r = 0.85 / eq
            for a in EQUITY: freed = w[a] - w[a] * r; w[a] *= r; w["cash"] = w.get("cash", 0) + freed
            changed = True
        if w.get("cash", 0) < 0.03:
            deficit = 0.03 - w["cash"]; others = [a for a in w if a != "cash" and w[a] > 0]
            tot = sum(w[a] for a in others)
            if tot > deficit:
                for a in others: w[a] -= deficit * (w[a] / tot)
            w["cash"] = 0.03; changed = True
        if not changed: break
    total = sum(max(v, 0) for v in w.values())
    if total > 0 and abs(total - 1.0) > 1e-6: w = {a: max(v, 0) / total for a, v in w.items()}
    return w


def main():
    lines = []
    def pr(s=""): print(s); lines.append(s)

    config = RegimeConfig()
    raw_macro = fetch_monthly_history(config)
    transformed = transform_variables(raw_macro, config)
    z_data = get_valid_zscored(transformed, config)
    z_data_lagged = z_data.shift(1).dropna()
    asset_ret = pd.read_parquet("data/macro/roth_asset_returns.parquet")
    asset_ret_fwd = asset_ret.shift(-1)

    print("Loading data...")
    daily_ret = fetch_daily_etf_returns()

    import yfinance as yf
    ep = {}
    for our, ticker in [("IVV","SPY"),("QQQ","QQQ"),("VGLT","TLT"),("IAU","GLD"),("DBC","DBC")]:
        d = yf.download(ticker, start="1998-01-01", progress=False)
        if d is not None and not d.empty:
            p = d["Close"]
            if hasattr(p,"columns"): p=p.iloc[:,0]
            p.index = pd.to_datetime(p.index).tz_localize(None)
            ep[our] = p
    prices_df = pd.DataFrame(ep).sort_index()
    multi_sma_df = compute_multi_sma(prices_df).shift(1)
    rvol_63 = prices_df.pct_change().rolling(63, min_periods=30).std() * np.sqrt(252)
    rvol_monthly = rvol_63.resample("MS").last().shift(1)

    print("Computing carry/value...")
    carry_df = compute_carry(prices_df)
    value_df = compute_value(prices_df)

    print("Fitting HMM...")
    spy_raw = fetch_daily_data()
    hmm_feat = compute_features(spy_raw)
    hmm_pred = fit_and_predict_rolling(hmm_feat).set_index("date")
    zone_raw = pd.Series("neutral", index=hmm_pred.index)
    zone_raw[hmm_pred["bull_prob"]>0.7]="bull"
    zone_raw[hmm_pred["bull_prob"]<0.3]="bear"
    hmm_pred["zone"] = apply_persistence_filter(zone_raw)

    basket = fetch_daily_basket()
    turb_smooth = compute_turbulence(basket)
    turb_pctl_s = compute_turbulence_pctl(turb_smooth, window=252)
    ar_obj = compute_absorption_ratio(basket, n_components=2)
    ar_z_s = compute_ar_zscore(ar_obj)
    turb_m = turb_pctl_s.resample("MS").last().shift(1)
    ar_z_m = ar_z_s.resample("MS").last().shift(1)

    from fredapi import Fred
    rfr_daily = pd.Series(0.0, index=daily_ret.index)
    key = os.environ.get("FRED_API_KEY")
    if key:
        tb = Fred(api_key=key).get_series("DTB3", observation_start="1998-01-01")
        tb.index = pd.to_datetime(tb.index)
        rfr_daily = (tb / 100 / 252).reindex(daily_ret.index, method="ffill").fillna(0)

    daily_prices_df = pd.DataFrame({a: ep[a] for a in ["IVV","QQQ"] if a in ep})
    monthly_close = daily_prices_df.resample("MS").last()
    sma_6m = monthly_close.rolling(6, min_periods=6).mean()
    sma_10m = monthly_close.rolling(10, min_periods=10).mean()
    sma_12m = monthly_close.rolling(12, min_periods=12).mean()

    # ER seed
    pre_start = pd.Timestamp("2002-01-01")
    pre_months = z_data_lagged.index[z_data_lagged.index < pre_start]
    pre_ers = {"IVV": [], "QQQ": []}
    for z_dt in pre_months:
        try:
            sim = compute_distances(z_data_lagged, z_dt, config)
            for a in ["IVV","QQQ"]:
                if a in asset_ret_fwd.columns:
                    rets = [asset_ret_fwd.loc[d,a] for d in sim.similar_dates
                            if d in asset_ret_fwd.index and pd.notna(asset_ret_fwd.loc[d,a])]
                    if rets: pre_ers[a].append(np.mean(rets))
        except ValueError: pass
    SEED = {a: float(np.mean(pre_ers[a])) if pre_ers[a] else 0.005 for a in ["IVV","QQQ"]}

    common_start = max(daily_ret.dropna(how="all").index.min(), hmm_pred.index.min(), pd.Timestamp("2002-01-01"))
    trading_days = daily_ret.loc[common_start:].index

    # ── Run full backtest, recording monthly snapshots ────────────────────────

    snapshots = []  # list of dicts per month
    strengths = {a: 3 for a in ASSETS if a != "cash"}
    harvey_er = {a: 0.0 for a in ASSETS}
    w = dict(BASE)
    tier = 0; delevered = False
    tier_hist = {"IVV": list(pre_ers["IVV"]), "QQQ": list(pre_ers["QQQ"])}
    tier_med = dict(SEED)
    delever_events = []

    for day_idx, day in enumerate(trading_days):
        if day not in daily_ret.index: continue
        dr = daily_ret.loc[day]
        is_ms = (day_idx == 0 or day.month != trading_days[day_idx - 1].month)
        is_friday = day.weekday() == 4

        avail = [a for a in ASSETS if a in dr.index and pd.notna(dr[a])]
        if len(avail) < 2: continue

        if is_ms:
            delevered = False
            ms_cands = multi_sma_df.index[multi_sma_df.index <= day]
            if len(ms_cands) > 0:
                ms_dt = ms_cands[-1]
                for a in ASSETS:
                    if a != "cash" and a in multi_sma_df.columns:
                        v = multi_sma_df.loc[ms_dt, a]
                        strengths[a] = int(v) if pd.notna(v) else 0

            z_cands = z_data_lagged.index[z_data_lagged.index < day]
            if len(z_cands) > 0:
                try:
                    sim = compute_distances(z_data_lagged, z_cands[-1], config)
                    for a in avail:
                        if a not in asset_ret_fwd.columns: harvey_er[a]=0; continue
                        rets = [asset_ret_fwd.loc[d,a] for d in sim.similar_dates
                                if d in asset_ret_fwd.index and pd.notna(asset_ret_fwd.loc[d,a])]
                        harvey_er[a] = np.mean(rets) if rets else 0
                except ValueError: pass

            rv_cands = rvol_monthly.index[rvol_monthly.index <= day]
            rvols = {}
            if len(rv_cands) > 0:
                for a in ASSETS:
                    if a != "cash" and a in rvol_monthly.columns:
                        v = rvol_monthly.loc[rv_cands[-1], a]
                        rvols[a] = float(v) if pd.notna(v) and v > 0 else 0.15

            carry = {}; value = {}
            for a in ASSETS:
                if a in carry_df.columns:
                    c_c = carry_df.index[carry_df.index <= day]
                    if len(c_c) > 0: cv = carry_df.loc[c_c[-1], a]; carry[a] = float(cv) if pd.notna(cv) else 0
                if a in value_df.columns:
                    v_c = value_df.index[value_df.index <= day]
                    if len(v_c) > 0: vv = value_df.loc[v_c[-1], a]; value[a] = float(vv) if pd.notna(vv) else 0

            prev_m = day - pd.DateOffset(months=1)
            hm = hmm_pred[(hmm_pred.index >= prev_m) & (hmm_pred.index < day)]
            bp = hm["bull_prob"].mean() if len(hm) > 0 else 0.5
            if pd.isna(bp): bp = 0.5
            tp = turb_m.get(day, 0.5) if day in turb_m.index else 0.5
            if pd.isna(tp): tp = 0.5
            arz = ar_z_m.get(day, 0) if day in ar_z_m.index else 0
            if pd.isna(arz): arz = 0

            hmm_zone = "neutral"
            zone_cands = hmm_pred[(hmm_pred.index >= prev_m) & (hmm_pred.index < day)]
            if len(zone_cands) > 0:
                last_zone = zone_cands["zone"].iloc[-1]
                hmm_zone = last_zone

            wn, pool = step1_faber(strengths)
            wn = step2_harvey_cv(wn, pool, harvey_er, rvols, carry, value)
            wn = step3_hmm(wn, bp)
            kritz_fired = tp > 0.95 and arz > 2.0
            wn = step4_kritzman(wn, tp, arz)
            w = step5_normalize(wn)

            # Tier
            f_c = strengths.get("IVV",0) >= 3 and strengths.get("QQQ",0) >= 3
            h_c = harvey_er.get("IVV",0) > 0 and harvey_er.get("QQQ",0) > 0
            if f_c and h_c:
                above = all(harvey_er.get(a,0) > tier_med.get(a,0) for a in ["IVV","QQQ"])
                tier = 2 if above else 1
                for a in ["IVV","QQQ"]:
                    tier_hist[a].append(harvey_er.get(a,0))
                    tier_med[a] = float(np.median(tier_hist[a]))
            else:
                tier = 0

            snapshots.append({
                "date": day,
                "IVV": w.get("IVV", 0), "QQQ": w.get("QQQ", 0),
                "IAU": w.get("IAU", 0), "cash": w.get("cash", 0),
                "tier": tier,
                "ivv_faber": strengths.get("IVV", 0),
                "qqq_faber": strengths.get("QQQ", 0),
                "iau_faber": strengths.get("IAU", 0),
                "pool": pool,
                "hmm_zone": hmm_zone,
                "bp": bp,
                "kritz": kritz_fired,
                "harvey_ivv": harvey_er.get("IVV", 0),
                "harvey_qqq": harvey_er.get("QQQ", 0),
                "harvey_iau": harvey_er.get("IAU", 0),
            })

        # Weekly circuit breaker
        if is_friday and tier > 0 and not delevered:
            breach = False
            breach_detail = ""
            for etf in ["IVV","QQQ"]:
                if etf not in daily_prices_df.columns: continue
                pb = daily_prices_df.loc[:day, etf]
                if len(pb)==0: continue
                pt = pb.iloc[-1]; lms = pd.Timestamp(f"{day.year}-{day.month:02d}-01")
                for period, sdf in [(6, sma_6m), (10, sma_10m), (12, sma_12m)]:
                    sd = sdf.index[sdf.index <= lms]
                    if len(sd)==0: continue
                    sv = sdf.loc[sd[-1], etf]
                    if pd.notna(sv) and pt < sv:
                        breach = True
                        breach_detail = f"{etf} < {period}m SMA"
                        break
                if breach: break
            if breach:
                delever_events.append({"date": day, "detail": breach_detail, "from_tier": tier})
                tier = 0; delevered = True

    snap_df = pd.DataFrame(snapshots).set_index("date")
    snap_df.index = pd.to_datetime(snap_df.index)

    # ── Report ────────────────────────────────────────────────────────────────

    pr("=" * 80)
    pr("  4-ASSET PORTFOLIO WEIGHTS AROUND CRISES")
    pr("=" * 80)
    pr(f"\nBaseline weights: IVV 45%, QQQ 25%, IAU 15%, Cash 15%")

    all_crisis_data = []

    for crisis_name, show_start, show_end, annotations in CRISES:
        pr(f"\n\n{crisis_name} ({show_start} to {show_end})")
        pr("=" * 80)

        cs = pd.Timestamp(show_start + "-01")
        ce = pd.Timestamp(show_end + "-01")
        window = snap_df[(snap_df.index >= cs) & (snap_df.index <= ce)]

        if len(window) == 0:
            pr("  No data in this window")
            continue

        pr(f"\n  {'Date':>8} {'IVV':>6} {'QQQ':>6} {'IAU':>6} {'Cash':>6} {'Tier':>5} {'IVV_F':>6} {'QQQ_F':>6} {'IAU_F':>6} {'HMM':>8} {'Notes'}")
        pr(f"  {'--------':>8} {'------':>6} {'------':>6} {'------':>6} {'------':>6} {'-----':>5} {'------':>6} {'------':>6} {'------':>6} {'--------':>8} {'-----'}")

        for dt, row in window.iterrows():
            dt_str = dt.strftime("%Y-%m")
            ann = annotations.get(dt_str, "")
            # Check for de-lever events in this month
            month_delevs = [e for e in delever_events
                            if e["date"].year == dt.year and e["date"].month == dt.month]
            if month_delevs:
                if ann: ann += "; "
                ann += f"DE-LEV {month_delevs[0]['date'].strftime('%m/%d')} ({month_delevs[0]['detail']})"

            kritz_note = ""
            if row.get("kritz", False):
                if ann: ann += "; "
                ann += "KRITZMAN"

            tier_str = f"T{int(row['tier'])}"
            hmm_short = {"bull": "bull", "bear": "bear", "neutral": "neut"}.get(str(row.get("hmm_zone", "neutral")), "neut")

            if ann: ann = f"← {ann}"

            pr(f"  {dt_str:>8} {row['IVV']:>5.0%} {row['QQQ']:>5.0%} {row['IAU']:>5.0%} {row['cash']:>5.0%}"
               f" {tier_str:>5} {int(row['ivv_faber']):>5}/3 {int(row['qqq_faber']):>5}/3 {int(row['iau_faber']):>5}/3"
               f" {hmm_short:>8} {ann}")

        # Summary
        eq_wt = window["IVV"] + window["QQQ"]
        cash_wt = window["cash"]
        tier0_months = (window["tier"] == 0).sum()

        # When did Faber exit/re-enter
        ivv_below = window[window["ivv_faber"] < 3]
        qqq_below = window[window["qqq_faber"] < 3]
        faber_exit = None
        faber_reenter = None

        eq_below_half = window[eq_wt < 0.35]  # equity below 35% = significant reduction
        if len(eq_below_half) > 0:
            faber_exit = eq_below_half.index[0].strftime("%Y-%m")
            # Find re-entry: first month after exit where equity > 50%
            after_exit = window[window.index > eq_below_half.index[0]]
            eq_after = after_exit["IVV"] + after_exit["QQQ"]
            reenter = eq_after[eq_after > 0.50]
            if len(reenter) > 0:
                faber_reenter = reenter.index[0].strftime("%Y-%m")

        pr(f"\n  Summary:")
        pr(f"    Equity weight range:  {eq_wt.min():.0%} to {eq_wt.max():.0%}")
        pr(f"    Cash weight range:    {cash_wt.min():.0%} to {cash_wt.max():.0%}")
        pr(f"    IAU weight range:     {window['IAU'].min():.0%} to {window['IAU'].max():.0%}")
        pr(f"    Months at Tier 0:     {tier0_months} of {len(window)}")
        if faber_exit:
            pr(f"    Equity reduced below 35%: {faber_exit}")
        if faber_reenter:
            pr(f"    Equity recovered above 50%: {faber_reenter}")

        month_delevs_crisis = [e for e in delever_events
                                if cs <= pd.Timestamp(e["date"]) <= ce]
        if month_delevs_crisis:
            pr(f"    Weekly de-lever events: {len(month_delevs_crisis)}")
            for e in month_delevs_crisis:
                pr(f"      {e['date'].strftime('%Y-%m-%d')}: from Tier {e['from_tier']} ({e['detail']})")
        else:
            pr(f"    Weekly de-lever events: none")

        # Store for cross-crisis comparison
        all_crisis_data.append({
            "name": crisis_name,
            "pre_eq": eq_wt.iloc[0] if len(eq_wt) > 0 else float("nan"),
            "min_eq": eq_wt.min(),
            "reduction": eq_wt.iloc[0] - eq_wt.min() if len(eq_wt) > 0 else 0,
            "months_to_min": (eq_wt.idxmin() - window.index[0]).days // 30 if len(eq_wt) > 0 else 0,
            "tier0_months": tier0_months,
            "n_delevs": len(month_delevs_crisis),
            "max_cash": cash_wt.max(),
        })

    # ── Cross-crisis comparison ───────────────────────────────────────────────
    pr(f"\n\n{'='*80}")
    pr(f"  CRISIS RESPONSE PATTERNS")
    pr(f"{'='*80}")

    if all_crisis_data:
        pr(f"\n  {'Metric':<30}" + "".join(f" {cd['name'][:11]:>13}" for cd in all_crisis_data))
        pr(f"  {'-'*30}" + f" {'-'*13}" * len(all_crisis_data))

        for label, key, fmt in [
            ("Pre-crisis equity weight", "pre_eq", "{:.0%}"),
            ("Min equity weight during", "min_eq", "{:.0%}"),
            ("Equity reduction", "reduction", "{:.0%}"),
            ("Months to min equity", "months_to_min", "{}"),
            ("Months at Tier 0", "tier0_months", "{}"),
            ("Weekly de-lever events", "n_delevs", "{}"),
            ("Max cash weight", "max_cash", "{:.0%}"),
        ]:
            row = f"  {label:<30}"
            for cd in all_crisis_data:
                row += f" {fmt.format(cd[key]):>13}"
            pr(row)

        # Interpretation
        pr(f"\n  Key insight:")
        best = min(all_crisis_data, key=lambda x: x["min_eq"])
        worst = max(all_crisis_data, key=lambda x: x["min_eq"])
        pr(f"    Most aggressive de-risking: {best['name']} (equity down to {best['min_eq']:.0%})")
        pr(f"    Least aggressive: {worst['name']} (equity floor at {worst['min_eq']:.0%})")

        gfc = next((c for c in all_crisis_data if "GFC" in c["name"]), None)
        covid = next((c for c in all_crisis_data if "COVID" in c["name"]), None)
        bear22 = next((c for c in all_crisis_data if "2022" in c["name"]), None)

        if gfc and covid and bear22:
            pr(f"\n    GFC: Slow grind gave Faber time to exit — equity reached {gfc['min_eq']:.0%}.")
            pr(f"    COVID: V-shaped crash was too fast for monthly Faber — equity only dropped to {covid['min_eq']:.0%}.")
            if covid["n_delevs"] > 0:
                pr(f"      Weekly circuit breaker fired {covid['n_delevs']}x, providing intra-month protection.")
            pr(f"    2022: Extended decline (rate hikes) — Faber had months to de-risk, equity at {bear22['min_eq']:.0%}.")

    # Save
    report_path = os.path.join(str(OUTPUT), "crisis_weights_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Report saved: {report_path}")


if __name__ == "__main__":
    main()
