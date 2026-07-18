"""V18: Drawdown Protection — Portfolio CB + Leading Indicator Leverage Scaling.

Part 1: Validate leading indicators as crash predictors.
Part 2A: Portfolio-level drawdown CB overlay on V16-B.
Part 2B: Leading indicator leverage scaling (if Part 1 passes).
Part 2C: Combined (if both A and B pass).
"""

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from dotenv import load_dotenv; load_dotenv()

import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

from experiments.v11_beta_scaled.backtest import (
    SMA_PERIODS, SSO_EXP, QLD_EXP,
    load_data, asset_score, check_breach, lev_ret,
    run_baseline, run_v9,
    cagr, max_dd, sharpe_r, sortino_r, calmar_r, dca_terminal, metrics_row,
)
from experiments.v16_two_pod_gold.backtest import run_v16


# ═════════════════════════════════════════════════════════════════════════════
# PART 1: LEADING INDICATOR ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

def load_indicators(daily_ret, dpdf):
    """Load leading indicator time series."""
    from fredapi import Fred
    key = os.environ.get("FRED_API_KEY")
    if not key:
        print("  WARNING: No FRED_API_KEY — skipping FRED indicators")
        return {}

    fred = Fred(api_key=key)
    indicators = {}

    # VIX
    try:
        vix = fred.get_series("VIXCLS", observation_start="2000-01-01")
        vix.index = pd.to_datetime(vix.index)
        indicators["VIX"] = vix.dropna().astype(float)
    except: pass

    # Yield curve (10Y - 3M)
    try:
        yc = fred.get_series("T10Y3M", observation_start="2000-01-01")
        yc.index = pd.to_datetime(yc.index)
        indicators["yield_curve"] = yc.dropna().astype(float)
    except: pass

    # Credit spread (Baa - 10Y)
    try:
        baa = fred.get_series("BAA10Y", observation_start="2000-01-01")
        baa.index = pd.to_datetime(baa.index)
        indicators["credit_spread"] = baa.dropna().astype(float)
    except: pass

    # VIX slope (21-day change)
    if "VIX" in indicators:
        vix_slope = indicators["VIX"].diff(21)
        indicators["VIX_slope"] = vix_slope.dropna()

    # S&P 500 drawdown from 252-day high
    if "IVV" in dpdf.columns:
        ivv_p = dpdf["IVV"].dropna()
        ivv_high = ivv_p.rolling(252, min_periods=126).max()
        indicators["sp500_drawdown"] = ((ivv_p - ivv_high) / ivv_high).dropna() * 100

    return indicators


def run_part1(indicators, daily_ret):
    """Validate leading indicators as crash predictors."""
    print(f"\n{'=' * 140}")
    print("  PART 1: LEADING INDICATOR VALIDATION")
    print(f"{'=' * 140}")

    qqq = daily_ret["QQQ"].dropna()
    # Monthly QQQ returns for forward-looking analysis
    qqq_monthly = qqq.resample("MS").apply(lambda x: (1 + x).prod() - 1)

    # Build monthly indicator values (end-of-month)
    monthly_ind = {}
    for name, series in indicators.items():
        monthly_ind[name] = series.resample("MS").last().dropna()

    configs = {
        "VIX": {"thresholds": [20, 25, 30], "direction": "above"},
        "VIX_slope": {"thresholds": [5, 10, 15], "direction": "above"},
        "yield_curve": {"thresholds": [0.0, -0.5, -1.0], "direction": "below"},
        "credit_spread": {"thresholds": [2.0, 2.5, 3.0], "direction": "above"},
        "sp500_drawdown": {"thresholds": [-5, -10, -15], "direction": "below"},
    }

    crises = {
        "Dot-com": pd.Timestamp("2000-09-01"),
        "GFC": pd.Timestamp("2007-10-01"),
        "COVID": pd.Timestamp("2020-02-01"),
        "2022": pd.Timestamp("2022-01-01"),
    }

    # For fwd return computation, use monthly
    common_months = qqq_monthly.loc["2002-01-01":"2025-12-31"].index
    total_m = len(common_months)

    print(f"\n  Evaluation period: {common_months[0].date()} to {common_months[-1].date()} ({total_m} months)")

    best_indicators = []

    for ind_name, cfg in configs.items():
        if ind_name not in monthly_ind:
            print(f"\n  {ind_name}: DATA NOT AVAILABLE — skipping")
            continue

        ind_series = monthly_ind[ind_name]

        print(f"\n  {'─' * 100}")
        print(f"  INDICATOR: {ind_name} (direction: {cfg['direction']})")
        print(f"  {'─' * 100}")

        for thresh in cfg["thresholds"]:
            # Signal: is indicator above/below threshold?
            if cfg["direction"] == "above":
                signal = (ind_series >= thresh).reindex(common_months).fillna(False)
                label = f"≥{thresh}"
            else:
                signal = (ind_series <= thresh).reindex(common_months).fillna(False)
                label = f"≤{thresh}"

            active_months = signal.sum()
            pct = active_months / total_m

            # Forward returns when signal active vs inactive
            fwd_1m = qqq_monthly.shift(-1).reindex(common_months)
            fwd_3m = qqq_monthly.rolling(3).apply(lambda x: (1 + x).prod() - 1).shift(-3).reindex(common_months)
            fwd_6m = qqq_monthly.rolling(6).apply(lambda x: (1 + x).prod() - 1).shift(-6).reindex(common_months)

            # Hit rate: P(fwd_6m < -10% | signal)
            on = signal == True
            off = signal == False

            def dd_rate(fwd, mask, thresh_dd):
                vals = fwd[mask].dropna()
                if len(vals) == 0: return 0, 0
                return (vals < thresh_dd).mean(), len(vals)

            hr_on_10, n_on = dd_rate(fwd_6m, on, -0.10)
            hr_off_10, n_off = dd_rate(fwd_6m, off, -0.10)
            hr_on_5, _ = dd_rate(fwd_3m, on, -0.05)
            hr_off_5, _ = dd_rate(fwd_3m, off, -0.05)

            # Crisis lead times
            leads = []
            for crisis_name, crisis_date in crises.items():
                # Check if signal was active in 1-6 months before crisis
                check_range = pd.date_range(crisis_date - pd.DateOffset(months=6), crisis_date, freq="MS")
                active_before = sum(1 for d in check_range if d in signal.index and signal.get(d, False))
                if active_before > 0:
                    first_active = min(d for d in check_range if d in signal.index and signal.get(d, False))
                    lead_days = (crisis_date - first_active).days
                    leads.append((crisis_name, lead_days, True))
                else:
                    leads.append((crisis_name, 0, False))

            crises_led = sum(1 for _, _, hit in leads if hit)

            # False positive rate: signal on, no drawdown > -10% within 6mo
            if n_on > 0:
                fp_rate = 1.0 - hr_on_10
            else:
                fp_rate = 1.0

            print(f"\n    Threshold {label}:")
            print(f"      Active: {active_months}/{total_m} ({pct:.0%})")
            print(f"      P(QQQ -10% in 6mo | signal):    {hr_on_10:.0%} (n={n_on})")
            print(f"      P(QQQ -10% in 6mo | NO signal): {hr_off_10:.0%} (n={n_off})")
            print(f"      P(QQQ -5% in 3mo | signal):     {hr_on_5:.0%}")
            print(f"      False positive rate:              {fp_rate:.0%}")
            print(f"      Crises led (within 6mo):          {crises_led}/4", end="")
            for cn, ld, hit in leads:
                print(f"  {cn}:{'✓'+str(ld)+'d' if hit else '✗'}", end="")
            print()

            best_indicators.append({
                "name": ind_name, "threshold": thresh, "label": label,
                "active_pct": pct, "hit_rate": hr_on_10, "fp_rate": fp_rate,
                "crises_led": crises_led, "lift": hr_on_10 - hr_off_10,
            })

    # Rank indicators
    print(f"\n  {'═' * 100}")
    print(f"  INDICATOR RANKING (sorted by hit rate - false positive rate)")
    print(f"  {'═' * 100}")
    ranked = sorted(best_indicators, key=lambda x: x["hit_rate"] - x["fp_rate"], reverse=True)
    print(f"\n  {'Indicator':<20}{'Thresh':>8}{'Active%':>9}{'HitRate':>9}{'FP Rate':>9}{'Crises':>8}{'Lift':>8}")
    print(f"  {'-'*20}{'-'*7:>8}{'-'*8:>9}{'-'*8:>9}{'-'*8:>9}{'-'*7:>8}{'-'*7:>8}")
    for r in ranked[:10]:
        print(f"  {r['name']:<20}{r['label']:>8}{r['active_pct']:>9.0%}{r['hit_rate']:>9.0%}"
              f"{r['fp_rate']:>9.0%}{r['crises_led']:>7}/4{r['lift']:>+8.0%}")

    # Determine which pass Part 1 criteria
    passing = [r for r in ranked if r["hit_rate"] >= 0.30 and r["fp_rate"] <= 0.80 and r["crises_led"] >= 2]
    print(f"\n  Indicators passing Part 1 (hit≥30%, FP≤80%, crises≥2): {len(passing)}")
    for p in passing[:5]:
        print(f"    {p['name']} {p['label']}: hit={p['hit_rate']:.0%}, FP={p['fp_rate']:.0%}, crises={p['crises_led']}/4")

    return passing, indicators, monthly_ind


# ═════════════════════════════════════════════════════════════════════════════
# PART 2A: PORTFOLIO-LEVEL DRAWDOWN CB
# ═════════════════════════════════════════════════════════════════════════════

def run_v16_with_pcb(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                     start_date, pcb_threshold=-0.10):
    """V16-B with portfolio-level drawdown circuit breaker overlay."""
    bt_start = pd.Timestamp(start_date)
    common_start = max(daily_ret.dropna(how="all").index.min(), bt_start)
    trading_days = daily_ret.loc[common_start:pd.Timestamp("2026-03-31")].index

    port = {}; cb1 = 0; cb2 = 0; pcb_events = []
    nav1 = 0.45; nav2 = 0.45; nav_g = 0.10
    hwm = nav1 + nav2 + nav_g
    pcb_active = False

    p1_mode = "cash"; p1_lev = False; p1_delev = False
    p2_mode = "cash"; p2_lev = False; p2_delev = False
    gold_mode = "cash"
    scores = {}

    for day in trading_days:
        dr = daily_ret.loc[day]
        is_ms = (day == trading_days[0] or day.month != trading_days[trading_days.get_loc(day) - 1].month)

        if is_ms:
            p1_delev = False; p2_delev = False; pcb_active = False
            prior = trading_days[trading_days < day]
            sd = prior[-1] if len(prior) > 0 else day
            scores = {a: asset_score(sd, a, dpdf, daily_smas) for a in ["QQQ", "IVV", "IAU"]}

            sc_q = scores["QQQ"]; sc_i = scores["IVV"]
            if sc_q >= 3:
                if sc_i <= 1: p1_mode = "qqq"; p1_lev = False
                else: p1_mode = "qld"; p1_lev = True
            elif sc_q == 2: p1_mode = "qqq_partial"; p1_lev = False
            else: p1_mode = "cash"; p1_lev = False

            if sc_i >= 3: p2_mode = "sso"; p2_lev = True
            elif sc_i == 2: p2_mode = "ivv_partial"; p2_lev = False
            else: p2_mode = "cash"; p2_lev = False

            gold_mode = "iau" if scores["IAU"] >= 3 else "cash"

            # Rebalance
            if day != trading_days[0]:
                total = nav1 + nav2 + nav_g
                if total > 0:
                    drift = max(abs(nav1/total - 0.45), abs(nav2/total - 0.45), abs(nav_g/total - 0.10))
                    if drift > 0.05:
                        nav1 = total * 0.45; nav2 = total * 0.45; nav_g = total * 0.10

        # Per-asset CB
        if p1_lev and not p1_delev:
            if check_breach(day, "QQQ", dpdf, daily_smas):
                p1_lev = False; p1_delev = True; cb1 += 1; p1_mode = "qqq"
        if p2_lev and not p2_delev:
            if check_breach(day, "IVV", dpdf, daily_smas):
                p2_lev = False; p2_delev = True; cb2 += 1; p2_mode = "ivv"

        # Portfolio CB check (after per-asset CB)
        total = nav1 + nav2 + nav_g
        if total > 0 and not pcb_active:
            current_dd = (total - hwm) / hwm
            if current_dd <= pcb_threshold:
                # Exit all leverage
                if p1_lev: p1_lev = False; p1_mode = "qqq"
                if p2_lev: p2_lev = False; p2_mode = "ivv"
                pcb_active = True
                pcb_events.append({"date": day, "dd": current_dd, "hwm": hwm, "total": total})

        rfr = float(rfr_daily.get(day, 0.0))
        qqq_u = float(dr.get("QQQ", 0.0)) if pd.notna(dr.get("QQQ", np.nan)) else 0.0
        ivv_u = float(dr.get("IVV", 0.0)) if pd.notna(dr.get("IVV", np.nan)) else 0.0
        iau_u = float(dr.get("IAU", 0.0)) if pd.notna(dr.get("IAU", np.nan)) else 0.0

        if p1_mode == "qld":
            r1 = lev_ret(qqq_u, rfr, QLD_EXP, day, actual_lev, "QLD", both_start) if p1_lev else qqq_u
        elif p1_mode == "qqq": r1 = qqq_u
        elif p1_mode == "qqq_partial": r1 = 0.70 * qqq_u + 0.30 * rfr
        else: r1 = rfr

        if p2_mode == "sso":
            r2 = lev_ret(ivv_u, rfr, SSO_EXP, day, actual_lev, "SSO", both_start) if p2_lev else ivv_u
        elif p2_mode == "ivv": r2 = ivv_u
        elif p2_mode == "ivv_partial": r2 = 0.70 * ivv_u + 0.30 * rfr
        else: r2 = rfr

        rg = iau_u if gold_mode == "iau" else rfr

        prev_total = nav1 + nav2 + nav_g
        nav1 *= (1 + r1); nav2 *= (1 + r2); nav_g *= (1 + rg)
        new_total = nav1 + nav2 + nav_g
        if new_total > hwm: hwm = new_total
        port[day] = new_total / prev_total - 1 if prev_total > 0 else 0

    return pd.Series(port).sort_index(), cb1 + cb2, pcb_events


# ═════════════════════════════════════════════════════════════════════════════
# PART 2B: LEADING INDICATOR LEVERAGE SCALING
# ═════════════════════════════════════════════════════════════════════════════

def run_v16_with_li(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                    start_date, indicator_series, threshold, direction, sub_pct=0.50):
    """V16-B with leading-indicator-conditioned leverage reduction."""
    bt_start = pd.Timestamp(start_date)
    common_start = max(daily_ret.dropna(how="all").index.min(), bt_start)
    trading_days = daily_ret.loc[common_start:pd.Timestamp("2026-03-31")].index

    # Monthly indicator values
    ind_monthly = indicator_series.resample("MS").last()

    port = {}; cb1 = 0; cb2 = 0
    nav1 = 0.45; nav2 = 0.45; nav_g = 0.10
    p1_mode = "cash"; p1_lev = False; p1_delev = False
    p2_mode = "cash"; p2_lev = False; p2_delev = False
    gold_mode = "cash"; risk_elevated = False
    scores = {}; li_months = 0

    for day in trading_days:
        dr = daily_ret.loc[day]
        is_ms = (day == trading_days[0] or day.month != trading_days[trading_days.get_loc(day) - 1].month)

        if is_ms:
            p1_delev = False; p2_delev = False
            prior = trading_days[trading_days < day]
            sd = prior[-1] if len(prior) > 0 else day
            scores = {a: asset_score(sd, a, dpdf, daily_smas) for a in ["QQQ", "IVV", "IAU"]}

            # Check leading indicator
            ind_val = ind_monthly.get(pd.Timestamp(day.year, day.month, 1), np.nan)
            if not pd.isna(ind_val):
                if direction == "above":
                    risk_elevated = ind_val >= threshold
                else:
                    risk_elevated = ind_val <= threshold
            if risk_elevated: li_months += 1

            sc_q = scores["QQQ"]; sc_i = scores["IVV"]

            # Pod 1 — leverage conditioned on risk
            if sc_q >= 3:
                if sc_i <= 1:
                    p1_mode = "qqq"; p1_lev = False
                elif risk_elevated:
                    # Reduced leverage: sub_pct of QLD, rest in QQQ
                    p1_mode = "qld_reduced"; p1_lev = True
                else:
                    p1_mode = "qld"; p1_lev = True
            elif sc_q == 2: p1_mode = "qqq_partial"; p1_lev = False
            else: p1_mode = "cash"; p1_lev = False

            # Pod 2
            if sc_i >= 3:
                if risk_elevated:
                    p2_mode = "sso_reduced"; p2_lev = True
                else:
                    p2_mode = "sso"; p2_lev = True
            elif sc_i == 2: p2_mode = "ivv_partial"; p2_lev = False
            else: p2_mode = "cash"; p2_lev = False

            gold_mode = "iau" if scores["IAU"] >= 3 else "cash"

            if day != trading_days[0]:
                total = nav1 + nav2 + nav_g
                if total > 0:
                    drift = max(abs(nav1/total - 0.45), abs(nav2/total - 0.45), abs(nav_g/total - 0.10))
                    if drift > 0.05:
                        nav1 = total * 0.45; nav2 = total * 0.45; nav_g = total * 0.10

        if p1_lev and not p1_delev:
            if check_breach(day, "QQQ", dpdf, daily_smas):
                p1_lev = False; p1_delev = True; cb1 += 1; p1_mode = "qqq"
        if p2_lev and not p2_delev:
            if check_breach(day, "IVV", dpdf, daily_smas):
                p2_lev = False; p2_delev = True; cb2 += 1; p2_mode = "ivv"

        rfr = float(rfr_daily.get(day, 0.0))
        qqq_u = float(dr.get("QQQ", 0.0)) if pd.notna(dr.get("QQQ", np.nan)) else 0.0
        ivv_u = float(dr.get("IVV", 0.0)) if pd.notna(dr.get("IVV", np.nan)) else 0.0
        iau_u = float(dr.get("IAU", 0.0)) if pd.notna(dr.get("IAU", np.nan)) else 0.0

        # Pod 1 return with possible reduced leverage
        if p1_mode == "qld":
            r1 = lev_ret(qqq_u, rfr, QLD_EXP, day, actual_lev, "QLD", both_start) if p1_lev else qqq_u
        elif p1_mode == "qld_reduced":
            qld_r = lev_ret(qqq_u, rfr, QLD_EXP, day, actual_lev, "QLD", both_start) if p1_lev else qqq_u
            r1 = sub_pct * qld_r + (1 - sub_pct) * qqq_u  # blend
        elif p1_mode == "qqq": r1 = qqq_u
        elif p1_mode == "qqq_partial": r1 = 0.70 * qqq_u + 0.30 * rfr
        else: r1 = rfr

        if p2_mode == "sso":
            r2 = lev_ret(ivv_u, rfr, SSO_EXP, day, actual_lev, "SSO", both_start) if p2_lev else ivv_u
        elif p2_mode == "sso_reduced":
            sso_r = lev_ret(ivv_u, rfr, SSO_EXP, day, actual_lev, "SSO", both_start) if p2_lev else ivv_u
            r2 = sub_pct * sso_r + (1 - sub_pct) * ivv_u
        elif p2_mode == "ivv": r2 = ivv_u
        elif p2_mode == "ivv_partial": r2 = 0.70 * ivv_u + 0.30 * rfr
        else: r2 = rfr

        rg = iau_u if gold_mode == "iau" else rfr

        prev_total = nav1 + nav2 + nav_g
        nav1 *= (1 + r1); nav2 *= (1 + r2); nav_g *= (1 + rg)
        new_total = nav1 + nav2 + nav_g
        port[day] = new_total / prev_total - 1 if prev_total > 0 else 0

    return pd.Series(port).sort_index(), cb1 + cb2, li_months


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 140)
    print("  V18 DRAWDOWN PROTECTION — PORTFOLIO CB + LEADING INDICATORS")
    print("=" * 140)

    print("\n  Loading data...")
    daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, dbmf_ret, dbmf_inception = load_data()

    # ── PART 1 ──
    indicators = load_indicators(daily_ret, dpdf)
    passing_indicators, indicators, monthly_ind = run_part1(indicators, daily_ret)

    # ── PART 2A: Portfolio CB ──
    print(f"\n{'=' * 140}")
    print("  PART 2A: PORTFOLIO-LEVEL DRAWDOWN CIRCUIT BREAKER")
    print(f"{'=' * 140}")

    print("\n  Running V16-B baseline...")
    v16_full, v16c1, v16c2, _, _ = run_v16(
        daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
        "2002-01-01", iau_threshold=3)

    pcb_results = {}
    for thresh in [-0.08, -0.10, -0.12, -0.15]:
        name = f"PCB-{abs(int(thresh*100))}"
        print(f"  Running {name}...")
        s, cb, events = run_v16_with_pcb(
            daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
            "2002-01-01", pcb_threshold=thresh)
        pcb_results[name] = (s, cb, events)

    print(f"\n  {'Strategy':<22} {'CAGR':>7} {'Vol':>7} {'Sharpe':>7} {'Sortino':>8} "
          f"{'MaxDD':>7} {'Calmar':>7} {'Term$1':>9} {'DCA$700':>9} {'PCB':>5}")
    print(f"  {'-' * 22} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 8} {'-' * 7} {'-' * 7} {'-' * 9} {'-' * 9} {'-' * 5}")
    for nm in ["PCB-8", "PCB-10", "PCB-12", "PCB-15"]:
        s, cb, events = pcb_results[nm]
        c = cagr(s); v = s.std()*np.sqrt(252); sh = sharpe_r(s); so = sortino_r(s)
        dd = max_dd(s); cl = calmar_r(s); t = (1+s).cumprod().iloc[-1]
        sm = s.resample("MS").apply(lambda x: (1+x).prod()-1)
        dca = dca_terminal(sm)
        print(f"  {nm:<22} {c:>6.2%} {v:>6.2%} {sh:>7.3f} {so:>8.3f} {dd:>6.1%} {cl:>7.2f} ${t:>8.2f} ${dca/1e6:>7.2f}M {len(events):>5}")
    print(metrics_row("V16-B (no PCB)        ", v16_full, v16c1 + v16c2))

    # PCB event details
    for nm in ["PCB-8", "PCB-10", "PCB-12", "PCB-15"]:
        events = pcb_results[nm][2]
        if events:
            print(f"\n  {nm} events ({len(events)} total):")
            for e in events:
                print(f"    {e['date'].strftime('%Y-%m-%d')}: DD={e['dd']:.1%}, HWM={e['hwm']:.4f}, total={e['total']:.4f}")

    # Crisis DDs
    print(f"\n  Crisis drawdowns:")
    crises = [("GFC","2007-11-01","2009-03-31"),("COVID","2020-02-01","2020-04-30"),("2022","2022-01-01","2022-12-31")]
    names = ["PCB-8", "PCB-10", "PCB-12", "PCB-15", "V16-B"]
    print(f"  {'Crisis':<12}" + "".join(f"{nm:>10}" for nm in names))
    for label, cs, ce in crises:
        cells = []
        for nm in names[:-1]:
            sp = pcb_results[nm][0][(pcb_results[nm][0].index >= cs) & (pcb_results[nm][0].index <= ce)]
            cells.append(max_dd(sp) if len(sp) > 5 else 0)
        sp = v16_full[(v16_full.index >= cs) & (v16_full.index <= ce)]
        cells.append(max_dd(sp) if len(sp) > 5 else 0)
        print(f"  {label:<12}" + "".join(f"{c:>10.1%}" for c in cells))

    # ── PART 2B: Leading indicator overlay ──
    if passing_indicators:
        print(f"\n{'=' * 140}")
        print("  PART 2B: LEADING INDICATOR LEVERAGE SCALING")
        print(f"{'=' * 140}")

        # Use top 3 passing indicators
        li_results = {}
        for p in passing_indicators[:3]:
            ind_name = p["name"]; thresh = p["threshold"]
            cfg_dir = {"VIX": "above", "VIX_slope": "above", "yield_curve": "below",
                       "credit_spread": "above", "sp500_drawdown": "below"}
            for sub in [0.50, 0.0]:
                label = f"{ind_name}{p['label']}_sub{int(sub*100)}"
                print(f"  Running {label}...")
                s, cb, li_m = run_v16_with_li(
                    daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                    "2002-01-01", indicators[ind_name], thresh,
                    cfg_dir.get(ind_name, "above"), sub_pct=sub)
                li_results[label] = (s, cb, li_m)

        print(f"\n  {'Strategy':<30} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>7} {'Term$1':>9} {'LI mos':>7}")
        print(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*7} {'-'*9} {'-'*7}")
        for nm, (s, cb, li_m) in li_results.items():
            c = cagr(s); sh = sharpe_r(s); dd = max_dd(s); t = (1+s).cumprod().iloc[-1]
            print(f"  {nm:<30} {c:>6.2%} {sh:>7.3f} {dd:>6.1%} ${t:>8.2f} {li_m:>7}")
        c16 = cagr(v16_full); sh16 = sharpe_r(v16_full); dd16 = max_dd(v16_full); t16 = (1+v16_full).cumprod().iloc[-1]
        print(f"  {'V16-B baseline':<30} {c16:>6.2%} {sh16:>7.3f} {dd16:>6.1%} ${t16:>8.2f}")
    else:
        print(f"\n  PART 2B: SKIPPED — no indicators passed Part 1 criteria")

    # ── PART 3: Summary ──
    print(f"\n{'=' * 140}")
    print("  PART 3: PASS / FAIL SUMMARY")
    print(f"{'=' * 140}")

    v16_c = cagr(v16_full); v16_sh = sharpe_r(v16_full); v16_dd = max_dd(v16_full)

    print(f"\n  V16-B reference: CAGR {v16_c:.2%}, Sharpe {v16_sh:.3f}, MaxDD {v16_dd:.1%}")

    # PCB pass check
    print(f"\n  Part 2A (Portfolio CB):")
    for nm in ["PCB-8", "PCB-10", "PCB-12", "PCB-15"]:
        s = pcb_results[nm][0]
        vc = cagr(s); vsh = sharpe_r(s); vdd = max_dd(s)
        dd_ok = vdd > v16_dd + 0.03  # 3pp improvement
        sh_ok = vsh >= v16_sh
        cagr_ok = vc >= v16_c - 0.01
        passed = dd_ok and sh_ok and cagr_ok
        print(f"    {nm}: DD {vdd:.1%} (imp {vdd-v16_dd:+.1%}), Sharpe {vsh:.3f}, CAGR {vc:.2%} → {'PASS' if passed else 'FAIL'}")

    if passing_indicators:
        print(f"\n  Part 2B (Leading Indicators):")
        for nm, (s, cb, li_m) in li_results.items():
            vc = cagr(s); vsh = sharpe_r(s); vdd = max_dd(s)
            dd_ok = vdd > v16_dd + 0.03
            sh_ok = vsh >= v16_sh
            cagr_ok = vc >= v16_c - 0.015
            passed = dd_ok and sh_ok and cagr_ok
            print(f"    {nm}: DD {vdd:.1%}, Sharpe {vsh:.3f}, CAGR {vc:.2%} → {'PASS' if passed else 'FAIL'}")

    print()


if __name__ == "__main__":
    main()
