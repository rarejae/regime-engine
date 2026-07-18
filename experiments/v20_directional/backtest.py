"""V20: Directional State Transitions on V19d.

Score 2/3 treatment varies by direction:
  3→2 (falling): more defensive
  1→2 (rising): more aggressive
  2→2 (stable): V19d default (70/30)

Variants A-D test different aggressiveness levels.
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
from experiments.v19_cb_cash_exit.backtest import run_v19


# Variant configs: (falling_equity_pct, rising_equity_pct)
# falling: at score 3→2, this fraction of the pod is in underlying (rest cash)
# rising: at score 1→2, this fraction is in underlying (rest cash)
VARIANTS = {
    "V20-A": {"fall": 0.50, "rise": 1.00},  # moderate
    "V20-B": {"fall": 0.00, "rise": 1.00},  # aggressive
    "V20-D": {"fall": 0.00, "rise": 0.70},  # defensive only (rising = V19d default)
    "V19d":  {"fall": 0.70, "rise": 0.70},  # control
}


def run_v20(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
            start_date, fall_pct=0.50, rise_pct=1.00, capture_diag=False):
    bt_start = pd.Timestamp(start_date)
    common_start = max(daily_ret.dropna(how="all").index.min(), bt_start)
    trading_days = daily_ret.loc[common_start:pd.Timestamp("2026-03-31")].index

    port = {}; cb1 = 0; cb2 = 0; cb_g = 0
    nav1 = 0.45; nav2 = 0.45; nav_g = 0.10

    p1_mode = "cash"; p1_lev = False; p1_delev = False
    p2_mode = "cash"; p2_lev = False; p2_delev = False
    gold_mode = "cash"; gold_delev = False

    prev_scores = {"QQQ": 0, "IVV": 0}
    scores = {"QQQ": 0, "IVV": 0, "IAU": 0}

    transitions = []  # for diagnostics

    for day in trading_days:
        dr = daily_ret.loc[day]
        is_ms = (day == trading_days[0] or day.month != trading_days[trading_days.get_loc(day) - 1].month)

        if is_ms:
            p1_delev = False; p2_delev = False; gold_delev = False
            prior = trading_days[trading_days < day]
            sd = prior[-1] if len(prior) > 0 else day

            prev_scores = dict(scores)
            scores = {a: asset_score(sd, a, dpdf, daily_smas) for a in ["QQQ", "IVV", "IAU"]}
            sc_q = scores["QQQ"]; sc_i = scores["IVV"]
            prev_q = prev_scores["QQQ"]; prev_i = prev_scores["IVV"]

            # Pod 1
            if sc_q >= 3:
                if sc_i <= 1: p1_mode = "qqq"; p1_lev = False
                else: p1_mode = "qld"; p1_lev = True
            elif sc_q == 2:
                # Directional treatment
                if prev_q >= 3:
                    direction = "falling"; eq_pct = fall_pct
                elif prev_q <= 1:
                    direction = "rising"; eq_pct = rise_pct
                else:
                    direction = "stable"; eq_pct = 0.70
                p1_mode = f"qqq_dir_{eq_pct}"; p1_lev = False
                if capture_diag:
                    transitions.append({"month": day, "asset": "QQQ", "prev": prev_q,
                                        "curr": sc_q, "direction": direction, "eq_pct": eq_pct})
            else:
                p1_mode = "cash"; p1_lev = False

            # Pod 2
            if sc_i >= 3: p2_mode = "sso"; p2_lev = True
            elif sc_i == 2:
                if prev_i >= 3:
                    direction = "falling"; eq_pct = fall_pct
                elif prev_i <= 1:
                    direction = "rising"; eq_pct = rise_pct
                else:
                    direction = "stable"; eq_pct = 0.70
                p2_mode = f"ivv_dir_{eq_pct}"; p2_lev = False
                if capture_diag:
                    transitions.append({"month": day, "asset": "IVV", "prev": prev_i,
                                        "curr": sc_i, "direction": direction, "eq_pct": eq_pct})
            else:
                p2_mode = "cash"; p2_lev = False

            gold_mode = "iau" if scores["IAU"] >= 3 else "cash"

            if day != trading_days[0]:
                total = nav1 + nav2 + nav_g
                if total > 0:
                    drift = max(abs(nav1/total - 0.45), abs(nav2/total - 0.45), abs(nav_g/total - 0.10))
                    if drift > 0.05:
                        nav1 = total * 0.45; nav2 = total * 0.45; nav_g = total * 0.10

        # CBs
        if p1_lev and not p1_delev:
            if check_breach(day, "QQQ", dpdf, daily_smas):
                p1_lev = False; p1_delev = True; cb1 += 1; p1_mode = "cash"
        if p2_lev and not p2_delev:
            if check_breach(day, "IVV", dpdf, daily_smas):
                p2_lev = False; p2_delev = True; cb2 += 1; p2_mode = "cash"
        if gold_mode == "iau" and not gold_delev:
            if check_breach(day, "IAU", dpdf, daily_smas):
                gold_mode = "cash"; gold_delev = True; cb_g += 1

        rfr = float(rfr_daily.get(day, 0.0))
        qqq_u = float(dr.get("QQQ", 0.0)) if pd.notna(dr.get("QQQ", np.nan)) else 0.0
        ivv_u = float(dr.get("IVV", 0.0)) if pd.notna(dr.get("IVV", np.nan)) else 0.0
        iau_u = float(dr.get("IAU", 0.0)) if pd.notna(dr.get("IAU", np.nan)) else 0.0

        # Pod 1 return
        if p1_mode == "qld":
            r1 = lev_ret(qqq_u, rfr, QLD_EXP, day, actual_lev, "QLD", both_start) if p1_lev else qqq_u
        elif p1_mode == "qqq": r1 = qqq_u
        elif p1_mode.startswith("qqq_dir_"):
            eq = float(p1_mode.split("_")[-1])
            r1 = eq * qqq_u + (1 - eq) * rfr
        else: r1 = rfr

        # Pod 2 return
        if p2_mode == "sso":
            r2 = lev_ret(ivv_u, rfr, SSO_EXP, day, actual_lev, "SSO", both_start) if p2_lev else ivv_u
        elif p2_mode == "ivv": r2 = ivv_u
        elif p2_mode.startswith("ivv_dir_"):
            eq = float(p2_mode.split("_")[-1])
            r2 = eq * ivv_u + (1 - eq) * rfr
        else: r2 = rfr

        rg = iau_u if gold_mode == "iau" else rfr

        prev_total = nav1 + nav2 + nav_g
        nav1 *= (1 + r1); nav2 *= (1 + r2); nav_g *= (1 + rg)
        new_total = nav1 + nav2 + nav_g
        port[day] = new_total / prev_total - 1 if prev_total > 0 else 0

    return pd.Series(port).sort_index(), cb1 + cb2, transitions


def main():
    print("=" * 140)
    print("  V20 DIRECTIONAL STATE TRANSITIONS — FULL BACKTEST")
    print("=" * 140)

    print("\n  Loading data...")
    daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, dbmf_ret, dbmf_inception = load_data()

    results = {}
    for name, cfg in VARIANTS.items():
        print(f"  Running {name}...")
        s, cb, trans = run_v20(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                                "2002-01-01", fall_pct=cfg["fall"], rise_pct=cfg["rise"],
                                capture_diag=(name == "V20-A"))
        results[name] = (s, cb, trans)

    # Also run V19d control directly
    print("  Running V19d control...")
    v19d_full, _, _, _ = run_v19(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, "2002-01-01")
    v9_full, v9_cb = run_v9(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, "2002-01-01")
    bl_full, bl_cb = run_baseline(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                                   dbmf_ret, dbmf_inception, "2002-01-01")
    qqq_full = daily_ret["QQQ"].loc["2002-01-01":"2026-03-31"].dropna()

    # ── TABLE 1 ──
    print(f"\n{'=' * 140}")
    print("  TABLE 1: CORE METRICS (2002-2026)")
    print(f"{'=' * 140}")
    print(f"\n  {'Strategy':<22} {'CAGR':>7} {'Vol':>7} {'Sharpe':>7} {'Sortino':>8} "
          f"{'MaxDD':>7} {'Calmar':>7} {'Term$1':>9} {'DCA$700':>9}")
    print(f"  {'-' * 22} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 8} {'-' * 7} {'-' * 7} {'-' * 9} {'-' * 9}")
    for nm in ["V20-A", "V20-B", "V20-D", "V19d"]:
        s = results[nm][0]
        c = cagr(s); v = s.std()*np.sqrt(252); sh = sharpe_r(s); so = sortino_r(s)
        dd = max_dd(s); cl = calmar_r(s); t = (1+s).cumprod().iloc[-1]
        sm = s.resample("MS").apply(lambda x: (1+x).prod()-1); dca = dca_terminal(sm)
        print(f"  {nm:<22} {c:>6.2%} {v:>6.2%} {sh:>7.3f} {so:>8.3f} {dd:>6.1%} {cl:>7.2f} ${t:>8.2f} ${dca/1e6:>7.2f}M")
    print(metrics_row("V9 QLD+IVVguard       ", v9_full, v9_cb))
    print(metrics_row("Baseline              ", bl_full, bl_cb))

    # ── TABLE 2: Transition frequency ──
    print(f"\n{'=' * 140}")
    print("  TABLE 2: TRANSITION FREQUENCY AND OUTCOME ANALYSIS")
    print(f"{'=' * 140}")
    trans = results["V20-A"][2]  # captured from V20-A
    tdf = pd.DataFrame(trans)

    for asset in ["QQQ", "IVV"]:
        at = tdf[tdf["asset"] == asset]
        total = len(at)
        falling = at[at["direction"] == "falling"]
        rising = at[at["direction"] == "rising"]
        stable = at[at["direction"] == "stable"]
        print(f"\n  {asset}: {total} score-2 months total")
        print(f"    3→2 (falling): {len(falling)}")
        print(f"    1→2 (rising):  {len(rising)}")
        print(f"    2→2 (stable):  {len(stable)}")

    # Transition outcome: what happened next month?
    print(f"\n  Transition outcomes (next month's score):")
    # Reconstruct scores per month for outcome analysis
    trading_days_full = daily_ret.loc["2002-01-01":"2026-03-31"].dropna(how="all").index
    monthly_scores = {}
    for day in trading_days_full:
        if day == trading_days_full[0] or day.month != trading_days_full[trading_days_full.get_loc(day)-1].month:
            prior = trading_days_full[trading_days_full < day]
            sd = prior[-1] if len(prior) > 0 else day
            for a in ["QQQ", "IVV"]:
                sc = asset_score(sd, a, dpdf, daily_smas)
                monthly_scores.setdefault(a, []).append((day, sc))

    for asset in ["QQQ", "IVV"]:
        scores_list = monthly_scores[asset]
        at = tdf[tdf["asset"] == asset]
        for direction in ["falling", "rising"]:
            subset = at[at["direction"] == direction]
            if len(subset) == 0: continue
            outcomes = {"→3": 0, "→2": 0, "→1": 0, "→0": 0}
            for _, row in subset.iterrows():
                m = row["month"]
                # Find next month's score
                idx = next((i for i, (d, _) in enumerate(scores_list) if d == m), None)
                if idx is not None and idx + 1 < len(scores_list):
                    next_sc = scores_list[idx + 1][1]
                    if next_sc >= 3: outcomes["→3"] += 1
                    elif next_sc == 2: outcomes["→2"] += 1
                    elif next_sc == 1: outcomes["→1"] += 1
                    else: outcomes["→0"] += 1
            total = sum(outcomes.values())
            print(f"\n    {asset} {direction} ({len(subset)} events):")
            for k, v in outcomes.items():
                print(f"      {k}: {v} ({v/total:.0%})" if total > 0 else f"      {k}: {v}")
            if direction == "falling" and total > 0:
                decline_pct = (outcomes["→1"] + outcomes["→0"]) / total
                print(f"      → Further decline rate: {decline_pct:.0%}")
            if direction == "rising" and total > 0:
                recovery_pct = outcomes["→3"] / total
                print(f"      → Recovery rate (→3): {recovery_pct:.0%}")

    # ── TABLE 3: Crisis drawdowns ──
    print(f"\n{'=' * 140}")
    print("  TABLE 3: CRISIS DRAWDOWNS")
    print(f"{'=' * 140}")
    crises = [("Dot-com","2002-01-01","2003-03-31"),("GFC","2007-11-01","2009-03-31"),
              ("COVID","2020-02-01","2020-04-30"),("2022","2022-01-01","2022-12-31")]
    names = ["V20-A", "V20-B", "V20-D", "V19d", "BL"]
    all_s = [results[n][0] for n in ["V20-A","V20-B","V20-D","V19d"]] + [bl_full]
    print(f"\n  {'Crisis':<18}" + "".join(f"{nm:>10}" for nm in names))
    for label, cs, ce in crises:
        cells = [max_dd(s[(s.index >= cs) & (s.index <= ce)]) if len(s[(s.index >= cs) & (s.index <= ce)]) > 5 else 0 for s in all_s]
        print(f"  {label:<18}" + "".join(f"{c:>10.1%}" for c in cells))

    # ── TABLE 4: Start-date sensitivity ──
    print(f"\n{'=' * 140}")
    print("  TABLE 4: CAGR BY START DATE")
    print(f"{'=' * 140}")
    start_dates = ["2002-01-01", "2007-01-01", "2010-01-01", "2013-01-01", "2019-01-01"]
    print(f"\n  {'Strategy':<22}" + "".join(f"{sd[:4]:>10}" for sd in start_dates))
    for nm in ["V20-A", "V20-B", "V20-D"]:
        cfg = VARIANTS[nm]
        row = f"  {nm:<22}"
        for sd in start_dates:
            s, _, _ = run_v20(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                               sd, fall_pct=cfg["fall"], rise_pct=cfg["rise"])
            row += f"{cagr(s):>10.2%}"
        print(row)
    row = f"  {'V19d':<22}"
    for sd in start_dates:
        s, _, _, _ = run_v19(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, sd)
        row += f"{cagr(s):>10.2%}"
    print(row)

    # ── TABLE 5: Pass/fail ──
    print(f"\n{'=' * 140}")
    print("  TABLE 5: PASS / FAIL")
    print(f"{'=' * 140}")
    v19d_c = cagr(v19d_full); v19d_sh = sharpe_r(v19d_full); v19d_dd = max_dd(v19d_full)
    print(f"\n  V19d reference: CAGR {v19d_c:.2%}, Sharpe {v19d_sh:.3f}, MaxDD {v19d_dd:.1%}")

    for nm in ["V20-A", "V20-B", "V20-D"]:
        s = results[nm][0]
        vc = cagr(s); vsh = sharpe_r(s); vdd = max_dd(s)
        sh_ok = vsh > v19d_sh; dd_ok = vdd >= v19d_dd; cagr_ok = vc >= v19d_c - 0.005
        passed = sh_ok and dd_ok and cagr_ok
        print(f"\n  {nm}: CAGR {vc:.2%}, Sharpe {vsh:.3f}, MaxDD {vdd:.1%}")
        print(f"    Sharpe > V19d:  {'✓' if sh_ok else '✗'} ({vsh:.3f} vs {v19d_sh:.3f})")
        print(f"    MaxDD ≤ V19d:   {'✓' if dd_ok else '✗'}")
        print(f"    CAGR ≥ {v19d_c - 0.005:.2%}: {'✓' if cagr_ok else '✗'}")
        print(f"    → {'PASS' if passed else 'FAIL'}")

    print()


if __name__ == "__main__":
    main()
