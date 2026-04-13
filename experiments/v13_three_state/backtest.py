"""V13: Three-State V9 with Weekly Re-Entry.

Three states (FULL=QLD, DELEVER=QQQ, CASH). Daily CB exits to cash.
Weekly (Friday) re-entry: 1 Friday with QQQ≥2 + IVV≥2 → QQQ; 2 consecutive
Fridays with both 3/3 → QLD.

Tests two hypotheses simultaneously:
  H1: the delever state (QQQ 1×) reduces DD vs V9's binary cash exit
  H2: weekly re-entry captures recovery upside without excessive whipsaw
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
    run_baseline, run_v9, run_v11,
    cagr, max_dd, sharpe_r, sortino_r, calmar_r, dca_terminal, metrics_row,
)
from experiments.v12_independent_2x.backtest import run_v12


def _monthly_state_from_scores(qqq_sc, ivv_sc):
    """Return the monthly-rebalance state per V13 spec."""
    if qqq_sc >= 3 and ivv_sc >= 3: return "FULL"
    if qqq_sc == 2 and ivv_sc >= 2: return "DELEVER"
    if qqq_sc >= 3 and ivv_sc == 2: return "DELEVER"
    return "CASH"


def run_v13(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
            start_date, capture_diag=False):
    """V13 three-state with weekly re-entry after CB."""
    bt_start = pd.Timestamp(start_date)
    common_start = max(daily_ret.dropna(how="all").index.min(), bt_start)
    trading_days = daily_ret.loc[common_start:pd.Timestamp("2026-03-31")].index

    port = {}
    cb_events = []  # list of dicts
    monthly_states = []  # list of (month, state)
    holding_day = {}  # date -> holding label ('QLD','QQQ','cash')

    holding = "cash"   # current holding
    cb_triggered = False  # in post-CB cash awaiting weekly re-entry
    consec_full_fridays = 0
    pending_transition = None  # holding to apply next trading day
    cur_cb_event = None  # in-progress event

    for day in trading_days:
        dr = daily_ret.loc[day]
        is_ms = (day == trading_days[0] or day.month != trading_days[trading_days.get_loc(day) - 1].month)

        # Apply pending transition from prior day's close decision
        if pending_transition is not None:
            holding = pending_transition
            pending_transition = None

        if is_ms:
            # Monthly rebalance always overrides current state
            prior = trading_days[trading_days < day]
            sd = prior[-1] if len(prior) > 0 else day
            qqq_sc = asset_score(sd, "QQQ", dpdf, daily_smas)
            ivv_sc = asset_score(sd, "IVV", dpdf, daily_smas)
            state = _monthly_state_from_scores(qqq_sc, ivv_sc)

            # Apply
            if state == "FULL": new_holding = "QLD"
            elif state == "DELEVER": new_holding = "QQQ"
            else: new_holding = "cash"

            # Close any in-progress CB event as resolved by monthly
            if cur_cb_event is not None and cur_cb_event["resolution"] is None:
                cur_cb_event["resolution"] = "monthly"
                cur_cb_event["resolution_date"] = day
                cur_cb_event["monthly_resolution_holding"] = new_holding
                cb_events.append(cur_cb_event)
                cur_cb_event = None

            holding = new_holding
            cb_triggered = False
            consec_full_fridays = 0
            monthly_states.append({"month": day, "state": state,
                                    "qqq_sc": qqq_sc, "ivv_sc": ivv_sc})

        # Daily circuit breaker — only if currently holding equity
        if holding in ("QLD", "QQQ"):
            if check_breach(day, "QQQ", dpdf, daily_smas):
                # Exit to cash at next open → transition pending
                pending_transition = "cash"
                cb_triggered = True
                consec_full_fridays = 0
                cur_cb_event = {
                    "date": day, "prior_holding": holding,
                    "resolution": None, "resolution_date": None,
                    "weekly_unlev_date": None, "weekly_lev_date": None,
                    "whipsaw_within_30d": False,
                }

        # Friday weekly check — only if in post-CB state
        if cb_triggered and day.dayofweek == 4:  # Friday
            qqq_sc_now = asset_score(day, "QQQ", dpdf, daily_smas)
            ivv_sc_now = asset_score(day, "IVV", dpdf, daily_smas)

            # Note: current holding for the check is (after any pending transition applied next day)
            # We use the *intended* holding after today (which for CB day is 'cash' since pending='cash')
            effective_holding = pending_transition if pending_transition is not None else holding

            if effective_holding == "cash":
                # Step 1: cash → QQQ
                if qqq_sc_now >= 2 and ivv_sc_now >= 2:
                    pending_transition = "QQQ"
                    consec_full_fridays = 0
                    if cur_cb_event is not None:
                        cur_cb_event["weekly_unlev_date"] = day
            elif effective_holding == "QQQ":
                # Step 2: QQQ → QLD (only via weekly path, cb_triggered still True)
                if qqq_sc_now >= 3 and ivv_sc_now >= 3:
                    consec_full_fridays += 1
                    if consec_full_fridays >= 2:
                        pending_transition = "QLD"
                        if cur_cb_event is not None:
                            cur_cb_event["weekly_lev_date"] = day
                            cur_cb_event["resolution"] = "weekly"
                            cur_cb_event["resolution_date"] = day
                            cb_events.append(cur_cb_event)
                            cur_cb_event = None
                        cb_triggered = False
                        consec_full_fridays = 0
                else:
                    consec_full_fridays = 0

        # Compute daily return — based on holding at START of day (before pending takes effect)
        rfr = float(rfr_daily.get(day, 0.0))
        ret = 0.0
        qqq_u = float(dr.get("QQQ", 0.0)) if pd.notna(dr.get("QQQ", np.nan)) else 0.0
        if holding == "QLD":
            ret = lev_ret(qqq_u, rfr, QLD_EXP, day, actual_lev, "QLD", both_start)
        elif holding == "QQQ":
            ret = qqq_u
        else:
            ret = rfr

        port[day] = ret
        holding_day[day] = holding

    # Resolve any trailing in-progress CB event
    if cur_cb_event is not None:
        cb_events.append(cur_cb_event)

    # Detect whipsaws: CB fires again within 30 days of a resolution
    resolved = [e for e in cb_events if e["resolution_date"] is not None]
    for i, e in enumerate(resolved):
        res_date = e["resolution_date"]
        for j in range(i + 1, len(resolved)):
            next_cb = resolved[j]["date"]
            if (next_cb - res_date).days <= 30:
                e["whipsaw_within_30d"] = True
            break

    return pd.Series(port).sort_index(), cb_events, monthly_states, holding_day


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 140)
    print("  V13 THREE-STATE WITH WEEKLY RE-ENTRY — FULL BACKTEST")
    print("=" * 140)

    print("\n  Loading data...")
    daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, dbmf_ret, dbmf_inception = load_data()

    print("  Running V13...")
    v13_full, v13_cb_events, v13_monthly, v13_holdings = run_v13(
        daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
        "2002-01-01", capture_diag=True)
    print("  Running V9...")
    v9_full, v9_cb = run_v9(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, "2002-01-01")
    print("  Running V12...")
    v12_full, v12_cb, _ = run_v12(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, "2002-01-01")
    print("  Running Baseline...")
    bl_full, bl_cb = run_baseline(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                                   dbmf_ret, dbmf_inception, "2002-01-01")

    qqq_full = daily_ret["QQQ"].loc["2002-01-01":"2026-03-31"].dropna()
    ivv_full = daily_ret["IVV"].loc["2002-01-01":"2026-03-31"].dropna()
    v13_cb_count = len(v13_cb_events)

    # ── TABLE 1: Core metrics ──
    print(f"\n{'=' * 140}")
    print("  TABLE 1: CORE METRICS (2002-2026)")
    print(f"{'=' * 140}")
    print(f"\n  {'Strategy':<22} {'CAGR':>7} {'Vol':>7} {'Sharpe':>7} {'Sortino':>8} "
          f"{'MaxDD':>7} {'Calmar':>7} {'Term$1':>9} {'DCA$700':>9} {'CB':>4}")
    print(f"  {'-' * 22} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 8} {'-' * 7} {'-' * 7} {'-' * 9} {'-' * 9} {'-' * 4}")
    print(metrics_row("V13 ThreeState+Weekly", v13_full, v13_cb_count))
    print(metrics_row("V9 QLD+IVVguard", v9_full, v9_cb))
    print(metrics_row("V12 Independent 2×", v12_full, v12_cb))
    print(metrics_row("Baseline (Sweep-40)", bl_full, bl_cb))
    print(metrics_row("QQQ B&H", qqq_full))
    print(metrics_row("IVV B&H", ivv_full))

    # ── TABLE 2: Start-date sensitivity ──
    print(f"\n{'=' * 140}")
    print("  TABLE 2: CAGR BY START DATE")
    print(f"{'=' * 140}")
    start_dates = ["2002-01-01", "2007-01-01", "2010-01-01", "2013-01-01", "2019-01-01"]
    print(f"\n  {'Strategy':<22}" + "".join(f"{sd[:4]:>10}" for sd in start_dates))
    print(f"  {'-' * 22}" + "".join(f" {'-' * 9}" for _ in start_dates))
    cagrs = {}
    for name, fn in [
        ("V13 ThreeState+Weekly", "v13"),
        ("V9 QLD+IVVguard",       "v9"),
        ("V12 Independent 2×",    "v12"),
        ("Baseline",              "bl"),
    ]:
        row = f"  {name:<22}"; cagrs[name] = {}
        for sd in start_dates:
            if fn == "v13":
                s, _, _, _ = run_v13(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, sd)
            elif fn == "v9":
                s, _ = run_v9(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, sd)
            elif fn == "v12":
                s, _, _ = run_v12(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, sd)
            else:
                s, _ = run_baseline(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                                    dbmf_ret, dbmf_inception, sd)
            c = cagr(s); cagrs[name][sd] = c
            row += f"{c:>10.2%}"
        print(row)
    row = f"  {'QQQ B&H':<22}"
    for sd in start_dates:
        qs = qqq_full[qqq_full.index >= pd.Timestamp(sd)]
        row += f"{cagr(qs):>10.2%}"
    print(row)

    # ── TABLE 3: State occupancy ──
    print(f"\n{'=' * 140}")
    print("  TABLE 3: V13 MONTHLY STATE OCCUPANCY")
    print(f"{'=' * 140}")
    states_df = pd.DataFrame(v13_monthly)
    total_m = len(states_df)
    print(f"\n  Total months: {total_m}")
    for st in ["FULL", "DELEVER", "CASH"]:
        n = (states_df["state"] == st).sum()
        print(f"  {st:<12}: {n:>4} months ({n / total_m:>5.1%})")

    # Compare to what V9 would hold in each month
    # V9 logic: QQQ>=3 and IVV>=2 → QLD; QQQ==2 → QQQ 70%/30%; else cash
    v9_state_labels = []
    for _, row in states_df.iterrows():
        q, i = row["qqq_sc"], row["ivv_sc"]
        if q >= 3:
            if i <= 1: v9s = "QQQ"  # 1x guard-off
            else: v9s = "QLD"
        elif q == 2:
            v9s = "QQQ70"
        else:
            v9s = "CASH"
        v9_state_labels.append(v9s)
    states_df["v9_state"] = v9_state_labels

    # V13 delever months where V9 would be in something different
    print(f"\n  V13 DELEVER month breakdown:")
    dl = states_df[states_df["state"] == "DELEVER"]
    for v9s in dl["v9_state"].unique():
        n = (dl["v9_state"] == v9s).sum()
        print(f"    while V9 would be in {v9s}: {n} months")

    # ── TABLE 4: Weekly re-entry diagnostics ──
    print(f"\n{'=' * 140}")
    print("  TABLE 4: WEEKLY RE-ENTRY DIAGNOSTICS")
    print(f"{'=' * 140}")
    print(f"\n  Total CB events: {v13_cb_count}")

    resolved_weekly = [e for e in v13_cb_events if e["resolution"] == "weekly"]
    resolved_monthly = [e for e in v13_cb_events if e["resolution"] == "monthly"]
    unlev_reentries = [e for e in v13_cb_events if e["weekly_unlev_date"] is not None]

    print(f"  Resolved via weekly (full relevered): {len(resolved_weekly)}")
    print(f"  Resolved via monthly rebalance:       {len(resolved_monthly)}")
    print(f"  Weekly unlevered re-entries (any):    {len(unlev_reentries)}")

    if unlev_reentries:
        days_to_unlev = [(e["weekly_unlev_date"] - e["date"]).days for e in unlev_reentries]
        print(f"  Mean days CB → weekly unlevered re-entry: {np.mean(days_to_unlev):.1f}")
        print(f"  Median: {np.median(days_to_unlev):.0f}")
    if resolved_weekly:
        days_to_lev = [(e["weekly_lev_date"] - e["date"]).days for e in resolved_weekly]
        print(f"  Mean days CB → weekly full QLD re-entry: {np.mean(days_to_lev):.1f}")
        print(f"  Median: {np.median(days_to_lev):.0f}")
    if resolved_monthly:
        days_to_monthly = [(e["resolution_date"] - e["date"]).days for e in resolved_monthly]
        print(f"  Mean days CB → monthly rebalance resolution: {np.mean(days_to_monthly):.1f}")

    whipsaws = sum(1 for e in v13_cb_events if e["whipsaw_within_30d"])
    wr = whipsaws / v13_cb_count if v13_cb_count else 0
    print(f"\n  Whipsaw events (CB fired again within 30d of resolution): {whipsaws}/{v13_cb_count} ({wr:.0%})")

    # List every CB event
    print(f"\n  {'CB Date':<12}{'PriorHold':>11}{'Weekly→QQQ':>14}{'Weekly→QLD':>14}{'Resolution':>13}{'Whipsaw':>10}")
    print(f"  {'-' * 12}{'-' * 10:>11}{'-' * 13:>14}{'-' * 13:>14}{'-' * 12:>13}{'-' * 9:>10}")
    for e in v13_cb_events:
        wu = e["weekly_unlev_date"].strftime("%Y-%m-%d") if e["weekly_unlev_date"] else "—"
        wl = e["weekly_lev_date"].strftime("%Y-%m-%d") if e["weekly_lev_date"] else "—"
        print(f"  {e['date'].strftime('%Y-%m-%d'):<12}{e['prior_holding']:>11}{wu:>14}{wl:>14}"
              f"{(e['resolution'] or '—'):>13}{'YES' if e['whipsaw_within_30d'] else '—':>10}")

    # ── TABLE 5: Delever state analysis ──
    print(f"\n{'=' * 140}")
    print("  TABLE 5: DELEVER STATE RETURN ANALYSIS")
    print(f"{'=' * 140}")

    # Build daily-holding DataFrame: for each day, V13 holding vs what V9 would hold
    v13_holding_series = pd.Series(v13_holdings)
    # Get QQQ monthly returns and cash returns
    v13_monthly_ret = v13_full.resample("MS").apply(lambda x: (1 + x).prod() - 1)
    v9_monthly_ret = v9_full.resample("MS").apply(lambda x: (1 + x).prod() - 1)

    # Months where V13 monthly state = DELEVER
    dl_months = states_df[states_df["state"] == "DELEVER"]
    if len(dl_months) > 0:
        v13_dl_rets = []; v9_dl_rets = []
        for _, row in dl_months.iterrows():
            ts = pd.Timestamp(row["month"].year, row["month"].month, 1)
            r13 = v13_monthly_ret.get(ts, np.nan); r9 = v9_monthly_ret.get(ts, np.nan)
            if not pd.isna(r13) and not pd.isna(r9):
                v13_dl_rets.append(r13); v9_dl_rets.append(r9)
        v13_mean = np.mean(v13_dl_rets); v9_mean = np.mean(v9_dl_rets)
        pos_count = sum(1 for r in v13_dl_rets if r > 0)
        print(f"\n  V13 DELEVER months: {len(v13_dl_rets)}")
        print(f"  Mean V13 return in these months: {v13_mean:.2%}")
        print(f"  Mean V9 return in these months:  {v9_mean:.2%}")
        print(f"  Delta (V13 - V9):                {v13_mean - v9_mean:+.2%}")
        print(f"  Months with positive V13 return: {pos_count}/{len(v13_dl_rets)} ({pos_count/len(v13_dl_rets):.0%})")
    else:
        print("  No DELEVER months observed.")

    # ── TABLE 6: 2022 month-by-month ──
    print(f"\n{'=' * 140}")
    print("  TABLE 6: 2022 MONTH-BY-MONTH")
    print(f"{'=' * 140}")
    print(f"\n  {'Month':<10}{'QQQ':>5}{'IVV':>5}{'V13state':>12}{'V13 ret':>10}{'V9 ret':>10}{'BL ret':>10}")
    print(f"  {'-'*10}{'-'*4:>5}{'-'*4:>5}{'-'*11:>12}{'-'*9:>10}{'-'*9:>10}{'-'*9:>10}")
    bl_monthly = bl_full.resample("MS").apply(lambda x: (1+x).prod()-1)
    for _, row in states_df.iterrows():
        if row["month"].year == 2022:
            ts = pd.Timestamp(row["month"].year, row["month"].month, 1)
            r13 = v13_monthly_ret.get(ts, np.nan); r9 = v9_monthly_ret.get(ts, np.nan); rbl = bl_monthly.get(ts, np.nan)
            print(f"  {row['month'].strftime('%Y-%m'):<10}{row['qqq_sc']:>5}{row['ivv_sc']:>5}"
                  f"{row['state']:>12}{r13:>10.2%}{r9:>10.2%}{rbl:>10.2%}")

    # ── TABLE 7: Crisis drawdowns ──
    print(f"\n{'=' * 140}")
    print("  TABLE 7: CRISIS DRAWDOWNS")
    print(f"{'=' * 140}")
    crises = [
        ("Dot-com 02-03", "2002-01-01", "2003-03-31"),
        ("GFC 07-09",     "2007-11-01", "2009-03-31"),
        ("COVID 2020",    "2020-02-01", "2020-04-30"),
        ("2022 bear",     "2022-01-01", "2022-12-31"),
    ]
    print(f"\n  {'Crisis':<18}{'V13':>10}{'V9':>10}{'V12':>10}{'Baseline':>12}{'QQQ B&H':>12}")
    print(f"  {'-' * 18}{'-' * 9:>10}{'-' * 9:>10}{'-' * 9:>10}{'-' * 11:>12}{'-' * 11:>12}")
    for label, cs, ce in crises:
        cells = []
        for s in [v13_full, v9_full, v12_full, bl_full, qqq_full]:
            sp = s[(s.index >= pd.Timestamp(cs)) & (s.index <= pd.Timestamp(ce))]
            cells.append(max_dd(sp) if len(sp) > 5 else 0)
        print(f"  {label:<18}{cells[0]:>10.1%}{cells[1]:>10.1%}{cells[2]:>10.1%}"
              f"{cells[3]:>12.1%}{cells[4]:>12.1%}")

    # ── TABLE 8: Recovery speed (GFC, COVID, 2022) ──
    print(f"\n{'=' * 140}")
    print("  TABLE 8: RECOVERY-PERIOD RETURNS")
    print(f"{'=' * 140}")
    recoveries = [
        ("GFC trough → 1yr", "2009-03-09", "2010-03-09"),
        ("COVID trough → 6mo", "2020-03-23", "2020-09-23"),
        ("2022 trough → 6mo", "2022-10-12", "2023-04-12"),
    ]
    print(f"\n  {'Window':<22}{'V13':>10}{'V9':>10}{'V12':>10}{'QQQ B&H':>12}")
    print(f"  {'-'*22}{'-'*9:>10}{'-'*9:>10}{'-'*9:>10}{'-'*11:>12}")
    for label, cs, ce in recoveries:
        cells = []
        for s in [v13_full, v9_full, v12_full, qqq_full]:
            sp = s[(s.index >= pd.Timestamp(cs)) & (s.index <= pd.Timestamp(ce))]
            tot = (1 + sp).prod() - 1 if len(sp) > 5 else 0
            cells.append(tot)
        print(f"  {label:<22}{cells[0]:>10.2%}{cells[1]:>10.2%}{cells[2]:>10.2%}{cells[3]:>12.2%}")

    # ── TABLE 9: DCA dollar gap ──
    print(f"\n{'=' * 140}")
    print("  TABLE 9: DCA TERMINAL BY YEAR-END (2013 start)")
    print(f"{'=' * 140}")
    print(f"\n  {'Year':<6}{'V13':>13}{'V9':>13}{'V12':>13}{'BL':>13}{'QQQ':>13}{'V13-QQQ':>13}")
    print(f"  {'-' * 6}{'-' * 12:>13}{'-' * 12:>13}{'-' * 12:>13}{'-' * 12:>13}{'-' * 12:>13}{'-' * 12:>13}")
    for yr in range(2013, 2027):
        end = f"{yr}-12-31"
        vals = {}
        for nm, s in [("V13", v13_full), ("V9", v9_full), ("V12", v12_full), ("BL", bl_full)]:
            sp = s[(s.index >= "2013-01-01") & (s.index <= end)]
            sm = sp.resample("MS").apply(lambda x: (1 + x).prod() - 1)
            vals[nm] = dca_terminal(sm)
        qs = qqq_full[(qqq_full.index >= "2013-01-01") & (qqq_full.index <= end)]
        qm = qs.resample("MS").apply(lambda x: (1 + x).prod() - 1)
        vals["QQQ"] = dca_terminal(qm)
        print(f"  {yr:<6}${vals['V13']/1e3:>11.0f}K ${vals['V9']/1e3:>11.0f}K "
              f"${vals['V12']/1e3:>11.0f}K ${vals['BL']/1e3:>11.0f}K ${vals['QQQ']/1e3:>11.0f}K "
              f"${(vals['V13']-vals['QQQ'])/1e3:>11.0f}K")

    # ── TABLE 10: Pass/fail ──
    print(f"\n{'=' * 140}")
    print("  TABLE 10: PASS / FAIL")
    print(f"{'=' * 140}")
    v13_c = cagr(v13_full); v13_sh = sharpe_r(v13_full); v13_dd = max_dd(v13_full)
    v9_c = cagr(v9_full); v9_sh = sharpe_r(v9_full); v9_dd = max_dd(v9_full)

    print(f"\n  V13: CAGR {v13_c:.2%}, Sharpe {v13_sh:.3f}, MaxDD {v13_dd:.1%}")
    print(f"  V9:  CAGR {v9_c:.2%}, Sharpe {v9_sh:.3f}, MaxDD {v9_dd:.1%}")

    path_a = (v13_dd > v9_dd) and (v13_c >= v9_c - 0.01) and (v13_sh >= v9_sh)
    path_b = (v13_c > v9_c) and (v13_dd >= v9_dd) and (v13_sh >= v9_sh)
    path_c = (v13_sh > 0.80) and (v13_c >= v9_c - 0.02)

    print(f"\n  Path A (DD improvement):     {'PASS' if path_a else 'FAIL'}")
    print(f"     Max DD shallower:          {'✓' if v13_dd > v9_dd else '✗'} ({v13_dd:.1%} vs {v9_dd:.1%})")
    print(f"     CAGR within 1pp:           {'✓' if v13_c >= v9_c - 0.01 else '✗'}")
    print(f"     Sharpe ≥ V9:               {'✓' if v13_sh >= v9_sh else '✗'}")

    print(f"\n  Path B (Return improvement): {'PASS' if path_b else 'FAIL'}")
    print(f"     CAGR higher:               {'✓' if v13_c > v9_c else '✗'}")
    print(f"     MaxDD not worse:           {'✓' if v13_dd >= v9_dd else '✗'}")
    print(f"     Sharpe ≥ V9:               {'✓' if v13_sh >= v9_sh else '✗'}")

    print(f"\n  Path C (Sharpe improvement): {'PASS' if path_c else 'FAIL'}")
    print(f"     Sharpe > 0.80:             {'✓' if v13_sh > 0.80 else '✗'}")

    # Fail conditions
    fail_whipsaw = wr > 0.50
    print(f"\n  Fail checks:")
    print(f"     Whipsaw rate > 50%: {'FAIL' if fail_whipsaw else 'ok'} ({wr:.0%})")
    if len(dl_months) > 0 and len(v13_dl_rets) > 0:
        v13_mean = np.mean(v13_dl_rets)
        print(f"     Delever mean return < 0: {'FAIL' if v13_mean < 0 else 'ok'} ({v13_mean:.2%})")
    print(f"     MaxDD worse than V9: {'FAIL' if v13_dd < v9_dd else 'ok'}")

    passed = path_a or path_b or path_c
    print(f"\n  OVERALL: {'PASS' if passed else 'FAIL'}")
    print()


if __name__ == "__main__":
    main()
