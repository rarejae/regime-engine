"""V9-DCA: Cash Redeployment via Stepped IVV Buying During V9 Off-Signal.

V9's offense is byte-for-byte identical. During cash periods, systematically
buy IVV in fixed tranches as price declines from an anchor. Sell all IVV when
V9's QLD signal restores.

Variants: step sizes 3%, 5%, 7%, 10% (10 tranches each).
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
from experiments.v12_independent_2x.backtest import run_v12


def run_v9_dca(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
               start_date, step_pct=0.05, tranche_pct=0.10, max_tranches=10):
    """V9 + DCA into IVV during cash periods."""
    bt_start = pd.Timestamp(start_date)
    common_start = max(daily_ret.dropna(how="all").index.min(), bt_start)
    trading_days = daily_ret.loc[common_start:pd.Timestamp("2026-03-31")].index

    port = {}; cb_count = 0

    # V9 state
    mode = "cash"  # "qld", "qqq", "cash"
    lev = False; delevered = False
    scores = {"QQQ": 0, "IVV": 0}

    # DCA state
    anchor_price = None
    tranches_deployed = 0
    ivv_weight = 0.0    # fraction of portfolio in IVV
    cash_weight = 1.0   # fraction in cash

    # Event log
    dca_events = []
    current_event = None

    def _start_dca(day):
        nonlocal anchor_price, tranches_deployed, ivv_weight, cash_weight, current_event
        ivv_p = dpdf.loc[:day, "IVV"]
        if len(ivv_p) == 0: return
        anchor_price = float(ivv_p.iloc[-1])
        tranches_deployed = 0; ivv_weight = 0.0; cash_weight = 1.0
        current_event = {
            "anchor_date": day, "anchor_price": anchor_price,
            "tranches": [], "exit_date": None, "exit_price": None,
            "max_decline": 0.0, "dca_return": 0.0, "cash_return": 0.0,
        }

    def _close_dca(day):
        nonlocal anchor_price, tranches_deployed, ivv_weight, cash_weight, current_event
        if current_event is not None:
            ivv_p = dpdf.loc[:day, "IVV"]
            current_event["exit_date"] = day
            current_event["exit_price"] = float(ivv_p.iloc[-1]) if len(ivv_p) > 0 else 0
            current_event["tranches_deployed"] = tranches_deployed
            current_event["ivv_weight_at_exit"] = ivv_weight
            dca_events.append(current_event)
            current_event = None
        anchor_price = None; tranches_deployed = 0; ivv_weight = 0.0; cash_weight = 1.0

    for day in trading_days:
        dr = daily_ret.loc[day]
        is_ms = (day == trading_days[0] or day.month != trading_days[trading_days.get_loc(day) - 1].month)

        if is_ms:
            delevered = False
            prior = trading_days[trading_days < day]
            sd = prior[-1] if len(prior) > 0 else day
            scores = {"QQQ": asset_score(sd, "QQQ", dpdf, daily_smas),
                      "IVV": asset_score(sd, "IVV", dpdf, daily_smas)}

            sc_q = scores["QQQ"]; sc_i = scores["IVV"]
            prev_mode = mode

            if sc_q >= 3:
                if sc_i <= 1:
                    mode = "qqq"; lev = False
                else:
                    mode = "qld"; lev = True
            elif sc_q == 2:
                mode = "qqq"; lev = False  # 70% QQQ + 30% cash
            else:
                mode = "cash"; lev = False

            # Handle DCA transitions
            if prev_mode == "cash" and mode != "cash":
                _close_dca(day)
            elif prev_mode != "cash" and mode == "cash":
                _start_dca(day)
            elif prev_mode == "qqq" and mode == "cash":
                _start_dca(day)

        # Daily CB (only when in QLD)
        if lev and not delevered:
            if check_breach(day, "QQQ", dpdf, daily_smas):
                lev = False; delevered = True; cb_count += 1
                mode = "qqq"  # QLD → QQQ per V9

        # DCA check (only when in cash mode with anchor set)
        if mode == "cash" and anchor_price is not None:
            ivv_p = dpdf.loc[:day, "IVV"]
            if len(ivv_p) > 0:
                cur_price = float(ivv_p.iloc[-1])
                decline = (anchor_price - cur_price) / anchor_price
                if decline > 0 and current_event is not None:
                    current_event["max_decline"] = max(current_event["max_decline"], decline)

                target = min(int(decline / step_pct), max_tranches) if decline > 0 else 0
                while tranches_deployed < target and cash_weight >= tranche_pct - 0.001:
                    tranches_deployed += 1
                    ivv_weight += tranche_pct
                    cash_weight -= tranche_pct
                    if current_event is not None:
                        current_event["tranches"].append({
                            "date": day, "price": cur_price,
                            "decline": decline, "tranche_num": tranches_deployed,
                        })

        # Compute return
        rfr = float(rfr_daily.get(day, 0.0))
        qqq_u = float(dr.get("QQQ", 0.0)) if pd.notna(dr.get("QQQ", np.nan)) else 0.0
        ivv_u = float(dr.get("IVV", 0.0)) if pd.notna(dr.get("IVV", np.nan)) else 0.0

        if mode == "qld":
            if lev:
                ret = lev_ret(qqq_u, rfr, QLD_EXP, day, actual_lev, "QLD", both_start)
            else:
                ret = qqq_u
        elif mode == "qqq":
            if scores["QQQ"] == 2 and not delevered:
                ret = 0.70 * qqq_u + 0.30 * rfr
            else:
                ret = qqq_u
        elif mode == "cash":
            # Blended: ivv_weight in IVV + cash_weight in cash
            ret = ivv_weight * ivv_u + cash_weight * rfr
        else:
            ret = rfr

        port[day] = ret

    # Close any open DCA event
    if current_event is not None:
        last_day = trading_days[-1]
        ivv_p = dpdf.loc[:last_day, "IVV"]
        current_event["exit_date"] = last_day
        current_event["exit_price"] = float(ivv_p.iloc[-1]) if len(ivv_p) > 0 else 0
        current_event["tranches_deployed"] = tranches_deployed
        current_event["ivv_weight_at_exit"] = ivv_weight
        dca_events.append(current_event)

    # Compute per-event returns
    s = pd.Series(port).sort_index()
    for ev in dca_events:
        if ev["exit_date"] and ev["anchor_date"]:
            sp = s[(s.index >= ev["anchor_date"]) & (s.index <= ev["exit_date"])]
            ev["dca_return"] = (1 + sp).prod() - 1 if len(sp) > 0 else 0
            # V9 cash return for same period (T-bills)
            rfr_sp = rfr_daily.reindex(sp.index).fillna(0)
            ev["cash_return"] = (1 + rfr_sp).prod() - 1 if len(rfr_sp) > 0 else 0

    return s, cb_count, dca_events


def main():
    print("=" * 140)
    print("  V9-DCA: CASH REDEPLOYMENT VIA STEPPED IVV BUYING — FULL BACKTEST")
    print("=" * 140)

    print("\n  Loading data...")
    daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, dbmf_ret, dbmf_inception = load_data()

    step_sizes = [0.03, 0.05, 0.07, 0.10]
    variants = {}
    for sp in step_sizes:
        name = f"V9-DCA-{int(sp*100)}"
        print(f"  Running {name}...")
        s, cb, events = run_v9_dca(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                                    "2002-01-01", step_pct=sp)
        variants[name] = (s, cb, events)

    print("  Running V9 (control)...")
    v9_full, v9_cb = run_v9(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, "2002-01-01")
    print("  Running V12...")
    v12_full, v12_cb, _ = run_v12(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, "2002-01-01")
    print("  Running Baseline...")
    bl_full, bl_cb = run_baseline(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                                   dbmf_ret, dbmf_inception, "2002-01-01")
    qqq_full = daily_ret["QQQ"].loc["2002-01-01":"2026-03-31"].dropna()

    # ── TABLE 1: Core metrics ──
    print(f"\n{'=' * 140}")
    print("  TABLE 1: CORE METRICS (2002-2026)")
    print(f"{'=' * 140}")
    print(f"\n  {'Strategy':<22} {'CAGR':>7} {'Vol':>7} {'Sharpe':>7} {'Sortino':>8} "
          f"{'MaxDD':>7} {'Calmar':>7} {'Term$1':>9} {'DCA$700':>9} {'CB':>4}")
    print(f"  {'-' * 22} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 8} {'-' * 7} {'-' * 7} {'-' * 9} {'-' * 9} {'-' * 4}")
    for nm in sorted(variants.keys()):
        s, cb, _ = variants[nm]
        print(metrics_row(nm, s, cb))
    print(metrics_row("V9 (control)", v9_full, v9_cb))
    print(metrics_row("V12 Independent 2×", v12_full, v12_cb))
    print(metrics_row("Baseline (Sweep-40)", bl_full, bl_cb))
    print(metrics_row("QQQ B&H", qqq_full))

    # ── TABLE 2: Start-date sensitivity ──
    print(f"\n{'=' * 140}")
    print("  TABLE 2: CAGR BY START DATE")
    print(f"{'=' * 140}")
    start_dates = ["2002-01-01", "2007-01-01", "2010-01-01", "2013-01-01", "2019-01-01"]
    print(f"\n  {'Strategy':<22}" + "".join(f"{sd[:4]:>10}" for sd in start_dates))
    print(f"  {'-' * 22}" + "".join(f" {'-' * 9}" for _ in start_dates))
    for nm in sorted(variants.keys()):
        row = f"  {nm:<22}"
        sp_val = int(nm.split("-")[-1]) / 100
        for sd in start_dates:
            s, _, _ = run_v9_dca(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                                  sd, step_pct=sp_val)
            row += f"{cagr(s):>10.2%}"
        print(row)
    for nm, fn in [("V9 (control)", "v9"), ("V12 Indep 2×", "v12")]:
        row = f"  {nm:<22}"
        for sd in start_dates:
            if fn == "v9":
                s, _ = run_v9(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, sd)
            else:
                s, _, _ = run_v12(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, sd)
            row += f"{cagr(s):>10.2%}"
        print(row)
    row = f"  {'QQQ B&H':<22}"
    for sd in start_dates:
        qs = qqq_full[qqq_full.index >= pd.Timestamp(sd)]
        row += f"{cagr(qs):>10.2%}"
    print(row)

    # ── TABLE 3: DCA event log (for 5% step) ──
    print(f"\n{'=' * 140}")
    print("  TABLE 3: DCA EVENT LOG (V9-DCA-5, step=5%)")
    print(f"{'=' * 140}")
    events_5 = variants["V9-DCA-5"][2]
    print(f"\n  Total cash periods with DCA: {len(events_5)}")
    dca_wins = sum(1 for e in events_5 if e["dca_return"] > e["cash_return"])
    print(f"  DCA beat T-bills: {dca_wins}/{len(events_5)} ({dca_wins/len(events_5):.0%})" if events_5 else "  No events")
    print(f"\n  {'Anchor':>12}{'Exit':>12}{'Days':>6}{'MaxDecl':>9}{'Tranch':>7}{'DCA ret':>9}{'Cash ret':>10}{'Winner':>8}")
    print(f"  {'-'*12}{'-'*11:>12}{'-'*5:>6}{'-'*8:>9}{'-'*6:>7}{'-'*8:>9}{'-'*9:>10}{'-'*7:>8}")
    for e in events_5:
        days = (e["exit_date"] - e["anchor_date"]).days if e["exit_date"] and e["anchor_date"] else 0
        winner = "DCA" if e["dca_return"] > e["cash_return"] else "CASH"
        print(f"  {e['anchor_date'].strftime('%Y-%m-%d'):>12}{e['exit_date'].strftime('%Y-%m-%d'):>12}"
              f"{days:>6}{e['max_decline']:>8.1%}{e.get('tranches_deployed',0):>7}"
              f"{e['dca_return']:>8.2%}{e['cash_return']:>9.2%}{winner:>8}")

    # Summary across all step sizes
    print(f"\n  DCA win rates across step sizes:")
    for nm in sorted(variants.keys()):
        evts = variants[nm][2]
        wins = sum(1 for e in evts if e["dca_return"] > e["cash_return"])
        deployed = [e for e in evts if e.get("tranches_deployed", 0) > 0]
        wins_dep = sum(1 for e in deployed if e["dca_return"] > e["cash_return"])
        total_dca = sum(e["dca_return"] for e in evts)
        total_cash = sum(e["cash_return"] for e in evts)
        print(f"    {nm}: {wins}/{len(evts)} overall, {wins_dep}/{len(deployed)} when tranches deployed, "
              f"cumul DCA {total_dca:+.2%} vs cash {total_cash:+.2%}")

    # ── TABLE 4: Crisis detail ──
    print(f"\n{'=' * 140}")
    print("  TABLE 4: PER-CRISIS DCA DETAIL (V9-DCA-5)")
    print(f"{'=' * 140}")
    for crisis_name, start, end in [("GFC", "2007-10-01", "2009-06-30"),
                                      ("COVID", "2020-02-01", "2020-06-30"),
                                      ("2022 Bear", "2022-01-01", "2023-03-31")]:
        crisis_events = [e for e in events_5
                         if e["anchor_date"] >= pd.Timestamp(start) and e["anchor_date"] <= pd.Timestamp(end)]
        print(f"\n  {crisis_name}:")
        if not crisis_events:
            print("    No DCA events in this period")
            continue
        for e in crisis_events:
            days = (e["exit_date"] - e["anchor_date"]).days
            print(f"    Anchor: {e['anchor_date'].strftime('%Y-%m-%d')} (IVV ${e['anchor_price']:.2f})")
            print(f"    Exit:   {e['exit_date'].strftime('%Y-%m-%d')} (IVV ${e['exit_price']:.2f}) — {days} days")
            print(f"    Max decline: {e['max_decline']:.1%}, Tranches deployed: {e.get('tranches_deployed',0)}")
            if e["tranches"]:
                avg_cost = np.mean([t["price"] for t in e["tranches"]])
                print(f"    Avg cost basis: ${avg_cost:.2f}, Exit price: ${e['exit_price']:.2f} "
                      f"→ {'profit' if e['exit_price'] > avg_cost else 'loss'} ({(e['exit_price']/avg_cost-1)*100:+.1f}%)")
            print(f"    DCA period return: {e['dca_return']:+.2%} vs T-bills: {e['cash_return']:+.2%} "
                  f"→ delta {e['dca_return']-e['cash_return']:+.2%}")

    # ── TABLE 5: Max DD comparison ──
    print(f"\n{'=' * 140}")
    print("  TABLE 5: MAX DRAWDOWN COMPARISON")
    print(f"{'=' * 140}")
    print(f"\n  {'Strategy':<22}{'Full DD':>10}{'GFC DD':>10}{'COVID DD':>10}{'2022 DD':>10}")
    print(f"  {'-'*22}{'-'*9:>10}{'-'*9:>10}{'-'*9:>10}{'-'*9:>10}")
    crises = [("2007-11-01","2009-03-31"), ("2020-02-01","2020-04-30"), ("2022-01-01","2022-12-31")]
    for nm in sorted(variants.keys()):
        s = variants[nm][0]
        row = f"  {nm:<22}{max_dd(s):>10.1%}"
        for cs, ce in crises:
            sp = s[(s.index >= pd.Timestamp(cs)) & (s.index <= pd.Timestamp(ce))]
            row += f"{max_dd(sp):>10.1%}" if len(sp) > 5 else f"{'N/A':>10}"
        print(row)
    row = f"  {'V9 (control)':<22}{max_dd(v9_full):>10.1%}"
    for cs, ce in crises:
        sp = v9_full[(v9_full.index >= pd.Timestamp(cs)) & (v9_full.index <= pd.Timestamp(ce))]
        row += f"{max_dd(sp):>10.1%}"
    print(row)

    # ── TABLE 6: DCA dollar gap ──
    print(f"\n{'=' * 140}")
    print("  TABLE 6: DCA TERMINAL BY YEAR-END (2013 start)")
    print(f"{'=' * 140}")
    show = sorted(variants.keys()) + ["V9 (control)"]
    print(f"\n  {'Year':<6}" + "".join(f"{nm:>15}" for nm in show) + f"{'QQQ':>13}")
    for yr in range(2013, 2027):
        end = f"{yr}-12-31"
        row = f"  {yr:<6}"
        for nm in show:
            s = variants[nm][0] if nm in variants else v9_full
            sp = s[(s.index >= "2013-01-01") & (s.index <= end)]
            sm = sp.resample("MS").apply(lambda x: (1+x).prod()-1)
            row += f"${dca_terminal(sm)/1e3:>13.0f}K"
        qs = qqq_full[(qqq_full.index >= "2013-01-01") & (qqq_full.index <= end)]
        qm = qs.resample("MS").apply(lambda x: (1+x).prod()-1)
        row += f"${dca_terminal(qm)/1e3:>11.0f}K"
        print(row)

    # ── TABLE 7: Pass/fail ──
    print(f"\n{'=' * 140}")
    print("  TABLE 7: PASS / FAIL")
    print(f"{'=' * 140}")
    v9_c = cagr(v9_full); v9_sh = sharpe_r(v9_full); v9_dd = max_dd(v9_full)
    print(f"\n  V9 reference: CAGR {v9_c:.2%}, Sharpe {v9_sh:.3f}, MaxDD {v9_dd:.1%}")

    for nm in sorted(variants.keys()):
        s = variants[nm][0]
        vc = cagr(s); vsh = sharpe_r(s); vdd = max_dd(s)
        cagr_ok = vc > v9_c
        sh_ok = vsh >= v9_sh
        dd_ok = vdd >= v9_dd - 0.03  # within 3pp
        evts = variants[nm][2]
        wins = sum(1 for e in evts if e["dca_return"] > e["cash_return"])
        mech_ok = wins > len(evts) / 2

        passed = cagr_ok and sh_ok and dd_ok
        print(f"\n  {nm}:")
        print(f"    CAGR:      {vc:.2%} vs V9 {v9_c:.2%} → {'✓' if cagr_ok else '✗'}")
        print(f"    Sharpe:    {vsh:.3f} vs V9 {v9_sh:.3f} → {'✓' if sh_ok else '✗'}")
        print(f"    MaxDD:     {vdd:.1%} vs V9 {v9_dd:.1%} (≤ -40.9%) → {'✓' if dd_ok else '✗'}")
        print(f"    DCA wins:  {wins}/{len(evts)} ({wins/len(evts):.0%}) → {'✓' if mech_ok else '✗'}")
        print(f"    → {'PASS' if passed else 'FAIL'}")

    print()


if __name__ == "__main__":
    main()
