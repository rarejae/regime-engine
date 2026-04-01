"""Phase 1 + graduated leverage with weekly de-lever checks."""

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv; load_dotenv()

import numpy as np, pandas as pd

from taa.data import (load_daily_etf_returns, load_monthly_prices,
                       load_realized_vols, load_monthly_macro, load_monthly_asset_returns)
from taa.faber import compute_trend_scores, apply_faber_filter
from taa.harvey import compute_zscore_variables, find_similar_months, compute_expected_returns
from taa.allocation import BASELINE, ASSETS, EQUITY, direct_capital, normalize

OUTPUT = Path("taa/output"); OUTPUT.mkdir(parents=True, exist_ok=True)
SMA_PERIODS = [6, 10, 12]
TIER_SUBS = {0: 0.0, 1: 0.30, 2: 0.65}
SEED = {"IVV": 0.0028, "QQQ": 0.0080}

print("=" * 80)
print("  PHASE 1 + WEEKLY LEVERAGE MONITORING (1.3x / 1.65x)")
print("=" * 80)

# Load data
daily_ret = load_daily_etf_returns()
daily_ret = daily_ret[[c for c in daily_ret.columns if c in ASSETS]]
monthly_prices = load_monthly_prices()
rvol_monthly = load_realized_vols()
asset_ret = load_monthly_asset_returns()
asset_ret_fwd = asset_ret.shift(-1)
trend_df = compute_trend_scores(monthly_prices)
z_data = compute_zscore_variables(load_monthly_macro())
z_clean = z_data[[c for c in z_data.columns if c.endswith("_z")]].dropna()

from fredapi import Fred
rfr_daily = pd.Series(0.0, index=daily_ret.index)
key = os.environ.get("FRED_API_KEY")
if key:
    tb = Fred(api_key=key).get_series("DTB3", observation_start="1998-01-01")
    tb.index = pd.to_datetime(tb.index)
    rfr_daily = (tb / 100 / 252).reindex(daily_ret.index, method="ffill").fillna(0)

# Daily prices for weekly SMA checks
import yfinance as yf
daily_prices = {}
for our, ticker in [("IVV","SPY"),("QQQ","QQQ")]:
    d = yf.download(ticker, start="1998-01-01", progress=False)
    if d is not None and not d.empty:
        p = d["Close"]
        if hasattr(p,"columns"): p=p.iloc[:,0]
        p.index = pd.to_datetime(p.index).tz_localize(None)
        daily_prices[our] = p
daily_prices_df = pd.DataFrame(daily_prices)

# Precompute monthly SMAs for weekly checks (using monthly closes)
monthly_close = daily_prices_df.resample("MS").last()
sma_6m = monthly_close.rolling(6, min_periods=6).mean()
sma_10m = monthly_close.rolling(10, min_periods=10).mean()
sma_12m = monthly_close.rolling(12, min_periods=12).mean()

# Backtest
common_start = max(daily_ret.dropna(how="all").index.min(), pd.Timestamp("2002-01-01"))
trading_days = daily_ret.loc[common_start:].index
print(f"Backtest: {len(trading_days)} days ({common_start.date()} to {trading_days.max().date()})")

# State
ct = {a: 3 for a in ASSETS if a != "cash"}
he = {a: 0.0 for a in ASSETS}
wp = dict(BASELINE)
tier_monthly = 0  # tier from monthly check
tier_active_weekly = 0  # effective tier after weekly checks
tier_active_monthly_only = 0  # for comparison (no weekly check)
delevered_this_month = False

# Expanding medians (seeded)
from taa.harvey import find_similar_months as fsm
pre_ers = {"IVV": [], "QQQ": []}
pre_months = z_clean.index[z_clean.index < common_start]
for z_dt in pre_months:
    try:
        sim, _ = fsm(z_clean, z_dt)
        er = compute_expected_returns(sim, asset_ret_fwd, ["IVV","QQQ"])
        for a in ["IVV","QQQ"]:
            if not np.isnan(er.get(a, np.nan)):
                pre_ers[a].append(er[a])
    except: pass

tier1_hist = {"IVV": list(pre_ers["IVV"]), "QQQ": list(pre_ers["QQQ"])}
tier1_med = {"IVV": SEED["IVV"], "QQQ": SEED["QQQ"]}
# Duplicate for monthly-only comparison
tier1_hist_mo = {"IVV": list(pre_ers["IVV"]), "QQQ": list(pre_ers["QQQ"])}
tier1_med_mo = {"IVV": SEED["IVV"], "QQQ": SEED["QQQ"]}

weekly_rets = {}; monthly_rets = {}; unlev_rets = {}; ivv_rets = {}
delever_events = []
total_months = 0

for day in trading_days:
    if day not in daily_ret.index: continue
    dr = daily_ret.loc[day]
    avail = [a for a in ASSETS if a in dr.index and pd.notna(dr[a])]
    if len(avail) < 3: continue
    actual = {a: float(dr[a]) for a in avail}
    rfr = float(rfr_daily.get(day, 0))

    is_ms = (day == trading_days[0] or day.month != trading_days[trading_days.get_loc(day)-1].month)
    is_friday = day.weekday() == 4  # Friday

    if is_ms:
        total_months += 1
        delevered_this_month = False

        # Update signals
        ts_c = trend_df.index[trend_df.index <= day]
        if len(ts_c) > 0:
            for a in ASSETS:
                if a == "cash": continue
                if a in trend_df.columns:
                    v = trend_df.loc[ts_c[-1], a]
                    ct[a] = int(v) if pd.notna(v) else 0
        z_c = z_clean.index[z_clean.index < day]
        if len(z_c) > 0:
            try:
                sim, _ = find_similar_months(z_clean, z_c[-1])
                he = compute_expected_returns(sim, asset_ret_fwd, avail)
            except: pass
        rv_c = rvol_monthly.index[rvol_monthly.index <= day]
        rvols = {}
        if len(rv_c) > 0:
            for a in ASSETS:
                if a == "cash": continue
                if a in rvol_monthly.columns:
                    v = rvol_monthly.loc[rv_c[-1], a]
                    rvols[a] = float(v) if pd.notna(v) and v > 0 else 0.15

        w1, pool = apply_faber_filter(ct, BASELINE)
        w2, _ = direct_capital(dict(w1), pool, he, rvols)
        wp = normalize(w2)

        # Monthly tier determination
        f_c = ct.get("IVV",0)>=3 and ct.get("QQQ",0)>=3
        h_c = he.get("IVV",0)>0 and he.get("QQQ",0)>0
        conv = f_c and h_c

        if conv:
            above = all(he.get(a,0) > tier1_med.get(a,0) for a in ["IVV","QQQ"])
            tier_monthly = 2 if above else 1
            for a in ["IVV","QQQ"]:
                tier1_hist[a].append(he.get(a,0))
                tier1_med[a] = float(np.median(tier1_hist[a]))
        else:
            tier_monthly = 0

        # Monthly-only tier (for comparison)
        if conv:
            above_mo = all(he.get(a,0) > tier1_med_mo.get(a,0) for a in ["IVV","QQQ"])
            tier_active_monthly_only = 2 if above_mo else 1
            for a in ["IVV","QQQ"]:
                tier1_hist_mo[a].append(he.get(a,0))
                tier1_med_mo[a] = float(np.median(tier1_hist_mo[a]))
        else:
            tier_active_monthly_only = 0

        # Set active tier (may be overridden by weekly check)
        tier_active_weekly = tier_monthly

    # Weekly de-lever check (Friday close)
    if is_friday and tier_active_weekly > 0 and not delevered_this_month:
        # Check if IVV and QQQ are still above ALL SMAs using current daily prices
        breach = False
        breach_detail = ""
        for etf in ["IVV", "QQQ"]:
            if etf not in daily_prices_df.columns: continue
            price_today = daily_prices_df.loc[:day, etf].iloc[-1] if day in daily_prices_df.index or len(daily_prices_df.loc[:day]) > 0 else None
            if price_today is None: continue

            # Get most recent monthly SMAs (available at last month end)
            last_month_start = pd.Timestamp(f"{day.year}-{day.month:02d}-01")
            for period, sma_df in [(6, sma_6m), (10, sma_10m), (12, sma_12m)]:
                # Use the SMA from the most recent month-start that's <= today
                sma_dates = sma_df.index[sma_df.index <= last_month_start]
                if len(sma_dates) == 0: continue
                sma_val = sma_df.loc[sma_dates[-1], etf]
                if pd.notna(sma_val) and price_today < sma_val:
                    breach = True
                    breach_detail = f"{etf} ({price_today:.1f}) < {period}m SMA ({sma_val:.1f})"
                    break
            if breach: break

        if breach:
            delever_events.append({
                "date": day, "from_tier": tier_active_weekly,
                "detail": breach_detail, "month": day.strftime("%Y-%m"),
            })
            tier_active_weekly = 0
            delevered_this_month = True

    # Compute daily returns
    iw = wp.get("IVV",0); qw = wp.get("QQQ",0)
    ir = actual.get("IVV",0); qr = actual.get("QQQ",0)
    sr = 2*ir - rfr - 0.0091/252; ql = 2*qr - rfr - 0.0089/252
    base = sum(wp.get(a,0)*actual.get(a,0) for a in avail if a not in ["IVV","QQQ"])

    # 1x
    unlev_rets[day] = iw*ir + qw*qr + base
    ivv_rets[day] = actual.get("IVV", 0)

    # Weekly-monitored
    sub_w = TIER_SUBS.get(tier_active_weekly, 0)
    if sub_w > 0:
        weekly_rets[day] = iw*(1-sub_w)*ir + iw*sub_w*sr + qw*(1-sub_w)*qr + qw*sub_w*ql + base
    else:
        weekly_rets[day] = iw*ir + qw*qr + base

    # Monthly-only
    sub_m = TIER_SUBS.get(tier_active_monthly_only, 0)
    if sub_m > 0:
        monthly_rets[day] = iw*(1-sub_m)*ir + iw*sub_m*sr + qw*(1-sub_m)*qr + qw*sub_m*ql + base
    else:
        monthly_rets[day] = iw*ir + qw*qr + base

ws = pd.Series(weekly_rets).sort_index()
ms = pd.Series(monthly_rets).sort_index()
us = pd.Series(unlev_rets).sort_index()
ivv_s = pd.Series(ivv_rets).sort_index()

def metrics(s, label):
    ar=s.mean()*252; av=s.std()*np.sqrt(252); sh=ar/av if av>0 else 0
    neg=s[s<0]; ds=neg.std()*np.sqrt(252) if len(neg)>10 else av
    sortino=ar/ds if ds>0 else 0
    cum=(1+s).cumprod(); dd=((cum-cum.expanding().max())/cum.expanding().max()).min()
    calmar=ar/abs(dd) if dd!=0 else 0; final=cum.iloc[-1]
    return ar, av, sh, sortino, dd, calmar, final

# Report
print(f"\n{'='*80}")
print(f"  PERFORMANCE")
print(f"{'='*80}")
print(f"\n  {'Config':>18} {'AnnRet':>7} {'AnnVol':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>8} {'Calmar':>7} {'Final':>8}")
print(f"  {'-'*68}")
for s, label in [(ws,"Weekly 1.3/1.65"),(ms,"Monthly 1.3/1.65"),(us,"P1 1x"),(ivv_s,"IVV B&H")]:
    ar,av,sh,so,dd,cal,final = metrics(s, label)
    print(f"  {label:>18} {ar:>6.1%} {av:>6.1%} {sh:>7.2f} {so:>8.2f} {dd:>7.1%} {cal:>7.2f} ${final:>7.2f}")

# De-lever events
print(f"\n{'='*80}")
print(f"  MID-MONTH DE-LEVER EVENTS")
print(f"{'='*80}")
de_df = pd.DataFrame(delever_events)
print(f"  Total events: {len(de_df)}")
if len(de_df) > 0:
    # Average duration
    print(f"\n  {'Date':>10} {'From':>6} {'Detail'}")
    for _, r in de_df.iterrows():
        print(f"  {r['date'].strftime('%Y-%m-%d'):>10} Tier {r['from_tier']:>1} {r['detail']}")

# Crisis deep dive
print(f"\n{'='*80}")
print(f"  CRISIS ANALYSIS")
print(f"{'='*80}")
for cn2,cs,ce in [("GFC","2008-09-01","2009-03-31"),("COVID","2020-02-19","2020-03-23"),("2022","2022-01-03","2022-10-31")]:
    print(f"\n  {cn2}:")
    for s,label in [(ws,"Weekly"),(ms,"Monthly"),(us,"P1 1x"),(ivv_s,"IVV")]:
        c = s[(s.index>=pd.Timestamp(cs))&(s.index<=pd.Timestamp(ce))]
        if len(c)>0:
            cum=(1+c).cumprod(); mdd=((cum-cum.expanding().max())/cum.expanding().max()).min()
            print(f"    {label:>18}: {((1+c).prod()-1):>+7.1%} (DD {mdd:.1%})")

    # Check for mid-month de-levers during crisis
    crisis_events = de_df[(de_df["date"]>=pd.Timestamp(cs))&(de_df["date"]<=pd.Timestamp(ce))]
    if len(crisis_events) > 0:
        first = crisis_events.iloc[0]
        print(f"    First de-lever: {first['date'].strftime('%Y-%m-%d')} ({first['detail']})")
    else:
        print(f"    No mid-month de-levers during this crisis")

# COVID specific
print(f"\n{'='*80}")
print(f"  COVID DETAILED ANALYSIS")
print(f"{'='*80}")
covid_w = ws[(ws.index >= "2020-02-01") & (ws.index <= "2020-04-30")]
covid_m = ms[(ms.index >= "2020-02-01") & (ms.index <= "2020-04-30")]
if len(covid_w) > 0 and len(covid_m) > 0:
    cw_cum = (1+covid_w).cumprod()
    cm_cum = (1+covid_m).cumprod()
    print(f"  Feb-Apr 2020 total: Weekly={cw_cum.iloc[-1]-1:+.1%}, Monthly={cm_cum.iloc[-1]-1:+.1%}")
    print(f"  Weekly max DD: {((cw_cum-cw_cum.expanding().max())/cw_cum.expanding().max()).min():.1%}")
    print(f"  Monthly max DD: {((cm_cum-cm_cum.expanding().max())/cm_cum.expanding().max()).min():.1%}")

    covid_events = de_df[(de_df["date"]>="2020-02-01")&(de_df["date"]<="2020-04-30")]
    if len(covid_events) > 0:
        print(f"  De-lever trigger: {covid_events.iloc[0]['date'].strftime('%Y-%m-%d')} ({covid_events.iloc[0]['detail']})")
        # What was the value on trigger date vs month end?
        trigger_dt = covid_events.iloc[0]["date"]
        trig_val = cw_cum.loc[:trigger_dt].iloc[-1] if trigger_dt in cw_cum.index else "?"
        month_end = pd.Timestamp("2020-02-28")
        me_val_w = cw_cum.loc[:month_end].iloc[-1] if len(cw_cum.loc[:month_end]) > 0 else "?"
        me_val_m = cm_cum.loc[:month_end].iloc[-1] if len(cm_cum.loc[:month_end]) > 0 else "?"
        print(f"  Value at trigger: weekly={trig_val}")
        print(f"  Value at Feb end: weekly={me_val_w}, monthly={me_val_m}")

# Calendar years
print(f"\n{'='*80}")
print(f"  CALENDAR YEAR RETURNS")
print(f"{'='*80}")
print(f"  {'Year':>6} {'Weekly':>8} {'Monthly':>8} {'P1 1x':>8} {'IVV':>8}")
for yr in range(2002,2025):
    row=f"  {yr:>6}"
    for s in [ws,ms,us,ivv_s]:
        y=s[s.index.year==yr]
        row+=f" {(1+y).prod()-1:>+7.1%}" if len(y)>20 else f" {'--':>8}"
    print(row)

# Final values
print(f"\n{'='*80}")
print(f"  FINAL VALUES ($1)")
print(f"{'='*80}")
for s,label in [(ws,"Weekly 1.3/1.65"),(ms,"Monthly 1.3/1.65"),(us,"P1 1x"),(ivv_s,"IVV B&H")]:
    print(f"  {label:>18}: ${(1+s).cumprod().iloc[-1]:>8.2f}")

# Validation
print(f"\n{'='*80}")
print(f"  VALIDATION")
print(f"{'='*80}")
m_final = (1+ms).cumprod().iloc[-1]
w_final = (1+ws).cumprod().iloc[-1]
print(f"  Monthly-only: ${m_final:.2f} (expected $15.02 ±$0.20) — {'PASS' if abs(m_final-15.02)<0.20 else 'FAIL'}")
print(f"  Weekly check:  ${w_final:.2f}")
print(f"  Weekly max DD {'better' if metrics(ws,'')[4] > metrics(ms,'')[4] else 'worse'} than monthly: "
      f"{metrics(ws,'')[4]:.1%} vs {metrics(ms,'')[4]:.1%}")
print(f"  De-lever events: {len(de_df)} ({'OK' if 5<=len(de_df)<=20 else 'CHECK' if len(de_df)<5 else 'TOO MANY'})")

print()
