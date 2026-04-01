"""Weekly leverage exit with 2/3 and 3/3 SMA breach thresholds."""

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
TIER_SUBS = {0: 0.0, 1: 0.30, 2: 0.65}
SEED = {"IVV": 0.0028, "QQQ": 0.0080}

print("=" * 80)
print("  WEEKLY DE-LEVER: 2/3 vs 3/3 SMA BREACH THRESHOLDS")
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

import yfinance as yf
daily_prices = {}
for our, ticker in [("IVV","SPY"),("QQQ","QQQ")]:
    d = yf.download(ticker, start="1998-01-01", progress=False)
    if d is not None and not d.empty:
        p = d["Close"]
        if hasattr(p,"columns"): p=p.iloc[:,0]
        p.index = pd.to_datetime(p.index).tz_localize(None)
        daily_prices[our] = p
dpdf = pd.DataFrame(daily_prices)

# Monthly SMAs
mc = dpdf.resample("MS").last()
sma_6 = mc.rolling(6, min_periods=6).mean()
sma_10 = mc.rolling(10, min_periods=10).mean()
sma_12 = mc.rolling(12, min_periods=12).mean()

# Pre-backtest seeds
from taa.harvey import find_similar_months as fsm
pre_ers = {"IVV": [], "QQQ": []}
bt_start = pd.Timestamp("2002-01-01")
for z_dt in z_clean.index[z_clean.index < bt_start]:
    try:
        sim, _ = fsm(z_clean, z_dt)
        er = compute_expected_returns(sim, asset_ret_fwd, ["IVV","QQQ"])
        for a in ["IVV","QQQ"]:
            if not np.isnan(er.get(a, np.nan)): pre_ers[a].append(er[a])
    except: pass

common_start = max(daily_ret.dropna(how="all").index.min(), bt_start)
trading_days = daily_ret.loc[common_start:].index
print(f"Backtest: {len(trading_days)} days ({common_start.date()} to {trading_days.max().date()})")

# Configs: monthly, weekly-2of3, weekly-3of3
CONFIGS = ["monthly", "w2of3", "w3of3"]

# State per config
state = {}
for cfg in CONFIGS:
    state[cfg] = {
        "tier": 0, "delevered": False,
        "hist": {"IVV": list(pre_ers["IVV"]), "QQQ": list(pre_ers["QQQ"])},
        "med": dict(SEED),
        "events": [],
    }

ct = {a: 3 for a in ASSETS if a != "cash"}
he = {a: 0.0 for a in ASSETS}
wp = dict(BASELINE)

strats = {cfg: {} for cfg in CONFIGS}
strats["p1_1x"] = {}
strats["ivv"] = {}
total_months = 0


def count_sma_breaches(etf, day):
    """Count how many of 3 SMAs the Friday price is below."""
    if etf not in dpdf.columns: return 0
    prices_up_to = dpdf.loc[:day, etf]
    if len(prices_up_to) == 0: return 0
    price = prices_up_to.iloc[-1]
    ms = pd.Timestamp(f"{day.year}-{day.month:02d}-01")
    breaches = 0
    for sma_df in [sma_6, sma_10, sma_12]:
        sma_dates = sma_df.index[sma_df.index <= ms]
        if len(sma_dates) == 0: continue
        val = sma_df.loc[sma_dates[-1], etf]
        if pd.notna(val) and price < val:
            breaches += 1
    return breaches


for day in trading_days:
    if day not in daily_ret.index: continue
    dr = daily_ret.loc[day]
    avail = [a for a in ASSETS if a in dr.index and pd.notna(dr[a])]
    if len(avail) < 3: continue
    actual = {a: float(dr[a]) for a in avail}
    rfr = float(rfr_daily.get(day, 0))

    is_ms = (day == trading_days[0] or day.month != trading_days[trading_days.get_loc(day)-1].month)
    is_friday = day.weekday() == 4

    if is_ms:
        total_months += 1

        # Signals
        ts_c = trend_df.index[trend_df.index <= day]
        if len(ts_c) > 0:
            for a in ASSETS:
                if a == "cash": continue
                if a in trend_df.columns:
                    v = trend_df.loc[ts_c[-1], a]; ct[a] = int(v) if pd.notna(v) else 0
        z_c = z_clean.index[z_clean.index < day]
        if len(z_c) > 0:
            try:
                sim, _ = find_similar_months(z_clean, z_c[-1])
                he = compute_expected_returns(sim, asset_ret_fwd, avail)
            except: pass
        rv_c = rvol_monthly.index[rvol_monthly.index <= day]; rvols = {}
        if len(rv_c) > 0:
            for a in ASSETS:
                if a == "cash": continue
                if a in rvol_monthly.columns:
                    v = rvol_monthly.loc[rv_c[-1], a]
                    rvols[a] = float(v) if pd.notna(v) and v > 0 else 0.15

        w1, pool = apply_faber_filter(ct, BASELINE)
        w2, _ = direct_capital(dict(w1), pool, he, rvols)
        wp = normalize(w2)

        f_c = ct.get("IVV",0)>=3 and ct.get("QQQ",0)>=3
        h_c = he.get("IVV",0)>0 and he.get("QQQ",0)>0
        conv = f_c and h_c

        for cfg in CONFIGS:
            s = state[cfg]
            s["delevered"] = False
            if conv:
                above = all(he.get(a,0) > s["med"].get(a,0) for a in ["IVV","QQQ"])
                s["tier"] = 2 if above else 1
                for a in ["IVV","QQQ"]:
                    s["hist"][a].append(he.get(a,0))
                    s["med"][a] = float(np.median(s["hist"][a]))
            else:
                s["tier"] = 0

    # Weekly checks
    if is_friday:
        ivv_breaches = count_sma_breaches("IVV", day)
        qqq_breaches = count_sma_breaches("QQQ", day)

        for cfg, threshold in [("w2of3", 2), ("w3of3", 3)]:
            s = state[cfg]
            if s["tier"] > 0 and not s["delevered"]:
                if ivv_breaches >= threshold or qqq_breaches >= threshold:
                    which = f"IVV({ivv_breaches}/3)" if ivv_breaches >= threshold else f"QQQ({qqq_breaches}/3)"
                    s["events"].append({"date": day, "tier": s["tier"], "detail": which})
                    s["tier"] = 0
                    s["delevered"] = True

    # Daily returns
    iw = wp.get("IVV",0); qw = wp.get("QQQ",0)
    ir = actual.get("IVV",0); qr = actual.get("QQQ",0)
    sr = 2*ir - rfr - 0.0091/252; ql = 2*qr - rfr - 0.0089/252
    base = sum(wp.get(a,0)*actual.get(a,0) for a in avail if a not in ["IVV","QQQ"])

    strats["p1_1x"][day] = iw*ir + qw*qr + base
    strats["ivv"][day] = actual.get("IVV", 0)

    for cfg in CONFIGS:
        sub = TIER_SUBS.get(state[cfg]["tier"], 0)
        if sub > 0:
            strats[cfg][day] = iw*(1-sub)*ir + iw*sub*sr + qw*(1-sub)*qr + qw*sub*ql + base
        else:
            strats[cfg][day] = iw*ir + qw*qr + base

results = {c: pd.Series(d).sort_index() for c, d in strats.items()}

def m(s):
    ar=s.mean()*252; av=s.std()*np.sqrt(252); sh=ar/av if av>0 else 0
    neg=s[s<0]; ds=neg.std()*np.sqrt(252) if len(neg)>10 else av; so=ar/ds if ds>0 else 0
    cum=(1+s).cumprod(); dd=((cum-cum.expanding().max())/cum.expanding().max()).min()
    cal=ar/abs(dd) if dd!=0 else 0; final=cum.iloc[-1]
    return ar,av,sh,so,dd,cal,final

labels = {"monthly":"Monthly","w2of3":"Weekly 2/3","w3of3":"Weekly 3/3","p1_1x":"P1 1x","ivv":"IVV B&H"}
order = ["w2of3","w3of3","monthly","p1_1x","ivv"]

print(f"\n{'='*80}")
print(f"  PERFORMANCE")
print(f"{'='*80}")
print(f"\n  {'Config':>14} {'AnnRet':>7} {'AnnVol':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>8} {'Calmar':>7} {'Final':>8}")
print(f"  {'-'*68}")
for cn in order:
    s = results[cn]; ar,av,sh,so,dd,cal,final = m(s)
    print(f"  {labels[cn]:>14} {ar:>6.1%} {av:>6.1%} {sh:>7.2f} {so:>8.2f} {dd:>7.1%} {cal:>7.2f} ${final:>7.2f}")

# De-lever events
print(f"\n{'='*80}")
print(f"  DE-LEVER EVENTS")
print(f"{'='*80}")
for cfg in ["w2of3","w3of3"]:
    events = state[cfg]["events"]
    print(f"\n  {labels[cfg]}: {len(events)} events ({len(events)/24:.1f}/year)")
    if events:
        print(f"  {'Date':>10} {'Tier':>5} {'Breach'}")
        for e in events:
            print(f"  {e['date'].strftime('%Y-%m-%d'):>10} {e['tier']:>5} {e['detail']}")

# Crisis
print(f"\n{'='*80}")
print(f"  CRISIS DRAWDOWNS")
print(f"{'='*80}")
for cn2,cs,ce in [("GFC","2008-09-01","2009-03-31"),("COVID","2020-02-19","2020-03-23"),("2022","2022-01-03","2022-10-31")]:
    print(f"\n  {cn2}:")
    for cn in order:
        s=results[cn]
        c=s[(s.index>=pd.Timestamp(cs))&(s.index<=pd.Timestamp(ce))]
        if len(c)>0:
            cum=(1+c).cumprod();mdd=((cum-cum.expanding().max())/cum.expanding().max()).min()
            print(f"    {labels[cn]:>14}: {((1+c).prod()-1):>+7.1%} (DD {mdd:.1%})")

    # Events during crisis
    for cfg in ["w2of3","w3of3"]:
        crisis_ev = [e for e in state[cfg]["events"] if pd.Timestamp(cs) <= e["date"] <= pd.Timestamp(ce)]
        if crisis_ev:
            print(f"    {labels[cfg]:>14} trigger: {crisis_ev[0]['date'].strftime('%Y-%m-%d')}")

# COVID specific
print(f"\n{'='*80}")
print(f"  COVID DETAIL (Feb 19 - Mar 23, 2020)")
print(f"{'='*80}")
for cfg in ["w2of3","w3of3","monthly"]:
    s = results[cfg]
    c = s[(s.index>="2020-02-19")&(s.index<="2020-03-23")]
    if len(c) > 0:
        cum=(1+c).cumprod(); mdd=((cum-cum.expanding().max())/cum.expanding().max()).min()
        print(f"  {labels[cfg]:>14}: total={((1+c).prod()-1):>+.1%}, maxDD={mdd:.1%}")
    events = [e for e in state[cfg]["events"] if pd.Timestamp("2020-02-01") <= e["date"] <= pd.Timestamp("2020-04-01")] if cfg != "monthly" else []
    if events:
        print(f"    Trigger: {events[0]['date'].strftime('%Y-%m-%d')} ({events[0]['detail']})")
    elif cfg != "monthly":
        print(f"    No trigger during COVID")

# Calendar years
print(f"\n{'='*80}")
print(f"  CALENDAR YEAR RETURNS")
print(f"{'='*80}")
print(f"  {'Year':>6} {'2of3':>8} {'3of3':>8} {'Monthly':>8} {'P1 1x':>8} {'IVV':>8}")
for yr in range(2002,2025):
    row=f"  {yr:>6}"
    for cn in order:
        s=results[cn];y=s[s.index.year==yr]
        row+=f" {(1+y).prod()-1:>+7.1%}" if len(y)>20 else f" {'--':>8}"
    print(row)

# Final
print(f"\n{'='*80}")
print(f"  FINAL VALUES ($1)")
print(f"{'='*80}")
for cn in order:
    s=results[cn]; print(f"  {labels[cn]:>14}: ${(1+s).cumprod().iloc[-1]:>8.2f}")

# Validation
print(f"\n{'='*80}")
print(f"  VALIDATION")
print(f"{'='*80}")
mf = (1+results["monthly"]).cumprod().iloc[-1]
print(f"  Monthly: ${mf:.2f} (expected ~$15.02 ±$0.30) — {'PASS' if abs(mf-15.02)<0.30 else 'CHECK'}")
n2 = len(state["w2of3"]["events"]); n3 = len(state["w3of3"]["events"])
print(f"  2of3 events: {n2} ({'OK' if 5<=n2<=15 else 'CHECK'})")
print(f"  3of3 events: {n3} ({'OK' if n3<10 else 'CHECK'})")
print(f"  3of3 < 2of3: {'PASS' if n3 <= n2 else 'FAIL'}")

_,_,_,_,dd2,_,_ = m(results["w2of3"])
_,_,_,_,dd3,_,_ = m(results["w3of3"])
_,_,_,_,ddm,_,_ = m(results["monthly"])
print(f"  Weekly 2/3 MaxDD ({dd2:.1%}) <= Monthly ({ddm:.1%}): {'PASS' if dd2 >= ddm else 'FAIL'}")
print(f"  Weekly 3/3 MaxDD ({dd3:.1%}) <= Monthly ({ddm:.1%}): {'PASS' if dd3 >= ddm else 'FAIL'}")

print()
