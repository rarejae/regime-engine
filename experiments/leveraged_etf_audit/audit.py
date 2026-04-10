"""Leveraged ETF formula audit: validate synthetic vs actual SSO/QLD.

Compares our simulation formula:
  synth_ret = 2.0 * underlying_ret - rfr/252 - expense/252

against actual SSO and QLD daily returns during the overlap period.
Then runs a hybrid backtest using actual leveraged ETF data where available.
"""

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from dotenv import load_dotenv; load_dotenv()

import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

from taa.data import load_daily_etf_returns, load_monthly_prices
from taa.faber import compute_trend_scores, apply_faber_filter

BASELINE = {"IVV": 0.45, "QQQ": 0.25, "VGLT": 0.05, "IAU": 0.10, "DBC": 0.05, "cash": 0.10}
ASSETS = list(BASELINE.keys())
SSO_EXP = 0.0089; QLD_EXP = 0.0095
DAILY_SMA_PERIODS = [126, 200, 252]


def load_data():
    import yfinance as yf

    daily_ret = load_daily_etf_returns()
    daily_ret = daily_ret[[c for c in daily_ret.columns if c in ASSETS]]
    monthly_prices = load_monthly_prices()
    trend_df = compute_trend_scores(monthly_prices)

    from fredapi import Fred
    rfr_daily = pd.Series(0.0, index=daily_ret.index)
    key = os.environ.get("FRED_API_KEY")
    if key:
        tb = Fred(api_key=key).get_series("DTB3", observation_start="1998-01-01")
        tb.index = pd.to_datetime(tb.index)
        rfr_daily = (tb / 100 / 252).reindex(daily_ret.index, method="ffill").fillna(0)

    # Daily prices for SMA
    ticker_map = {"IVV": "SPY", "QQQ": "QQQ", "VGLT": "TLT", "IAU": "GLD", "DBC": "DBC"}
    dp = {}
    for our, ticker in ticker_map.items():
        d = yf.download(ticker, start="1998-01-01", progress=False)
        if d is not None and not d.empty:
            p = d["Close"]
            if hasattr(p, "columns"): p = p.iloc[:, 0]
            p.index = pd.to_datetime(p.index).tz_localize(None)
            dp[our] = p
    dpdf = pd.DataFrame(dp).sort_index()
    daily_smas = {p: dpdf.rolling(p, min_periods=p).mean() for p in DAILY_SMA_PERIODS}

    # Actual leveraged ETF data
    actual_lev = {}
    for ticker in ["SSO", "QLD"]:
        d = yf.download(ticker, start="2006-01-01", progress=False, auto_adjust=True)
        if d is not None and not d.empty:
            p = d["Close"]
            if hasattr(p, "columns"): p = p.iloc[:, 0]
            p.index = pd.to_datetime(p.index).tz_localize(None)
            actual_lev[ticker] = p.pct_change().dropna()
            print(f"  {ticker}: {actual_lev[ticker].index.min().date()} to {actual_lev[ticker].index.max().date()} ({len(actual_lev[ticker])} days)")

    return daily_ret, trend_df, rfr_daily, dpdf, daily_smas, actual_lev


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: FORMULA VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def formula_validation(daily_ret, rfr_daily, actual_lev):
    print(f"\n{'='*100}")
    print(f"  PART 1: FORMULA VALIDATION — SYNTHETIC vs ACTUAL")
    print(f"{'='*100}")

    pairs = [
        ("SSO", "IVV", SSO_EXP, 2.0),
        ("QLD", "QQQ", QLD_EXP, 2.0),
    ]

    results = {}
    for lev_ticker, underlying, expense, mult in pairs:
        if lev_ticker not in actual_lev:
            print(f"\n  {lev_ticker}: ACTUAL DATA NOT AVAILABLE")
            continue

        actual = actual_lev[lev_ticker]
        underlying_ret = daily_ret[underlying] if underlying in daily_ret.columns else None
        if underlying_ret is None:
            print(f"\n  {underlying}: UNDERLYING DATA NOT AVAILABLE")
            continue

        # Compute synthetic
        common = actual.index.intersection(underlying_ret.dropna().index).intersection(rfr_daily.index)
        common = common.sort_values()

        a = actual.reindex(common).dropna()
        u = underlying_ret.reindex(common).dropna()
        r = rfr_daily.reindex(common).fillna(0)
        common2 = a.index.intersection(u.index).intersection(r.index)
        a = a.reindex(common2); u = u.reindex(common2); r = r.reindex(common2)

        synth = mult * u - (mult - 1) * r - expense / 252

        # Daily correlation
        daily_corr = float(a.corr(synth))

        # Monthly returns
        a_m = a.resample("MS").apply(lambda x: (1 + x).prod() - 1)
        s_m = synth.resample("MS").apply(lambda x: (1 + x).prod() - 1)
        monthly_corr = float(a_m.corr(s_m))

        # Annual returns
        a_ann = a.resample("YS").apply(lambda x: (1 + x).prod() - 1)
        s_ann = synth.resample("YS").apply(lambda x: (1 + x).prod() - 1)
        ann_corr = float(a_ann.corr(s_ann))

        # Return differences
        a_ann_ret = a.mean() * 252
        s_ann_ret = synth.mean() * 252
        ann_ret_diff = s_ann_ret - a_ann_ret

        # Tracking error
        diff = synth - a
        te_daily = diff.std() * np.sqrt(252)

        # Cumulative divergence
        cum_a = (1 + a).cumprod()
        cum_s = (1 + synth).cumprod()
        final_a = cum_a.iloc[-1]
        final_s = cum_s.iloc[-1]

        print(f"\n  {lev_ticker} vs synthetic ({mult}x * {underlying} - rfr - {expense})")
        print(f"  {'-'*70}")
        print(f"    Overlap: {common2.min().date()} to {common2.max().date()} ({len(common2)} days)")
        print(f"    Daily return correlation:   {daily_corr:.6f}")
        print(f"    Monthly return correlation: {monthly_corr:.4f}")
        print(f"    Annual return correlation:  {ann_corr:.4f}")
        print(f"    Actual annualised return:   {a_ann_ret:.2%}")
        print(f"    Synthetic annualised return:{s_ann_ret:.2%}")
        print(f"    Return diff (synth-actual): {ann_ret_diff:+.2%}")
        print(f"    Tracking error (ann):       {te_daily:.2%}")
        print(f"    Cumulative growth actual:   {final_a:.2f}x")
        print(f"    Cumulative growth synthetic:{final_s:.2f}x")
        print(f"    Cumulative divergence:      {(final_s/final_a - 1)*100:+.1f}%")

        # Pass/fail
        corr_pass = daily_corr >= 0.99
        ret_pass = abs(ann_ret_diff) <= 0.015
        print(f"\n    CORRELATION: {daily_corr:.4f}  [{'PASS' if corr_pass else 'FAIL'} — threshold 0.99]")
        print(f"    RETURN DIFF: {ann_ret_diff:+.2%}  [{'PASS' if ret_pass else 'FAIL'} — threshold 1.5%]")

        # Year-by-year comparison
        print(f"\n    Year-by-year return comparison:")
        print(f"    {'Year':>6} {'Actual':>10} {'Synthetic':>10} {'Diff':>10}")
        for yr in range(int(common2.min().year), int(common2.max().year) + 1):
            ay = a[a.index.year == yr]
            sy = synth[synth.index.year == yr]
            if len(ay) > 100:
                ar_y = (1 + ay).prod() - 1
                sr_y = (1 + sy).prod() - 1
                print(f"    {yr:>6} {ar_y:>+9.1%} {sr_y:>+9.1%} {sr_y - ar_y:>+9.2%}")

        results[lev_ticker] = {
            "daily_corr": daily_corr, "monthly_corr": monthly_corr,
            "ann_corr": ann_corr, "ann_ret_diff": ann_ret_diff,
            "te": te_daily, "corr_pass": corr_pass, "ret_pass": ret_pass,
        }

    return results


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: VOLATILITY DECAY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def volatility_decay_analysis(daily_ret, rfr_daily, actual_lev):
    print(f"\n{'='*100}")
    print(f"  PART 2: VOLATILITY DECAY ANALYSIS")
    print(f"{'='*100}")

    for lev_ticker, underlying, expense, mult in [("SSO", "IVV", SSO_EXP, 2.0), ("QLD", "QQQ", QLD_EXP, 2.0)]:
        if lev_ticker not in actual_lev or underlying not in daily_ret.columns:
            continue

        actual = actual_lev[lev_ticker]
        u = daily_ret[underlying]
        common = actual.dropna().index.intersection(u.dropna().index).sort_values()
        a = actual.reindex(common); u_c = u.reindex(common)

        # Theoretical: if leverage were free, no expense, no decay
        theo_annual = u_c.mean() * 252 * mult
        actual_annual = a.mean() * 252

        # Vol of underlying
        u_vol = u_c.std() * np.sqrt(252)

        # Theoretical vol decay: for Nx leverage, daily compounding drag ≈ -(N^2 - N) * σ^2 / 2 per year
        theo_drag = (mult**2 - mult) * u_vol**2 / 2
        # Expense drag
        exp_drag = expense
        # Borrowing cost (rfr * (mult-1))
        r = rfr_daily.reindex(common).fillna(0)
        borrow_drag = r.mean() * 252 * (mult - 1)

        total_theo_drag = theo_drag + exp_drag + borrow_drag
        actual_drag = theo_annual - actual_annual
        residual = actual_drag - total_theo_drag

        print(f"\n  {lev_ticker} ({mult}x {underlying}):")
        print(f"    Underlying annual return:   {u_c.mean()*252:.2%}")
        print(f"    Underlying annual vol:      {u_vol:.2%}")
        print(f"    Theoretical {mult}x return:    {theo_annual:.2%}")
        print(f"    Actual {lev_ticker} return:       {actual_annual:.2%}")
        print(f"    Total drag:                 {actual_drag:.2%}")
        print(f"    Decomposition:")
        print(f"      Vol decay (-(N^2-N)*s^2/2):  {theo_drag:.2%}")
        print(f"      Expense ratio:               {exp_drag:.2%}")
        print(f"      Borrowing cost (rfr*(N-1)):   {borrow_drag:.2%}")
        print(f"      Sum of known drags:          {total_theo_drag:.2%}")
        print(f"      Residual (unexplained):      {residual:+.2%}")
        print(f"    Formula captures drag?     {'YES' if abs(residual) < 0.005 else 'PARTIALLY — residual '+f'{residual:+.2%}'}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 3: HYBRID BACKTEST (actual leveraged ETF data where available)
# ══════════════════════════════════════════════════════════════════════════════

def sma_scores(day, dpdf, smas):
    scores = {}
    for a in ["IVV", "QQQ", "VGLT", "IAU", "DBC"]:
        if a not in dpdf.columns: scores[a] = 0; continue
        p = dpdf.loc[:day, a]
        if len(p) == 0 or pd.isna(p.iloc[-1]): scores[a] = 0; continue
        price = p.iloc[-1]; sc = 0
        for per in DAILY_SMA_PERIODS:
            s = smas[per].loc[:day, a]
            if len(s) > 0 and pd.notna(s.iloc[-1]) and price > s.iloc[-1]: sc += 1
        scores[a] = sc
    return scores


def check_breach(day, dpdf, smas):
    for etf in ["IVV", "QQQ"]:
        if etf not in dpdf.columns: continue
        p = dpdf.loc[:day, etf]
        if len(p) == 0: continue
        price = p.iloc[-1]; breaches = 0
        for per in DAILY_SMA_PERIODS:
            s = smas[per].loc[:day, etf]
            if len(s) > 0 and pd.notna(s.iloc[-1]) and price < s.iloc[-1]: breaches += 1
        if breaches >= 3: return True
    return False


def hybrid_backtest(daily_ret, trend_df, rfr_daily, dpdf, daily_smas, actual_lev):
    print(f"\n{'='*100}")
    print(f"  PART 3: HYBRID BACKTEST — ACTUAL LEVERAGED ETFs WHERE AVAILABLE")
    print(f"{'='*100}")

    bt_start = pd.Timestamp("2002-01-01")
    common_start = max(daily_ret.dropna(how="all").index.min(), bt_start)
    trading_days = daily_ret.loc[common_start:].index

    # Determine actual data availability
    sso_start = actual_lev["SSO"].index.min() if "SSO" in actual_lev else pd.Timestamp("2099-01-01")
    qld_start = actual_lev["QLD"].index.min() if "QLD" in actual_lev else pd.Timestamp("2099-01-01")
    both_start = max(sso_start, qld_start)
    print(f"  SSO available from: {sso_start.date()}")
    print(f"  QLD available from: {qld_start.date()}")
    print(f"  Both available from: {both_start.date()}")
    print(f"  Hybrid: synthetic pre-{both_start.date()}, actual after")

    cur_trends = {a: 3 for a in ASSETS if a != "cash"}
    w_faber = dict(BASELINE)
    faber_conv = False; leverage_active = False; delevered = False

    # Run for 40% and 100% sub, both synthetic-only and hybrid
    configs = {
        "S40-40%-synth":  (0.40, False),
        "S40-40%-hybrid": (0.40, True),
        "S40-100%-synth":  (1.00, False),
        "S40-100%-hybrid": (1.00, True),
    }
    results = {k: {} for k in configs}
    results["IVV B&H"] = {}

    for day in trading_days:
        if day not in daily_ret.index: continue
        dr = daily_ret.loc[day]
        avail = [a for a in ASSETS if a in dr.index and pd.notna(dr[a])]
        if len(avail) < 3: continue
        actual = {a: float(dr[a]) for a in avail}
        rfr = float(rfr_daily.get(day, 0))

        is_ms = (day == trading_days[0] or
                 day.month != trading_days[trading_days.get_loc(day) - 1].month)

        if is_ms:
            delevered = False
            prior = trading_days[trading_days < day]
            sd = prior[-1] if len(prior) > 0 else day
            cur_trends = sma_scores(sd, dpdf, daily_smas)
            w1, pool = apply_faber_filter(cur_trends, BASELINE)
            w_faber = dict(w1)
            w_faber["cash"] = w_faber.get("cash", 0) + pool
            faber_conv = (cur_trends.get("IVV", 0) >= 3 and cur_trends.get("QQQ", 0) >= 3)
            leverage_active = faber_conv

        if leverage_active and not delevered:
            if check_breach(day, dpdf, daily_smas):
                leverage_active = False; delevered = True

        iw = w_faber.get("IVV", 0)
        qw = w_faber.get("QQQ", 0)
        ir = actual.get("IVV", 0)
        qr = actual.get("QQQ", 0)
        base_ret = sum(w_faber.get(a, 0) * actual.get(a, 0) for a in avail if a not in ["IVV", "QQQ"])

        for cfg_name, (sub_pct, use_actual) in configs.items():
            active_sub = sub_pct if leverage_active else 0.0

            if active_sub <= 0:
                results[cfg_name][day] = iw * ir + qw * qr + base_ret
            elif use_actual and day >= both_start and "SSO" in actual_lev and "QLD" in actual_lev:
                # Use actual SSO/QLD returns
                sso_r = float(actual_lev["SSO"].get(day, np.nan))
                qld_r = float(actual_lev["QLD"].get(day, np.nan))
                if np.isnan(sso_r): sso_r = 2.0 * ir - rfr - SSO_EXP / 252
                if np.isnan(qld_r): qld_r = 2.0 * qr - rfr - QLD_EXP / 252
                results[cfg_name][day] = iw * ((1 - active_sub) * ir + active_sub * sso_r) + \
                                          qw * ((1 - active_sub) * qr + active_sub * qld_r) + base_ret
            else:
                # Synthetic
                sso_r = 2.0 * ir - rfr - SSO_EXP / 252
                qld_r = 2.0 * qr - rfr - QLD_EXP / 252
                results[cfg_name][day] = iw * ((1 - active_sub) * ir + active_sub * sso_r) + \
                                          qw * ((1 - active_sub) * qr + active_sub * qld_r) + base_ret

        results["IVV B&H"][day] = actual.get("IVV", 0)

    ret_series = {k: pd.Series(v).sort_index() for k, v in results.items()}

    # Compute metrics
    def m(s):
        s = s.dropna()
        ar = s.mean() * 252; av = s.std() * np.sqrt(252)
        sh = ar / av if av > 0 else 0
        cum = (1 + s).cumprod()
        dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
        cal = ar / abs(dd) if dd != 0 else 0
        return {"ar": ar, "av": av, "sh": sh, "dd": dd, "cal": cal, "final": cum.iloc[-1]}

    perfs = {k: m(ret_series[k]) for k in ret_series}

    print(f"\n  {'Strategy':<22} {'Return':>8} {'Vol':>7} {'Sharpe':>8} {'MaxDD':>8} {'Terminal':>10}")
    print(f"  {'-'*22} {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*10}")
    for k in ["S40-40%-synth", "S40-40%-hybrid", "S40-100%-synth", "S40-100%-hybrid", "IVV B&H"]:
        p = perfs[k]
        print(f"  {k:<22} {p['ar']:>7.1%} {p['av']:>6.1%} {p['sh']:>8.3f} {p['dd']:>7.1%} ${p['final']:>9.2f}")

    # Differences
    print(f"\n  SYNTHETIC vs HYBRID DIFFERENCES:")
    for sub_label in ["40%", "100%"]:
        ps = perfs[f"S40-{sub_label}-synth"]
        ph = perfs[f"S40-{sub_label}-hybrid"]
        ret_diff = ph["ar"] - ps["ar"]
        term_diff = ph["final"] - ps["final"]
        print(f"\n  S40-{sub_label}:")
        print(f"    Return diff (hybrid - synth): {ret_diff:+.2%} annualised")
        print(f"    Terminal diff:                ${term_diff:+.2f}")
        print(f"    Sharpe diff:                  {ph['sh'] - ps['sh']:+.3f}")
        print(f"    MaxDD diff:                   {(ph['dd'] - ps['dd'])*100:+.1f}%")
        if abs(ret_diff) > 0.005:
            print(f"    → Prior results {'OVERSTATED' if ret_diff < 0 else 'UNDERSTATED'} by ~{abs(ret_diff):.2%} annualised")
        else:
            print(f"    → Prior results ACCURATE (diff < 0.5%)")

    # Age-65 projections
    print(f"\n  REVISED AGE-65 PROJECTIONS ($21,000, 40 years):")
    for sub_label in ["40%", "100%"]:
        ph = perfs[f"S40-{sub_label}-hybrid"]
        ps = perfs[f"S40-{sub_label}-synth"]
        proj_h = 21000 * (1 + ph["ar"]) ** 40
        proj_s = 21000 * (1 + ps["ar"]) ** 40
        print(f"    S40-{sub_label}-synth:  ${proj_s:>12,.0f}  ({ps['ar']:.1%} ann)")
        print(f"    S40-{sub_label}-hybrid: ${proj_h:>12,.0f}  ({ph['ar']:.1%} ann)")

    return perfs


# ══════════════════════════════════════════════════════════════════════════════
# FINAL VERDICT
# ══════════════════════════════════════════════════════════════════════════════

def final_verdict(formula_results, hybrid_perfs):
    print(f"\n{'='*100}")
    print(f"  === FULL AUDIT VERDICT ===")
    print(f"{'='*100}")

    # Formula validation
    print(f"\n  Formula validation:")
    for ticker in ["SSO", "QLD"]:
        r = formula_results.get(ticker, {})
        if not r:
            print(f"    {ticker}: NO DATA")
            continue
        print(f"    {ticker} correlation:          {r['daily_corr']:.4f}  [{'PASS' if r['corr_pass'] else 'FAIL'} — threshold 0.99]")
        print(f"    {ticker} annual return diff:   {r['ann_ret_diff']:+.2%}  [{'PASS' if r['ret_pass'] else 'FAIL'} — threshold 1.5%]")

    # Return impact
    print(f"\n  Return impact of using real data:")
    for sub in ["40%", "100%"]:
        ps = hybrid_perfs.get(f"S40-{sub}-synth", {})
        ph = hybrid_perfs.get(f"S40-{sub}-hybrid", {})
        if ps and ph:
            rd = ph["ar"] - ps["ar"]
            td = ph["final"] - ps["final"]
            print(f"    S40-{sub}:   {rd:+.2%} return difference  (${td:+.2f} terminal)")

    # Overstated?
    ps100 = hybrid_perfs.get("S40-100%-synth", {})
    ph100 = hybrid_perfs.get("S40-100%-hybrid", {})
    if ps100 and ph100:
        rd = ph100["ar"] - ps100["ar"]
        overstated = rd < -0.005
        print(f"\n  Are prior results overstated?  {'YES' if overstated else 'NO'}")
        if overstated:
            print(f"  If yes, by approximately:      {abs(rd):.2%} annualised")

    # Revised projections
    print(f"\n  Revised age-65 projections from $21,000:")
    for sub in ["40%", "100%"]:
        ph = hybrid_perfs.get(f"S40-{sub}-hybrid", {})
        if ph:
            proj = 21000 * (1 + ph["ar"]) ** 40
            print(f"    S40-{sub}-hybrid:   ${proj:>12,.0f}")

    # Overall
    all_corr_pass = all(formula_results.get(t, {}).get("corr_pass", False) for t in ["SSO", "QLD"])
    all_ret_pass = all(formula_results.get(t, {}).get("ret_pass", False) for t in ["SSO", "QLD"])

    ps40 = hybrid_perfs.get("S40-40%-synth", {})
    ph40 = hybrid_perfs.get("S40-40%-hybrid", {})
    ret_diff_40 = abs(ph40.get("ar", 0) - ps40.get("ar", 0)) if ps40 and ph40 else 0
    ret_diff_100 = abs(ph100.get("ar", 0) - ps100.get("ar", 0)) if ps100 and ph100 else 0
    reliable = all_corr_pass and max(ret_diff_40, ret_diff_100) < 0.01

    # Is 100% still correct?
    ph100_sh = ph100.get("sh", 0) if ph100 else 0
    ph40_sh = ph40.get("sh", 0) if ph40 else 0
    ph100_final = ph100.get("final", 0) if ph100 else 0
    ph40_final = ph40.get("final", 0) if ph40 else 0
    still_100 = ph100_final > ph40_final and ph100_sh > 0.85

    print(f"\n  Overall verdict:")
    print(f"    Are the prior backtest results reliable?         {'YES' if reliable else 'NO — investigate further'}")
    print(f"    Is 100% substitution still the correct choice?  {'YES' if still_100 else 'REASSESS'}")

    # Key finding
    if reliable and still_100:
        key = "Synthetic leveraged ETF formula is accurate; 100% substitution validated with real data."
    elif reliable and not still_100:
        key = "Formula accurate but 100% sub may not be optimal with real data."
    else:
        key = f"Formula has return gap of {max(ret_diff_40, ret_diff_100):.2%}; results directionally correct but magnitude uncertain."

    print(f"    Key finding in one sentence: {key}")


if __name__ == "__main__":
    print("=" * 100)
    print("  LEVERAGED ETF FORMULA AUDIT")
    print("=" * 100)

    print(f"\n  Loading data...")
    daily_ret, trend_df, rfr_daily, dpdf, daily_smas, actual_lev = load_data()

    formula_results = formula_validation(daily_ret, rfr_daily, actual_lev)
    volatility_decay_analysis(daily_ret, rfr_daily, actual_lev)
    hybrid_perfs = hybrid_backtest(daily_ret, trend_df, rfr_daily, dpdf, daily_smas, actual_lev)
    final_verdict(formula_results, hybrid_perfs)
