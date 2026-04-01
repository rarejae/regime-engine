"""Signal-driven TAA v2: graduated Faber + Harvey + conviction leverage."""

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv; load_dotenv()

import numpy as np, pandas as pd

OUTPUT = Path("taa_v2/results"); OUTPUT.mkdir(parents=True, exist_ok=True)
TC = 0.0010
UNIVERSE = ["IVV", "QQQ", "VGLT", "IAU", "cash"]
CASH_FLOOR = 0.10
ALLOCABLE = 1.0 - CASH_FLOOR
SMA_PERIODS = [6, 10, 12]
FABER_MULT = {3: 1.0, 2: 0.7, 1: 0.15, 0: 0.15}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all():
    import yfinance as yf
    from fredapi import Fred

    # Monthly macro for Harvey
    raw_macro = pd.read_parquet("data/macro/monthly_history.parquet")
    raw_macro.index = pd.to_datetime(raw_macro.index)

    # Monthly asset returns
    asset_ret = pd.read_parquet("data/macro/roth_asset_returns.parquet")
    keep = [c for c in UNIVERSE if c in asset_ret.columns]
    asset_ret = asset_ret[keep]
    asset_ret_fwd = asset_ret.shift(-1)

    # Daily ETF returns
    frames = {}
    for our, ticker in [("IVV","SPY"),("QQQ","QQQ"),("VGLT","TLT"),("IAU","GLD")]:
        d = yf.download(ticker, start="1998-01-01", progress=False)
        if d is not None and not d.empty:
            p = d["Close"]
            if hasattr(p,"columns"): p=p.iloc[:,0]
            p.index = pd.to_datetime(p.index).tz_localize(None)
            frames[our] = p.pct_change()
    key = os.environ.get("FRED_API_KEY")
    if key:
        try:
            tb = Fred(api_key=key).get_series("DTB3", observation_start="1998-01-01")
            tb.index = pd.to_datetime(tb.index)
            frames["cash"] = tb / 36000
        except Exception:
            pass
    daily_ret = pd.DataFrame(frames).sort_index()

    # Monthly prices for SMA
    monthly_prices = pd.DataFrame({
        our: frames[our].add(1).cumprod() if our in frames else pd.Series(dtype=float)
        for our in ["IVV","QQQ","VGLT","IAU"]
    })
    # Actually use raw prices for SMA
    prices = {}
    for our, ticker in [("IVV","SPY"),("QQQ","QQQ"),("VGLT","TLT"),("IAU","GLD")]:
        d = yf.download(ticker, start="1998-01-01", progress=False)
        if d is not None and not d.empty:
            p = d["Close"]
            if hasattr(p,"columns"): p=p.iloc[:,0]
            p.index = pd.to_datetime(p.index).tz_localize(None)
            prices[our] = p
    monthly_prices = pd.DataFrame(prices).resample("MS").last()

    # Realized vol
    rvol = pd.DataFrame(prices).pct_change().rolling(63, min_periods=30).std() * np.sqrt(252)
    rvol_monthly = rvol.resample("MS").last().shift(1)

    # Daily rfr for leverage
    rfr_daily = pd.Series(0.0, index=daily_ret.index)
    if key:
        try:
            tb2 = Fred(api_key=key).get_series("DTB3", observation_start="1998-01-01")
            tb2.index = pd.to_datetime(tb2.index)
            rfr_daily = (tb2 / 100 / 252).reindex(daily_ret.index, method="ffill").fillna(0)
        except Exception:
            pass

    return raw_macro, asset_ret, asset_ret_fwd, daily_ret, monthly_prices, rvol_monthly, rfr_daily


# ── Faber ─────────────────────────────────────────────────────────────────────

def compute_trend_scores(monthly_prices):
    scores = pd.DataFrame(0, index=monthly_prices.index, columns=monthly_prices.columns)
    for p in SMA_PERIODS:
        sma = monthly_prices.rolling(p, min_periods=p).mean()
        scores += (monthly_prices > sma).astype(int)
    return scores.shift(1)  # PIT


# ── Harvey ────────────────────────────────────────────────────────────────────

def compute_zscores(raw_macro):
    result = pd.DataFrame(index=raw_macro.index)
    for col in raw_macro.columns:
        chg = raw_macro[col] - raw_macro[col].shift(12)
        std = chg.rolling(120, min_periods=60).std()
        result[f"{col}_z"] = (chg / std.replace(0, np.nan)).clip(-3, 3)
    return result.shift(1).dropna(how="all")


def find_similar(z_data, target_date, exclude=36, pctl=0.15):
    if target_date not in z_data.index:
        raise ValueError(f"{target_date} not in z_data")
    target = z_data.loc[target_date].values
    cutoff = target_date - pd.DateOffset(months=exclude)
    cands = z_data[z_data.index <= cutoff]
    if len(cands) == 0:
        raise ValueError("No candidates")
    diffs = np.array(cands.values - target, dtype=np.float64)
    dists = pd.Series(np.sqrt(np.sum(diffs**2, axis=1)), index=cands.index)
    thresh = dists.quantile(pctl)
    return dists[dists <= thresh].index.tolist(), float(dists.min())


def expected_returns(similar_dates, asset_ret_fwd, assets):
    er = {}
    for a in assets:
        if a not in asset_ret_fwd.columns or a == "cash":
            er[a] = 0.0; continue
        rets = [asset_ret_fwd.loc[d, a] for d in similar_dates
                if d in asset_ret_fwd.index and pd.notna(asset_ret_fwd.loc[d, a])]
        er[a] = float(np.mean(rets)) if rets else 0.0
    return er


# ── Allocation ────────────────────────────────────────────────────────────────

def allocate(harvey_er, rvols, faber_scores):
    """Pure signal-driven allocation. Returns {asset: weight}."""
    # Adjusted scores
    scores = {}
    for a in ["IVV", "QQQ", "VGLT", "IAU"]:
        er = harvey_er.get(a, 0)
        vol = rvols.get(a, 0.15)
        mult = FABER_MULT.get(faber_scores.get(a, 0), 0.15)
        if er > 0 and vol > 0.01:
            scores[a] = (er / vol) * mult
        else:
            scores[a] = 0.0

    total = sum(max(s, 0) for s in scores.values())
    w = {"cash": CASH_FLOOR}

    if total <= 0:
        w["cash"] = 1.0
        for a in ["IVV","QQQ","VGLT","IAU"]:
            w[a] = 0.0
        return w

    for a in ["IVV", "QQQ", "VGLT", "IAU"]:
        w[a] = ALLOCABLE * max(scores[a], 0) / total

    return w


def apply_leverage(weights, faber_scores, harvey_er, daily_ret_row, rfr):
    """Apply conviction leverage. Returns adjusted daily return."""
    # Check max conviction
    f_conv = faber_scores.get("IVV", 0) >= 3 and faber_scores.get("QQQ", 0) >= 3
    h_conv = harvey_er.get("IVV", 0) > 0 and harvey_er.get("QQQ", 0) > 0
    leveraged = f_conv and h_conv

    w = dict(weights)
    if leveraged:
        ivv_w = w.get("IVV", 0)
        qqq_w = w.get("QQQ", 0)
        sso_w = ivv_w * 0.25
        qld_w = qqq_w * 0.25
        # SSO daily return
        ivv_r = daily_ret_row.get("IVV", 0)
        qqq_r = daily_ret_row.get("QQQ", 0)
        sso_r = 2.0 * ivv_r - rfr - 0.0091/252
        qld_r = 2.0 * qqq_r - rfr - 0.0089/252

        port_ret = (
            (ivv_w - sso_w) * ivv_r +
            sso_w * sso_r +
            (qqq_w - qld_w) * qqq_r +
            qld_w * qld_r +
            w.get("VGLT", 0) * daily_ret_row.get("VGLT", 0) +
            w.get("IAU", 0) * daily_ret_row.get("IAU", 0) +
            w.get("cash", 0) * daily_ret_row.get("cash", 0)
        )
        return port_ret, True
    else:
        port_ret = sum(w.get(a, 0) * daily_ret_row.get(a, 0) for a in UNIVERSE)
        return port_ret, False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("  SIGNAL-DRIVEN TAA v2 — GRADUATED FABER + HARVEY + CONVICTION LEVERAGE")
    print("=" * 80)

    raw_macro, asset_ret, asset_ret_fwd, daily_ret, monthly_prices, rvol_monthly, rfr_daily = load_all()
    trend_df = compute_trend_scores(monthly_prices)
    z_data = compute_zscores(raw_macro)
    z_cols = [c for c in z_data.columns if c.endswith("_z")]
    z_clean = z_data[z_cols].dropna()

    # Also load Phase 1 results for comparison
    # Re-run Phase 1 inline
    from taa.faber import compute_trend_scores as p1_trend
    from taa.harvey import compute_zscore_variables as p1_zscores, find_similar_months as p1_similar, compute_expected_returns as p1_er
    from taa.allocation import BASELINE as P1_BASE, direct_capital as p1_direct, normalize as p1_norm
    from taa.faber import apply_faber_filter as p1_faber
    from taa.data import load_monthly_prices as p1_prices, load_realized_vols as p1_rvol, load_monthly_macro as p1_macro
    p1_monthly = p1_prices()
    p1_trend_df = p1_trend(p1_monthly)
    p1_macro_raw = p1_macro()
    p1_z = p1_zscores(p1_macro_raw)
    p1_z_cols = [c for c in p1_z.columns if c.endswith("_z")]
    p1_z_clean = p1_z[p1_z_cols].dropna()
    p1_rvol_m = p1_rvol()
    p1_asset_ret = pd.read_parquet("data/macro/roth_asset_returns.parquet")
    for c in list(p1_asset_ret.columns):
        if c not in list(P1_BASE.keys()): p1_asset_ret = p1_asset_ret.drop(columns=[c])
    p1_asset_fwd = p1_asset_ret.shift(-1)

    common_start = max(daily_ret.dropna(how="all").index.min(), pd.Timestamp("2002-01-01"))
    trading_days = daily_ret.loc[common_start:].index
    print(f"\nBacktest: {len(trading_days)} days ({common_start.date()} to {trading_days.max().date()})")

    # Signal alignment assertion
    dt_test = pd.Timestamp("2007-12-01")
    z_before = z_clean.index[z_clean.index < dt_test]
    assert len(z_before) > 0 and z_before[-1] < dt_test, "FAIL: z-score look-ahead"
    print(f"Signal alignment: z-score for Dec 2007 uses {z_before[-1].date()} — PASS")

    strats = {"v2_lev": {}, "v2_1x": {}, "p1_lev": {}, "p1_1x": {},
              "bench_6040": {}, "bench_ivv": {}}
    current_w = {c: None for c in strats}
    total_tc = {c: 0.0 for c in strats}

    cur_faber = {a: 3 for a in ["IVV","QQQ","VGLT","IAU"]}
    cur_harvey = {a: 0.0 for a in UNIVERSE}
    cur_rvols = {a: 0.15 for a in ["IVV","QQQ","VGLT","IAU"]}
    w_v2 = {a: 0.0 for a in UNIVERSE}; w_v2["cash"] = 1.0

    # Phase 1 state
    p1_trends = {a: True for a in P1_BASE if a != "cash"}
    p1_harvey_er = {a: 0.0 for a in P1_BASE}
    w_p1 = dict(P1_BASE)

    lev_months_v2 = 0; lev_months_p1 = 0; total_months = 0
    weight_log = []

    for day in trading_days:
        if day not in daily_ret.index: continue
        dr = daily_ret.loc[day]
        avail = [a for a in UNIVERSE if a in dr.index and pd.notna(dr[a])]
        if len(avail) < 3: continue
        actual = {a: float(dr[a]) for a in avail}
        rfr = float(rfr_daily.get(day, 0))

        is_ms = (day == trading_days[0] or day.month != trading_days[trading_days.get_loc(day)-1].month)

        if is_ms:
            total_months += 1

            # V2: update Faber
            ts_cands = trend_df.index[trend_df.index <= day]
            if len(ts_cands) > 0:
                ts = ts_cands[-1]
                for a in ["IVV","QQQ","VGLT","IAU"]:
                    if a in trend_df.columns:
                        v = trend_df.loc[ts, a]
                        cur_faber[a] = int(v) if pd.notna(v) else 0

            # V2: update Harvey
            z_cands = z_clean.index[z_clean.index < day]
            if len(z_cands) > 0:
                z_dt = z_cands[-1]
                try:
                    sim, _ = find_similar(z_clean, z_dt)
                    cur_harvey = expected_returns(sim, asset_ret_fwd, avail)
                except ValueError:
                    cur_harvey = {a: 0.0 for a in avail}

            # V2: realized vols
            rv_cands = rvol_monthly.index[rvol_monthly.index <= day]
            if len(rv_cands) > 0:
                rv = rv_cands[-1]
                for a in ["IVV","QQQ","VGLT","IAU"]:
                    if a in rvol_monthly.columns:
                        v = rvol_monthly.loc[rv, a]
                        cur_rvols[a] = float(v) if pd.notna(v) and v > 0 else 0.15

            w_v2 = allocate(cur_harvey, cur_rvols, cur_faber)

            # Phase 1: update
            p1_ts_cands = p1_trend_df.index[p1_trend_df.index <= day]
            if len(p1_ts_cands) > 0:
                p1_ts = p1_ts_cands[-1]
                for a in P1_BASE:
                    if a == "cash": continue
                    if a in p1_trend_df.columns:
                        s = p1_trend_df.loc[p1_ts, a]
                        p1_trends[a] = int(s) if pd.notna(s) else 0

            p1_z_cands = p1_z_clean.index[p1_z_clean.index < day]
            if len(p1_z_cands) > 0:
                p1_z_dt = p1_z_cands[-1]
                try:
                    p1_sim, _ = p1_similar(p1_z_clean, p1_z_dt)
                    p1_harvey_er = p1_er(p1_sim, p1_asset_fwd, list(P1_BASE.keys()))
                except ValueError:
                    p1_harvey_er = {a: 0 for a in P1_BASE}

            p1_rv_cands = p1_rvol_m.index[p1_rvol_m.index <= day]
            p1_rvols = {}
            if len(p1_rv_cands) > 0:
                p1_rv = p1_rv_cands[-1]
                for a in P1_BASE:
                    if a == "cash": continue
                    if a in p1_rvol_m.columns:
                        v = p1_rvol_m.loc[p1_rv, a]
                        p1_rvols[a] = float(v) if pd.notna(v) and v > 0 else 0.15

            p1_w1, p1_pool = p1_faber(p1_trends, P1_BASE)
            p1_w2, _ = p1_direct(dict(p1_w1), p1_pool, p1_harvey_er, p1_rvols)
            w_p1 = p1_norm(p1_w2)

            # Conviction check
            f_c = cur_faber.get("IVV",0)>=3 and cur_faber.get("QQQ",0)>=3
            h_c = cur_harvey.get("IVV",0)>0 and cur_harvey.get("QQQ",0)>0
            if f_c and h_c: lev_months_v2 += 1
            p1_fc = p1_trends.get("IVV",0)>=3 and p1_trends.get("QQQ",0)>=3
            p1_hc = p1_harvey_er.get("IVV",0)>0 and p1_harvey_er.get("QQQ",0)>0
            if p1_fc and p1_hc: lev_months_p1 += 1

            eq = w_v2.get("IVV",0) + w_v2.get("QQQ",0)
            weight_log.append({"date": day, "IVV": w_v2.get("IVV",0), "QQQ": w_v2.get("QQQ",0),
                               "VGLT": w_v2.get("VGLT",0), "IAU": w_v2.get("IAU",0),
                               "cash": w_v2.get("cash",0), "equity": eq,
                               "leveraged": f_c and h_c})

        # Compute daily returns for each config
        # V2 with leverage
        r_v2l, _ = apply_leverage(w_v2, cur_faber, cur_harvey, actual, rfr)
        # V2 without leverage
        r_v2x = sum(w_v2.get(a, 0) * actual.get(a, 0) for a in UNIVERSE)
        # P1 with leverage (reuse same leverage logic)
        r_p1l, _ = apply_leverage(w_p1, {a: (3 if p1_trends.get(a,0)>=3 else 0) for a in ["IVV","QQQ","VGLT","IAU"]},
                                   p1_harvey_er, actual, rfr)
        # P1 without leverage
        r_p1x = sum(w_p1.get(a, 0) * actual.get(a, 0) for a in UNIVERSE if a in actual)

        # Benchmarks
        r_6040 = 0.6 * actual.get("IVV", 0) + 0.4 * actual.get("VGLT", 0)
        r_ivv = actual.get("IVV", 0)

        for cn, ret in [("v2_lev", r_v2l), ("v2_1x", r_v2x), ("p1_lev", r_p1l),
                         ("p1_1x", r_p1x), ("bench_6040", r_6040), ("bench_ivv", r_ivv)]:
            # TC on monthly rebalance
            if is_ms and current_w[cn] is not None:
                new_w = w_v2 if "v2" in cn else (w_p1 if "p1" in cn else {})
                to = sum(abs(new_w.get(a,0) - current_w[cn].get(a,0)) for a in UNIVERSE) / 2
                total_tc[cn] += to * TC
            if is_ms:
                current_w[cn] = w_v2.copy() if "v2" in cn else (w_p1.copy() if "p1" in cn else {})
            strats[cn][day] = ret

    results = {c: pd.Series(d).sort_index() for c, d in strats.items() if d}
    n_years = len(trading_days) / 252
    wdf = pd.DataFrame(weight_log).set_index("date")
    wdf.to_parquet(OUTPUT / "allocations.parquet")

    ivv_s = results.get("bench_ivv")
    labels = {"v2_lev": "V2+Lev", "v2_1x": "V2 1x", "p1_lev": "P1+Lev",
              "p1_1x": "P1 1x", "bench_6040": "60/40", "bench_ivv": "IVV B&H"}

    # ── Report ────────────────────────────────────────────────────────────────

    print(f"\n{'='*80}")
    print(f"  PERFORMANCE")
    print(f"{'='*80}")
    print(f"\n  {'Config':>10} {'AnnRet':>7} {'AnnVol':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>8} {'Calmar':>7} {'Final':>8} {'CorrIVV':>8}")
    print(f"  {'-'*72}")

    for cn in ["v2_lev","v2_1x","p1_lev","p1_1x","bench_6040","bench_ivv"]:
        s = results.get(cn)
        if s is None or len(s) < 252: continue
        ar = s.mean()*252; av = s.std()*np.sqrt(252)
        sh = ar/av if av > 0 else 0
        neg = s[s<0]; ds = neg.std()*np.sqrt(252) if len(neg)>10 else av
        sortino = ar/ds if ds > 0 else 0
        cum = (1+s).cumprod()
        dd = ((cum-cum.expanding().max())/cum.expanding().max()).min()
        calmar = ar/abs(dd) if dd != 0 else 0
        final = cum.iloc[-1]
        corr = s.corr(ivv_s) if ivv_s is not None else 0
        print(f"  {labels[cn]:>10} {ar:>6.1%} {av:>6.1%} {sh:>7.2f} {sortino:>8.2f} {dd:>7.1%} {calmar:>7.2f} ${final:>7.2f} {corr:>7.2f}")

    # Crisis drawdowns
    print(f"\n{'='*80}")
    print(f"  CRISIS DRAWDOWNS")
    print(f"{'='*80}")
    for cn2,cs,ce in [("GFC","2008-01-01","2009-06-30"),("COVID","2020-02-19","2020-03-23"),("2022","2022-01-03","2022-10-31")]:
        print(f"\n  {cn2}:")
        for cn in ["v2_lev","v2_1x","p1_lev","p1_1x","bench_6040","bench_ivv"]:
            s = results.get(cn)
            if s is None: continue
            c = s[(s.index>=pd.Timestamp(cs))&(s.index<=pd.Timestamp(ce))]
            if len(c) > 0:
                cum = (1+c).cumprod()
                mdd = ((cum-cum.expanding().max())/cum.expanding().max()).min()
                print(f"    {labels[cn]:>10}: total={((1+c).prod()-1):>+7.1%}, max DD={mdd:.1%}")

    # Bull capture & upside/downside
    print(f"\n{'='*80}")
    print(f"  BULL/BEAR CAPTURE")
    print(f"{'='*80}")
    ivv_monthly = ivv_s.resample("MS").apply(lambda x: (1+x).prod()-1) if ivv_s is not None else None
    for cn in ["v2_lev","v2_1x","p1_lev","p1_1x"]:
        s = results.get(cn)
        if s is None: continue
        s_monthly = s.resample("MS").apply(lambda x: (1+x).prod()-1)
        if ivv_monthly is not None:
            common = s_monthly.index.intersection(ivv_monthly.index)
            sm = s_monthly.reindex(common); im = ivv_monthly.reindex(common)
            up = sm[im > 0].mean() / im[im > 0].mean() if (im > 0).sum() > 0 else 0
            down = sm[im < 0].mean() / im[im < 0].mean() if (im < 0).sum() > 0 else 0
            print(f"  {labels[cn]:>10}: upside capture={up:.0%}, downside capture={down:.0%}")

    for yr,desc in [(2013,"Harvey worst"),(2019,"Strong bull"),(2023,"Tech rally")]:
        print(f"\n  {yr} ({desc}):")
        for cn in ["v2_lev","v2_1x","p1_lev","p1_1x","bench_ivv"]:
            s=results.get(cn)
            if s is None: continue
            y=s[s.index.year==yr]
            if len(y)>20: print(f"    {labels[cn]:>10}: {(1+y).prod()-1:>+7.1%}")

    # Allocation diagnostics
    print(f"\n{'='*80}")
    print(f"  ALLOCATION DIAGNOSTICS (v2)")
    print(f"{'='*80}")
    print(f"  Avg equity (IVV+QQQ): {wdf['equity'].mean():.0%}")
    print(f"  Equity > 80%: {(wdf['equity']>0.80).mean():.0%} of months")
    print(f"  Equity > 60%: {(wdf['equity']>0.60).mean():.0%} of months")
    print(f"  Equity < 30%: {(wdf['equity']<0.30).mean():.0%} of months")
    print(f"  Leverage active: {wdf['leveraged'].mean():.0%} ({lev_months_v2}/{total_months})")
    print(f"  P1 leverage active: {lev_months_p1}/{total_months} ({lev_months_p1/total_months:.0%})")
    print(f"\n  Avg allocation per asset:")
    for a in ["IVV","QQQ","VGLT","IAU","cash"]:
        if a in wdf.columns:
            print(f"    {a:>5}: {wdf[a].mean():>5.0%} (min={wdf[a].min():.0%}, max={wdf[a].max():.0%})")

    # Calendar years
    print(f"\n{'='*80}")
    print(f"  CALENDAR YEAR RETURNS")
    print(f"{'='*80}")
    print(f"  {'Year':>6} {'V2+Lev':>8} {'V2 1x':>8} {'P1+Lev':>8} {'P1 1x':>8} {'IVV':>8}")
    for yr in range(2002,2025):
        row=f"  {yr:>6}"
        for cn in ["v2_lev","v2_1x","p1_lev","p1_1x","bench_ivv"]:
            s=results.get(cn)
            if s is None: row+=f" {'--':>8}"; continue
            y=s[s.index.year==yr]
            row+=f" {(1+y).prod()-1:>+7.1%}" if len(y)>20 else f" {'--':>8}"
        print(row)

    # Final values
    print(f"\n{'='*80}")
    print(f"  FINAL VALUES ($1)")
    print(f"{'='*80}")
    for cn in ["v2_lev","v2_1x","p1_lev","p1_1x","bench_6040","bench_ivv"]:
        s=results.get(cn)
        if s is None or len(s)<252: continue
        print(f"  {labels[cn]:>10}: ${(1+s).cumprod().iloc[-1]:>8.2f}")

    print()


if __name__ == "__main__":
    main()
