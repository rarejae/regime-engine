"""Signal-driven TAA v2 across three universes with graduated leverage."""

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv; load_dotenv()

import numpy as np, pandas as pd

OUTPUT = Path("taa_v2/results"); OUTPUT.mkdir(parents=True, exist_ok=True)
TC = 0.0010
CASH_FLOOR = 0.10
ALLOCABLE = 0.90
SMA_PERIODS = [6, 10, 12]
FABER_MULT = {3: 1.0, 2: 0.7, 1: 0.15, 0: 0.15}

UNIVERSES = {
    "A": {"risky": ["IVV", "QQQ", "IAU"], "equity": ["IVV", "QQQ"], "label": "IVV+QQQ+IAU"},
    "B": {"risky": ["IVV", "IAU"], "equity": ["IVV"], "label": "IVV+IAU"},
    "C": {"risky": ["IVV", "QQQ", "VGLT", "IAU"], "equity": ["IVV", "QQQ"], "label": "IVV+QQQ+VGLT+IAU"},
}
LEV_MAP = {"IVV": ("SSO", 2.0, 0.0091), "QQQ": ("QLD", 2.0, 0.0089)}


# ── Data ──────────────────────────────────────────────────────────────────────

def load_data():
    import yfinance as yf
    from fredapi import Fred

    raw_macro = pd.read_parquet("data/macro/monthly_history.parquet")
    raw_macro.index = pd.to_datetime(raw_macro.index)

    asset_ret = pd.read_parquet("data/macro/roth_asset_returns.parquet")
    asset_ret_fwd = asset_ret.shift(-1)

    frames = {}; prices = {}
    for our, ticker in [("IVV","SPY"),("QQQ","QQQ"),("VGLT","TLT"),("IAU","GLD")]:
        d = yf.download(ticker, start="1998-01-01", progress=False)
        if d is not None and not d.empty:
            p = d["Close"]
            if hasattr(p,"columns"): p=p.iloc[:,0]
            p.index = pd.to_datetime(p.index).tz_localize(None)
            frames[our] = p.pct_change()
            prices[our] = p

    key = os.environ.get("FRED_API_KEY")
    rfr_daily = pd.Series(0.0, dtype=float)
    if key:
        try:
            tb = Fred(api_key=key).get_series("DTB3", observation_start="1998-01-01")
            tb.index = pd.to_datetime(tb.index)
            frames["cash"] = tb / 36000
            rfr_daily = tb / 100 / 252
        except Exception:
            pass

    daily_ret = pd.DataFrame(frames).sort_index()
    rfr_daily = rfr_daily.reindex(daily_ret.index, method="ffill").fillna(0)
    monthly_prices = pd.DataFrame(prices).resample("MS").last()
    rvol = pd.DataFrame(prices).pct_change().rolling(63, min_periods=30).std() * np.sqrt(252)
    rvol_monthly = rvol.resample("MS").last().shift(1)

    return raw_macro, asset_ret, asset_ret_fwd, daily_ret, monthly_prices, rvol_monthly, rfr_daily


# ── Signals ───────────────────────────────────────────────────────────────────

def compute_trends(monthly_prices):
    scores = pd.DataFrame(0, index=monthly_prices.index, columns=monthly_prices.columns)
    for p in SMA_PERIODS:
        sma = monthly_prices.rolling(p, min_periods=p).mean()
        scores += (monthly_prices > sma).astype(int)
    return scores.shift(1)

def compute_zscores(raw_macro):
    result = pd.DataFrame(index=raw_macro.index)
    for col in raw_macro.columns:
        chg = raw_macro[col] - raw_macro[col].shift(12)
        std = chg.rolling(120, min_periods=60).std()
        result[f"{col}_z"] = (chg / std.replace(0, np.nan)).clip(-3, 3)
    return result.shift(1).dropna(how="all")

def find_similar(z_data, dt, exclude=36, pctl=0.15):
    if dt not in z_data.index: raise ValueError
    target = z_data.loc[dt].values
    cutoff = dt - pd.DateOffset(months=exclude)
    cands = z_data[z_data.index <= cutoff]
    if len(cands) == 0: raise ValueError
    diffs = np.array(cands.values - target, dtype=np.float64)
    dists = pd.Series(np.sqrt(np.sum(diffs**2, axis=1)), index=cands.index)
    return dists[dists <= dists.quantile(pctl)].index.tolist()


# ── Allocation + Leverage ─────────────────────────────────────────────────────

def allocate(harvey_er, rvols, faber_scores, risky_assets):
    scores = {}
    for a in risky_assets:
        er = harvey_er.get(a, 0)
        vol = rvols.get(a, 0.15)
        mult = FABER_MULT.get(faber_scores.get(a, 0), 0.15)
        scores[a] = (er / max(vol, 0.01)) * mult if er > 0 else 0.0
    total = sum(max(s, 0) for s in scores.values())
    w = {"cash": CASH_FLOOR}
    for a in risky_assets:
        w[a] = ALLOCABLE * max(scores[a], 0) / total if total > 0 else 0.0
    if total <= 0:
        w["cash"] = 1.0
    return w


def get_leverage_tier(faber_scores, harvey_er, equity_assets, tier1_medians):
    """Determine leverage tier 0/1/2."""
    # Tier 1: all equity at 3/3 Faber AND Harvey ER > 0
    all_strong = all(faber_scores.get(a, 0) >= 3 for a in equity_assets)
    all_positive = all(harvey_er.get(a, 0) > 0 for a in equity_assets)
    if not (all_strong and all_positive):
        return 0

    # Tier 2: additionally, each equity ER > expanding median of Tier 1+ ERs
    all_above_median = True
    for a in equity_assets:
        med = tier1_medians.get(a, 0)
        if harvey_er.get(a, 0) <= med:
            all_above_median = False
            break

    return 2 if all_above_median else 1


def apply_leverage_daily(weights, tier, equity_assets, daily_rets, rfr):
    """Compute daily portfolio return with graduated leverage."""
    if tier == 0:
        return sum(weights.get(a, 0) * daily_rets.get(a, 0) for a in weights)

    sub_frac = 0.25 if tier == 1 else 0.50
    port_ret = 0.0
    for a in weights:
        if a in equity_assets and a in LEV_MAP:
            lev_name, lev_mult, expense = LEV_MAP[a]
            w = weights.get(a, 0)
            w_lev = w * sub_frac
            w_unlev = w - w_lev
            r_unlev = daily_rets.get(a, 0)
            r_lev = lev_mult * r_unlev - (lev_mult - 1) * rfr - expense / 252
            port_ret += w_unlev * r_unlev + w_lev * r_lev
        else:
            port_ret += weights.get(a, 0) * daily_rets.get(a, 0)
    return port_ret


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("  MULTI-UNIVERSE TAA v2 WITH GRADUATED LEVERAGE")
    print("=" * 80)

    raw_macro, asset_ret, asset_ret_fwd, daily_ret, monthly_prices, rvol_monthly, rfr_daily = load_data()
    trend_df = compute_trends(monthly_prices)
    z_data = compute_zscores(raw_macro)
    z_clean = z_data[[c for c in z_data.columns if c.endswith("_z")]].dropna()

    # Phase 1 reproduction
    from taa.faber import compute_trend_scores as p1_trends, apply_faber_filter as p1_faber
    from taa.harvey import compute_zscore_variables as p1_zs, find_similar_months as p1_sim, compute_expected_returns as p1_er
    from taa.allocation import BASELINE as P1_BASE, direct_capital as p1_dir, normalize as p1_norm
    from taa.data import load_monthly_prices as p1_mp, load_realized_vols as p1_rv, load_monthly_macro as p1_mm
    p1_trend_df = p1_trends(p1_mp())
    p1_z = p1_zs(p1_mm())
    p1_zc = p1_z[[c for c in p1_z.columns if c.endswith("_z")]].dropna()
    p1_rvm = p1_rv()
    p1_ar = pd.read_parquet("data/macro/roth_asset_returns.parquet")
    for c in list(p1_ar.columns):
        if c not in P1_BASE: p1_ar = p1_ar.drop(columns=[c])
    p1_arf = p1_ar.shift(-1)

    common_start = max(daily_ret.dropna(how="all").index.min(), pd.Timestamp("2002-01-01"))
    trading_days = daily_ret.loc[common_start:].index
    print(f"Backtest: {len(trading_days)} days ({common_start.date()} to {trading_days.max().date()})")

    # Assertion
    z_before = z_clean.index[z_clean.index < pd.Timestamp("2007-12-01")]
    assert len(z_before) > 0 and z_before[-1] < pd.Timestamp("2007-12-01"), "Z-score look-ahead!"
    print(f"Signal alignment: PASS")

    # All configs
    config_names = []
    for u_key in ["A", "B", "C"]:
        config_names.extend([f"{u_key}_lev", f"{u_key}_1x"])
    config_names.extend(["p1_lev125", "p1_1x", "bench_6040", "bench_ivv"])

    strats = {cn: {} for cn in config_names}
    current_w = {cn: None for cn in config_names}

    # Per-universe state
    u_state = {}
    for u_key, u_cfg in UNIVERSES.items():
        u_state[u_key] = {
            "faber": {a: 3 for a in u_cfg["risky"]},
            "harvey_er": {a: 0.0 for a in u_cfg["risky"] + ["cash"]},
            "rvols": {a: 0.15 for a in u_cfg["risky"]},
            "weights": {"cash": 1.0},
            "tier1_er_history": {a: [] for a in u_cfg["equity"]},
            "tier1_medians": {a: 0.0 for a in u_cfg["equity"]},
            "tier": 0,
            "tier_counts": {0: 0, 1: 0, 2: 0},
        }

    # Phase 1 state
    p1_state = {"trends": {a: True for a in P1_BASE if a != "cash"},
                "harvey_er": {a: 0 for a in P1_BASE}, "w": dict(P1_BASE)}

    total_months = 0

    for day in trading_days:
        if day not in daily_ret.index: continue
        dr = daily_ret.loc[day]
        actual = {a: float(dr[a]) for a in daily_ret.columns if pd.notna(dr[a])}
        rfr = float(rfr_daily.get(day, 0))
        is_ms = (day == trading_days[0] or day.month != trading_days[trading_days.get_loc(day)-1].month)

        if is_ms:
            total_months += 1

            # Update all universe signals
            for u_key, u_cfg in UNIVERSES.items():
                us = u_state[u_key]
                # Faber
                ts_cands = trend_df.index[trend_df.index <= day]
                if len(ts_cands) > 0:
                    ts = ts_cands[-1]
                    for a in u_cfg["risky"]:
                        if a in trend_df.columns:
                            v = trend_df.loc[ts, a]
                            us["faber"][a] = int(v) if pd.notna(v) else 0

                # Harvey
                z_cands = z_clean.index[z_clean.index < day]
                if len(z_cands) > 0:
                    z_dt = z_cands[-1]
                    try:
                        sim = find_similar(z_clean, z_dt)
                        for a in u_cfg["risky"] + ["cash"]:
                            if a not in asset_ret_fwd.columns:
                                us["harvey_er"][a] = 0; continue
                            rets = [asset_ret_fwd.loc[d, a] for d in sim
                                    if d in asset_ret_fwd.index and pd.notna(asset_ret_fwd.loc[d, a])]
                            us["harvey_er"][a] = float(np.mean(rets)) if rets else 0
                    except ValueError:
                        pass

                # Rvols
                rv_cands = rvol_monthly.index[rvol_monthly.index <= day]
                if len(rv_cands) > 0:
                    rv = rv_cands[-1]
                    for a in u_cfg["risky"]:
                        if a in rvol_monthly.columns:
                            v = rvol_monthly.loc[rv, a]
                            us["rvols"][a] = float(v) if pd.notna(v) and v > 0 else 0.15

                # Allocate
                us["weights"] = allocate(us["harvey_er"], us["rvols"], us["faber"], u_cfg["risky"])

                # Leverage tier
                tier = get_leverage_tier(us["faber"], us["harvey_er"], u_cfg["equity"], us["tier1_medians"])
                us["tier"] = tier
                us["tier_counts"][tier] += 1

                # Update expanding medians for Tier 2
                if tier >= 1:
                    for a in u_cfg["equity"]:
                        us["tier1_er_history"][a].append(us["harvey_er"].get(a, 0))
                        us["tier1_medians"][a] = float(np.median(us["tier1_er_history"][a]))

            # Phase 1
            p1_ts_cands = p1_trend_df.index[p1_trend_df.index <= day]
            if len(p1_ts_cands) > 0:
                for a in P1_BASE:
                    if a == "cash": continue
                    if a in p1_trend_df.columns:
                        s = p1_trend_df.loc[p1_ts_cands[-1], a]
                        p1_state["trends"][a] = int(s) if pd.notna(s) else 0
            p1_z_cands = p1_zc.index[p1_zc.index < day]
            if len(p1_z_cands) > 0:
                try:
                    p1_s, _ = p1_sim(p1_zc, p1_z_cands[-1])
                    p1_state["harvey_er"] = p1_er(p1_s, p1_arf, list(P1_BASE.keys()))
                except ValueError:
                    pass
            p1_rv_cands = p1_rvm.index[p1_rvm.index <= day]
            p1_rvols = {}
            if len(p1_rv_cands) > 0:
                for a in P1_BASE:
                    if a == "cash": continue
                    if a in p1_rvm.columns:
                        v = p1_rvm.loc[p1_rv_cands[-1], a]
                        p1_rvols[a] = float(v) if pd.notna(v) and v > 0 else 0.15
            p1_w1, p1_pool = p1_faber(p1_state["trends"], P1_BASE)
            p1_w2, _ = p1_dir(dict(p1_w1), p1_pool, p1_state["harvey_er"], p1_rvols)
            p1_state["w"] = p1_norm(p1_w2)

        # Daily returns
        for u_key, u_cfg in UNIVERSES.items():
            us = u_state[u_key]
            # With leverage
            r_lev = apply_leverage_daily(us["weights"], us["tier"], u_cfg["equity"], actual, rfr)
            strats[f"{u_key}_lev"][day] = r_lev
            # Without leverage
            r_1x = sum(us["weights"].get(a, 0) * actual.get(a, 0) for a in us["weights"])
            strats[f"{u_key}_1x"][day] = r_1x

        # Phase 1
        p1_w = p1_state["w"]
        p1_fc = p1_state["trends"].get("IVV",0)>=3 and p1_state["trends"].get("QQQ",0)>=3
        p1_hc = p1_state["harvey_er"].get("IVV",0)>0 and p1_state["harvey_er"].get("QQQ",0)>0
        p1_tier = 1 if (p1_fc and p1_hc) else 0
        strats["p1_lev125"][day] = apply_leverage_daily(p1_w, p1_tier, ["IVV","QQQ"], actual, rfr)
        strats["p1_1x"][day] = sum(p1_w.get(a, 0) * actual.get(a, 0) for a in p1_w if a in actual)

        # Benchmarks
        strats["bench_6040"][day] = 0.6 * actual.get("IVV", 0) + 0.4 * actual.get("VGLT", 0)
        strats["bench_ivv"][day] = actual.get("IVV", 0)

    results = {c: pd.Series(d).sort_index() for c, d in strats.items() if d}
    n_years = len(trading_days) / 252
    ivv_s = results.get("bench_ivv")

    # ── Report ────────────────────────────────────────────────────────────────

    all_labels = {}
    for u_key in ["A","B","C"]:
        all_labels[f"{u_key}_lev"] = f"{u_key}+Lev"
        all_labels[f"{u_key}_1x"] = f"{u_key} 1x"
    all_labels.update({"p1_lev125": "P1+Lev", "p1_1x": "P1 1x", "bench_6040": "60/40", "bench_ivv": "IVV B&H"})

    print(f"\n{'='*80}")
    print(f"  PERFORMANCE (all configs)")
    print(f"{'='*80}")
    order = ["A_lev","A_1x","B_lev","B_1x","C_lev","C_1x","p1_lev125","p1_1x","bench_6040","bench_ivv"]
    print(f"\n  {'Config':>10} {'AnnRet':>7} {'AnnVol':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>8} {'Calmar':>7} {'Final':>8}")
    print(f"  {'-'*68}")
    for cn in order:
        s = results.get(cn)
        if s is None or len(s) < 252: continue
        ar=s.mean()*252; av=s.std()*np.sqrt(252); sh=ar/av if av>0 else 0
        neg=s[s<0]; ds=neg.std()*np.sqrt(252) if len(neg)>10 else av
        sortino=ar/ds if ds>0 else 0
        cum=(1+s).cumprod(); dd=((cum-cum.expanding().max())/cum.expanding().max()).min()
        calmar=ar/abs(dd) if dd!=0 else 0; final=cum.iloc[-1]
        print(f"  {all_labels[cn]:>10} {ar:>6.1%} {av:>6.1%} {sh:>7.2f} {sortino:>8.2f} {dd:>7.1%} {calmar:>7.2f} ${final:>7.2f}")

    # Crisis
    print(f"\n{'='*80}")
    print(f"  CRISIS DRAWDOWNS")
    print(f"{'='*80}")
    for cn2,cs,ce in [("GFC","2008-01-01","2009-06-30"),("COVID","2020-02-19","2020-03-23"),("2022","2022-01-03","2022-10-31")]:
        print(f"\n  {cn2}:")
        for cn in order:
            s=results.get(cn)
            if s is None: continue
            c=s[(s.index>=pd.Timestamp(cs))&(s.index<=pd.Timestamp(ce))]
            if len(c)>0:
                cum=(1+c).cumprod()
                mdd=((cum-cum.expanding().max())/cum.expanding().max()).min()
                print(f"    {all_labels[cn]:>10}: {((1+c).prod()-1):>+7.1%} (DD {mdd:.1%})")

    # Bull capture
    print(f"\n{'='*80}")
    print(f"  BULL CAPTURE")
    print(f"{'='*80}")
    for yr,desc in [(2013,"Harvey worst"),(2019,"Strong bull"),(2023,"Tech rally")]:
        print(f"\n  {yr} ({desc}):")
        for cn in ["A_lev","B_lev","C_lev","p1_lev125","bench_ivv"]:
            s=results.get(cn)
            if s is None: continue
            y=s[s.index.year==yr]
            if len(y)>20: print(f"    {all_labels[cn]:>10}: {(1+y).prod()-1:>+7.1%}")

    # Tier diagnostics
    print(f"\n{'='*80}")
    print(f"  LEVERAGE TIER DIAGNOSTICS")
    print(f"{'='*80}")
    for u_key in ["A","B","C"]:
        us = u_state[u_key]
        tc = us["tier_counts"]
        tot = sum(tc.values())
        t1_pct = (tc[1]+tc[2])/tot if tot>0 else 0
        t2_pct = tc[2]/tot if tot>0 else 0
        t2_of_t1 = tc[2]/(tc[1]+tc[2]) if (tc[1]+tc[2])>0 else 0
        print(f"\n  Config {u_key} ({UNIVERSES[u_key]['label']}):")
        print(f"    Tier 0 (1.0x): {tc[0]:>4} months ({tc[0]/tot:.0%})")
        print(f"    Tier 1 (1.25x):{tc[1]:>4} months ({tc[1]/tot:.0%})")
        print(f"    Tier 2 (1.5x): {tc[2]:>4} months ({tc[2]/tot:.0%})")
        print(f"    Tier 2 as % of Tier 1+: {t2_of_t1:.0%}")
        print(f"    Avg equity: {np.mean([us['weights'].get('IVV',0)+us['weights'].get('QQQ',0)]):.0%}")

    # Calendar years
    print(f"\n{'='*80}")
    print(f"  CALENDAR YEAR RETURNS")
    print(f"{'='*80}")
    show = ["A_lev","B_lev","C_lev","p1_lev125","bench_ivv"]
    hdr = f"  {'Year':>6}"
    for cn in show: hdr += f" {all_labels[cn]:>8}"
    print(hdr)
    for yr in range(2002,2025):
        row=f"  {yr:>6}"
        for cn in show:
            s=results.get(cn)
            if s is None: row+=f" {'--':>8}"; continue
            y=s[s.index.year==yr]
            row+=f" {(1+y).prod()-1:>+7.1%}" if len(y)>20 else f" {'--':>8}"
        print(row)

    # Final values
    print(f"\n{'='*80}")
    print(f"  FINAL VALUES ($1)")
    print(f"{'='*80}")
    for cn in order:
        s=results.get(cn)
        if s is None or len(s)<252: continue
        print(f"  {all_labels[cn]:>10}: ${(1+s).cumprod().iloc[-1]:>8.2f}")

    print()


if __name__ == "__main__":
    main()
