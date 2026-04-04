"""Component value-add analysis: isolate Faber, Harvey, and leverage contributions."""

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
DATA_DIR = Path("data/macro")
SMA_PERIODS = [6, 10, 12]
EQUITY = ["IVV", "QQQ"]
BASE = {"IVV": 0.45, "QQQ": 0.25, "IEF": 0.10, "IAU": 0.10, "cash": 0.10}
ASSETS = list(BASE.keys())
TIER_SUBS = {0: 0.0, 1: 0.30, 2: 0.65}


def normalize(w):
    w = dict(w)
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


def alloc_static():
    return dict(BASE)


def alloc_faber_only(strengths):
    """Faber filter only — pool goes to cash."""
    w = {}
    for a in ASSETS:
        if a == "cash": w[a] = BASE[a]; continue
        s = strengths.get(a, 0)
        if s >= 3: w[a] = BASE[a]
        elif s == 2: w[a] = BASE[a] * 0.70; w["cash"] = w.get("cash", 0) + BASE[a] * 0.30
        else: w[a] = 0.0; w["cash"] = w.get("cash", 0) + BASE[a]
    return normalize(w)


def alloc_harvey_only(harvey_er, rvols):
    """Harvey directs all capital — no Faber gate."""
    scores = {}
    for a in ASSETS:
        if a == "cash": continue
        er = harvey_er.get(a, 0)
        vol = rvols.get(a, 0.15)
        scores[a] = max(er, 0.001) / max(vol, 0.01)
    total = sum(scores.values())
    if total <= 0:
        return dict(BASE)
    w = {a: sc / total * 0.90 for a, sc in scores.items()}
    w["cash"] = 0.10
    return normalize(w)


def alloc_faber_harvey(strengths, harvey_er, rvols, carry, value):
    """Full Faber + Harvey (+ HMM/Kritzman) — no leverage."""
    w = {}; pool = 0.0
    for a in ASSETS:
        if a == "cash": w[a] = BASE[a]; continue
        s = strengths.get(a, 0)
        if s >= 3: w[a] = BASE[a]
        elif s == 2: w[a] = BASE[a] * 0.70; pool += BASE[a] * 0.30
        else: w[a] = 0.0; pool += BASE[a]
    # Harvey directs pool
    if pool > 0.001:
        cands = {}
        for a in ASSETS:
            if a == "cash" or w.get(a, 0) <= 0: continue
            adj = harvey_er.get(a, 0) + carry.get(a, 0)
            if adj <= 0: continue
            sc = adj / max(rvols.get(a, 0.15), 0.01) + 0.01 * value.get(a, 0)
            if sc > 0: cands[a] = sc
        if not cands:
            w["cash"] = w.get("cash", 0) + pool
        elif len(cands) == 1:
            a = list(cands.keys())[0]
            alloc = max(min(pool, 0.40 - w.get(a, 0)), 0)
            w[a] = w.get(a, 0) + alloc; w["cash"] = w.get("cash", 0) + (pool - alloc)
        else:
            ts = sum(cands.values())
            for a, sc in cands.items(): w[a] = w.get(a, 0) + pool * (sc / ts)
    return normalize(w)


def alloc_full(strengths, harvey_er, rvols, carry, value, bp, tp, arz):
    """Full system: Faber + Harvey + HMM + Kritzman."""
    w = alloc_faber_harvey(strengths, harvey_er, rvols, carry, value)
    # HMM
    if bp > 0.7:
        for a in EQUITY:
            if w.get(a, 0) > 0: b = w[a] * 0.15; w[a] += b; w["cash"] = max(w.get("cash", 0) - b, 0)
    elif bp < 0.3:
        for a in EQUITY:
            if w.get(a, 0) > 0: r = w[a] * 0.15; w[a] -= r; w["cash"] = w.get("cash", 0) + r
    # Kritzman
    if tp > 0.95 and arz > 2.0:
        for a in EQUITY:
            if w.get(a, 0) > 0: f = w[a] * 0.50; w[a] -= f; w["cash"] = w.get("cash", 0) + f
    return normalize(w)


def main():
    lines = []
    def pr(s=""): print(s); lines.append(s)

    config = RegimeConfig()
    raw_macro = fetch_monthly_history(config)
    transformed = transform_variables(raw_macro, config)
    z_data = get_valid_zscored(transformed, config)
    z_data_lagged = z_data.shift(1).dropna()
    asset_ret = pd.read_parquet(DATA_DIR / "roth_asset_returns_5asset.parquet").rename(columns={"TSY": "IEF"})
    asset_ret_fwd = asset_ret.shift(-1)

    print("Loading data...")
    daily_ret = fetch_daily_etf_returns()
    tsy_daily = pd.read_parquet(DATA_DIR / "treasury_daily_prices.parquet")["TSY"]
    daily_ret["IEF"] = tsy_daily.pct_change()

    import yfinance as yf
    ep = {}
    for our, ticker in [("IVV", "SPY"), ("QQQ", "QQQ"), ("IAU", "GLD")]:
        d = yf.download(ticker, start="1998-01-01", progress=False)
        if d is not None and not d.empty:
            p = d["Close"]
            if hasattr(p, "columns"): p = p.iloc[:, 0]
            p.index = pd.to_datetime(p.index).tz_localize(None)
            ep[our] = p
    ep["IEF"] = tsy_daily
    prices_df = pd.DataFrame(ep).sort_index()
    monthly_prices = prices_df.resample("MS").last()
    sma_dfs = [monthly_prices.rolling(p, min_periods=p).mean().shift(1) for p in SMA_PERIODS]
    rvol_monthly = prices_df.pct_change().rolling(63, min_periods=30).std().mul(np.sqrt(252)).resample("MS").last().shift(1)

    print("Computing carry/value...")
    carry_df = compute_carry(prices_df)
    value_df = compute_value(prices_df)
    from fredapi import Fred
    fkey = os.environ.get("FRED_API_KEY")
    if fkey:
        try:
            f = Fred(api_key=fkey)
            gs20 = f.get_series("GS20", observation_start="1998-01-01")
            tb3 = f.get_series("DTB3", observation_start="1998-01-01")
            gs20.index = pd.to_datetime(gs20.index); tb3.index = pd.to_datetime(tb3.index)
            carry_df["IEF"] = (gs20.resample("MS").last() / 1200 - tb3.resample("MS").last() / 1200).shift(1)
        except Exception: pass

    print("Fitting HMM...")
    spy_raw = fetch_daily_data(); hmm_feat = compute_features(spy_raw)
    hmm_pred = fit_and_predict_rolling(hmm_feat).set_index("date")
    zone_raw = pd.Series("neutral", index=hmm_pred.index)
    zone_raw[hmm_pred["bull_prob"] > 0.7] = "bull"; zone_raw[hmm_pred["bull_prob"] < 0.3] = "bear"
    hmm_pred["zone"] = apply_persistence_filter(zone_raw)

    basket = fetch_daily_basket()
    turb_m = compute_turbulence_pctl(compute_turbulence(basket), window=252).resample("MS").last().shift(1)
    ar_z_m = compute_ar_zscore(compute_absorption_ratio(basket, n_components=2)).resample("MS").last().shift(1)

    rfr_daily = pd.Series(0.0, index=daily_ret.index)
    if fkey:
        try:
            tb_d = Fred(api_key=fkey).get_series("DTB3", observation_start="1998-01-01")
            tb_d.index = pd.to_datetime(tb_d.index)
            rfr_daily = (tb_d / 100 / 252).reindex(daily_ret.index, method="ffill").fillna(0)
        except Exception: pass

    pre_start = pd.Timestamp("2002-01-01")
    pre_ers = {"IVV": [], "QQQ": []}
    for z_dt in z_data_lagged.index[z_data_lagged.index < pre_start]:
        try:
            sim = compute_distances(z_data_lagged, z_dt, config)
            for a in ["IVV", "QQQ"]:
                if a in asset_ret_fwd.columns:
                    r = [asset_ret_fwd.loc[d, a] for d in sim.similar_dates if d in asset_ret_fwd.index and pd.notna(asset_ret_fwd.loc[d, a])]
                    if r: pre_ers[a].append(np.mean(r))
        except ValueError: pass
    SEED = {a: float(np.mean(pre_ers[a])) if pre_ers[a] else 0.005 for a in ["IVV", "QQQ"]}

    common_start = max(daily_ret.dropna(how="all").index.min(), hmm_pred.index.min(), pd.Timestamp("2002-07-01"))
    trading_days = daily_ret.loc[common_start:].index

    pr("=" * 80)
    pr("  COMPONENT VALUE-ADD ANALYSIS")
    pr("=" * 80)
    pr(f"\nBacktest: {common_start.date()} to {trading_days.max().date()}")
    pr(f"Universe: IVV 45%, QQQ 25%, IEF 10%, IAU 10%, Cash 10%")

    # ── Run all 5 configs ─────────────────────────────────────────────────────

    CFGS = ["Static", "Faber Only", "Harvey Only", "Faber+Harvey (1x)", "Full System"]
    strats = {c: {} for c in CFGS}
    ivv_rets = {}

    # Shared signal state
    strengths = {a: 3 for a in ASSETS if a != "cash"}
    harvey_er = {a: 0.0 for a in ASSETS}
    rvols_v = {}; carry_v = {}; value_v = {}
    bp = 0.5; tp = 0.5; arz = 0.0
    tier = 0
    tier_hist = {"IVV": list(pre_ers["IVV"]), "QQQ": list(pre_ers["QQQ"])}
    tier_med = dict(SEED)

    # Per-config weights
    weights = {c: dict(BASE) for c in CFGS}

    for day_idx, day in enumerate(trading_days):
        if day not in daily_ret.index: continue
        dr = daily_ret.loc[day]
        is_ms = (day_idx == 0 or day.month != trading_days[day_idx - 1].month)
        is_friday = day.weekday() == 4
        rfr = float(rfr_daily.get(day, 0))
        if "IVV" in dr.index and pd.notna(dr["IVV"]): ivv_rets[day] = float(dr["IVV"])
        avail = [a for a in ASSETS if a in dr.index and pd.notna(dr[a])]
        if len(avail) < 2: continue
        actual = {a: float(dr[a]) for a in avail}

        # Update monthly signals
        if is_ms:
            z_c = z_data_lagged.index[z_data_lagged.index < day]
            if len(z_c) > 0:
                try:
                    sim = compute_distances(z_data_lagged, z_c[-1], config)
                    for a in ASSETS:
                        if a not in asset_ret_fwd.columns: harvey_er[a] = 0; continue
                        r = [asset_ret_fwd.loc[d, a] for d in sim.similar_dates if d in asset_ret_fwd.index and pd.notna(asset_ret_fwd.loc[d, a])]
                        harvey_er[a] = np.mean(r) if r else 0
                except ValueError: pass
            rv_c = rvol_monthly.index[rvol_monthly.index <= day]
            if len(rv_c) > 0:
                for a in ASSETS:
                    if a != "cash" and a in rvol_monthly.columns:
                        v = rvol_monthly.loc[rv_c[-1], a]; rvols_v[a] = float(v) if pd.notna(v) and v > 0 else 0.15
            carry_v = {}; value_v = {}
            for a in ASSETS:
                if a in carry_df.columns:
                    cc = carry_df.index[carry_df.index <= day]
                    if len(cc) > 0: cv = carry_df.loc[cc[-1], a]; carry_v[a] = float(cv) if pd.notna(cv) else 0
                if a in value_df.columns:
                    vc = value_df.index[value_df.index <= day]
                    if len(vc) > 0: vv = value_df.loc[vc[-1], a]; value_v[a] = float(vv) if pd.notna(vv) else 0
            prev_m = day - pd.DateOffset(months=1)
            hm = hmm_pred[(hmm_pred.index >= prev_m) & (hmm_pred.index < day)]
            bp = hm["bull_prob"].mean() if len(hm) > 0 else 0.5
            if pd.isna(bp): bp = 0.5
            tp = turb_m.get(day, 0.5) if day in turb_m.index else 0.5
            if pd.isna(tp): tp = 0.5
            arz = ar_z_m.get(day, 0) if day in ar_z_m.index else 0
            if pd.isna(arz): arz = 0

        # Weekly rebalance
        if is_friday or is_ms:
            # Update Faber scores
            for a in ASSETS:
                if a == "cash" or a not in prices_df.columns: continue
                pb = prices_df.loc[:day, a]
                if len(pb) == 0: continue
                cp = pb.iloc[-1]; score = 0
                for sdf in sma_dfs:
                    sd = sdf.index[sdf.index <= pd.Timestamp(f"{day.year}-{day.month:02d}-01")]
                    if len(sd) == 0: continue
                    sv = sdf.loc[sd[-1], a] if a in sdf.columns else np.nan
                    if pd.notna(sv) and cp > sv: score += 1
                strengths[a] = score

            # Compute weights for each config
            weights["Static"] = alloc_static()
            weights["Faber Only"] = alloc_faber_only(strengths)
            weights["Harvey Only"] = alloc_harvey_only(harvey_er, rvols_v)
            weights["Faber+Harvey (1x)"] = alloc_faber_harvey(strengths, harvey_er, rvols_v, carry_v, value_v)
            weights["Full System"] = alloc_full(strengths, harvey_er, rvols_v, carry_v, value_v, bp, tp, arz)

            # Tier for Full System
            f_c = strengths.get("IVV", 0) >= 3 and strengths.get("QQQ", 0) >= 3
            h_c = harvey_er.get("IVV", 0) > 0 and harvey_er.get("QQQ", 0) > 0
            if f_c and h_c:
                above = all(harvey_er.get(a, 0) > tier_med.get(a, 0) for a in ["IVV", "QQQ"])
                tier = 2 if above else 1
                for a in ["IVV", "QQQ"]:
                    tier_hist[a].append(harvey_er.get(a, 0))
                    tier_med[a] = float(np.median(tier_hist[a]))
            else: tier = 0

        # Compute daily returns
        for cfg in CFGS:
            wp = weights[cfg]
            iw = wp.get("IVV", 0); qw = wp.get("QQQ", 0)
            ir = actual.get("IVV", 0); qr = actual.get("QQQ", 0)
            base_r = sum(wp.get(a, 0) * actual.get(a, 0) for a in avail if a not in ["IVV", "QQQ"])

            if cfg == "Full System":
                sub = TIER_SUBS.get(tier, 0)
                if sub > 0:
                    sr = 2 * ir - rfr - 0.0091 / 252; ql = 2 * qr - rfr - 0.0089 / 252
                    strats[cfg][day] = iw * (1 - sub) * ir + iw * sub * sr + qw * (1 - sub) * qr + qw * sub * ql + base_r
                else:
                    strats[cfg][day] = iw * ir + qw * qr + base_r
            else:
                strats[cfg][day] = iw * ir + qw * qr + base_r

    results = {k: pd.Series(v).sort_index() for k, v in strats.items()}
    ivv_s = pd.Series(ivv_rets).sort_index()

    # ── Report ────────────────────────────────────────────────────────────────

    def perf(s):
        ar = s.mean() * 252; av = s.std() * np.sqrt(252)
        sh = ar / av if av > 0 else 0
        neg = s[s < 0]; ds = neg.std() * np.sqrt(252) if len(neg) > 10 else av
        so = ar / ds if ds > 0 else 0
        cum = (1 + s).cumprod(); dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
        cal = ar / abs(dd) if dd != 0 else 0; final = cum.iloc[-1]
        return {"ar": ar, "av": av, "sh": sh, "so": so, "dd": dd, "cal": cal, "final": final}

    perfs = {c: perf(results[c]) for c in CFGS}

    pr(f"\n\nPERFORMANCE (2002-2026)")
    pr("-" * 85)
    pr(f"  {'Config':<22} {'Return':>8} {'Vol':>7} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'Terminal':>10}")
    pr(f"  {'-' * 22} {'-' * 8} {'-' * 7} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 10}")
    for c in CFGS:
        p = perfs[c]
        pr(f"  {c:<22} {p['ar']:>7.1%} {p['av']:>6.1%} {p['sh']:>8.3f} {p['so']:>8.3f} {p['dd']:>7.1%} ${p['final']:>9.2f}")

    # Value-add decomposition
    pr(f"\n\nVALUE-ADD DECOMPOSITION")
    pr("-" * 75)
    pr(f"  {'Component':<30} {'Sharpe Δ':>10} {'Return Δ':>10} {'MaxDD Δ':>10}")
    pr(f"  {'-' * 30} {'-' * 10} {'-' * 10} {'-' * 10}")

    ps = perfs["Static"]; pf = perfs["Faber Only"]; ph = perfs["Harvey Only"]
    pfh = perfs["Faber+Harvey (1x)"]; pfs = perfs["Full System"]

    decomp = [
        ("Faber (vs Static)", pf, ps),
        ("Harvey (vs Static)", ph, ps),
        ("Faber+Harvey (vs Static)", pfh, ps),
        ("Harvey marginal (vs Faber)", pfh, pf),
        ("HMM+Kritzman (vs F+H 1x)", pfs, pfh),  # Note: includes leverage
        ("Leverage (vs F+H 1x)", pfs, pfh),
    ]
    for label, pa, pb in decomp:
        dsh = pa["sh"] - pb["sh"]
        dret = pa["ar"] - pb["ar"]
        ddd = pa["dd"] - pb["dd"]
        pr(f"  {label:<30} {dsh:>+10.3f} {dret:>+9.1%} {ddd:>+9.1%}")

    # Attribution percentages
    total_sh_add = pfs["sh"] - ps["sh"]
    faber_sh = pf["sh"] - ps["sh"]
    harvey_marginal_sh = pfh["sh"] - pf["sh"]
    lev_sh = pfs["sh"] - pfh["sh"]

    pr(f"\n\nALPHA ATTRIBUTION (Sharpe decomposition)")
    pr("-" * 50)
    pr(f"  Total system Sharpe add vs static: {total_sh_add:+.3f}")
    if total_sh_add != 0:
        pr(f"  Faber contribution:      {faber_sh:+.3f} ({faber_sh / total_sh_add * 100:>5.0f}%)")
        pr(f"  Harvey marginal:         {harvey_marginal_sh:+.3f} ({harvey_marginal_sh / total_sh_add * 100:>5.0f}%)")
        pr(f"  Leverage + HMM/Kritz:    {lev_sh:+.3f} ({lev_sh / total_sh_add * 100:>5.0f}%)")

    total_ret_add = pfs["ar"] - ps["ar"]
    faber_ret = pf["ar"] - ps["ar"]
    harvey_marginal_ret = pfh["ar"] - pf["ar"]
    lev_ret = pfs["ar"] - pfh["ar"]

    pr(f"\n  Total system Return add vs static: {total_ret_add:+.1%}")
    if total_ret_add != 0:
        pr(f"  Faber contribution:      {faber_ret:+.1%} ({faber_ret / total_ret_add * 100:>5.0f}%)")
        pr(f"  Harvey marginal:         {harvey_marginal_ret:+.1%} ({harvey_marginal_ret / total_ret_add * 100:>5.0f}%)")
        pr(f"  Leverage + HMM/Kritz:    {lev_ret:+.1%} ({lev_ret / total_ret_add * 100:>5.0f}%)")

    # Crisis analysis
    pr(f"\n\n{'=' * 80}")
    pr(f"  CRISIS ANALYSIS")
    pr(f"{'=' * 80}")

    crises = [("GFC", "2008-09-01", "2009-03-31"), ("COVID", "2020-02-19", "2020-03-23"), ("2022 Bear", "2022-01-03", "2022-10-31")]
    for cname, cs, ce in crises:
        pr(f"\n  {cname}:")
        pr(f"  {'Config':<22} {'Return':>9} {'MaxDD':>9}")
        pr(f"  {'-' * 22} {'-' * 9} {'-' * 9}")
        for c in CFGS:
            s = results[c]; cr = s[(s.index >= pd.Timestamp(cs)) & (s.index <= pd.Timestamp(ce))]
            if len(cr) == 0: continue
            cum = (1 + cr).prod() - 1
            cum_c = (1 + cr).cumprod(); dd = ((cum_c - cum_c.expanding().max()) / cum_c.expanding().max()).min()
            pr(f"  {c:<22} {cum:>+8.1%} {dd:>8.1%}")

    # Calendar years
    pr(f"\n\n{'=' * 80}")
    pr(f"  CALENDAR YEAR RETURNS")
    pr(f"{'=' * 80}")
    pr(f"  {'Year':>6} {'Static':>8} {'Faber':>8} {'Harvey':>8} {'F+H 1x':>8} {'Full':>8} {'IVV':>8}")
    for yr in range(2003, 2026):
        row = f"  {yr:>6}"
        for c in CFGS:
            s = results[c]; y = s[s.index.year == yr]
            row += f" {(1 + y).prod() - 1:>+7.1%}" if len(y) > 20 else f" {'--':>8}"
        y_ivv = ivv_s[ivv_s.index.year == yr]
        row += f" {(1 + y_ivv).prod() - 1:>+7.1%}" if len(y_ivv) > 20 else f" {'--':>8}"
        pr(row)

    pr(f"\n  FINAL VALUES ($1)")
    for c in CFGS:
        pr(f"    {c}: ${(1 + results[c]).cumprod().iloc[-1]:.2f}")
    pr(f"    IVV B&H: ${(1 + ivv_s).cumprod().iloc[-1]:.2f}")

    # Interpretation
    pr(f"\n\n{'=' * 80}")
    pr(f"  INTERPRETATION")
    pr(f"{'=' * 80}")

    pr(f"\n  Q1: How much Sharpe does Faber add vs static?")
    pr(f"      {faber_sh:+.3f} ({pf['sh']:.3f} vs {ps['sh']:.3f})")
    if faber_sh > 0.1:
        pr(f"      → Faber is a MAJOR contributor. Trend-following drives most of the alpha.")
    elif faber_sh > 0.03:
        pr(f"      → Faber is a moderate contributor.")
    else:
        pr(f"      → Faber adds minimal Sharpe (but may improve drawdowns).")

    pr(f"\n  Q2: How much Sharpe does Harvey add vs Faber-only?")
    pr(f"      {harvey_marginal_sh:+.3f} ({pfh['sh']:.3f} vs {pf['sh']:.3f})")
    if harvey_marginal_sh > 0.05:
        pr(f"      → Harvey adds meaningful alpha beyond Faber. Worth the complexity.")
    elif harvey_marginal_sh > 0:
        pr(f"      → Harvey adds small but positive value. Marginal complexity trade-off.")
    else:
        pr(f"      → Harvey does NOT add Sharpe beyond Faber. Consider simplifying.")

    pr(f"\n  Q3: Is Harvey worth the complexity?")
    fh_dd = pfh["dd"]; f_dd = pf["dd"]
    if harvey_marginal_sh > 0 and fh_dd >= f_dd:
        pr(f"      YES — improves both Sharpe ({harvey_marginal_sh:+.3f}) and drawdown ({fh_dd - f_dd:+.1%})")
    elif harvey_marginal_sh > 0:
        pr(f"      MAYBE — improves Sharpe ({harvey_marginal_sh:+.3f}) but deepens drawdown ({fh_dd - f_dd:+.1%})")
    else:
        pr(f"      NO — Faber alone captures most of the system's value.")
        pr(f"      Faber Sharpe: {pf['sh']:.3f}, Faber+Harvey: {pfh['sh']:.3f}")

    rp = OUTPUT / "component_valueadd_report.txt"
    with open(rp, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Report saved: {rp}")


if __name__ == "__main__":
    main()
