"""Vol-scaled conviction MVO: Faber trend × conditioned vol as return proxy."""

import sys, os, warnings, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from dotenv import load_dotenv; load_dotenv()
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

import numpy as np, pandas as pd
from scipy.optimize import minimize

from regime.config import RegimeConfig
from regime.data import fetch_monthly_history
from regime.transform import transform_variables, get_valid_zscored
from regime.similarity import compute_distances

from experiments.kritzman_relevance.config import CONFIG
from experiments.kritzman_relevance.relevance_engine import (
    compute_indicator_covariance, compute_relevance,
    select_relevant_subsample, compute_relevance_weights,
)
from experiments.kritzman_relevance.conditioned_estimates import (
    conditioned_covariance_matrix, _nearest_pd,
)
from experiments.kritzman_relevance.allocation import (
    apply_faber_filter, distribute_pool,
)
from experiments.kritzman_relevance.backtest import (
    load_data, compute_faber_scores, load_prices,
)

OUTPUT = Path("experiments/kritzman_relevance/output")
ASSETS = CONFIG["assets"]
CASH_VOL = 0.001

TREND_MOD = {3: 1.0, 2: 0.5, 1: -0.5, 0: -0.5}


def build_mu(faber_scores, cond_cov, s_monthly):
    """Build vol-scaled conviction vector: mu_i = S_m * sigma_i * trend_mod."""
    mu = {}
    for a in ASSETS:
        if a == "cash":
            mu[a] = 0.0
            continue
        score = faber_scores.get(a, 0)
        mod = TREND_MOD.get(score, -0.5)
        if a in cond_cov.columns:
            sigma_i = np.sqrt(max(cond_cov.loc[a, a], 1e-8))
        else:
            sigma_i = 0.04  # fallback ~4% monthly
        mu[a] = s_monthly * sigma_i * mod
    return mu


def augment_cov(cov_df):
    """Add cash to covariance matrix."""
    assets = list(cov_df.columns) + (["cash"] if "cash" not in cov_df.columns else [])
    n = len(assets)
    aug = np.zeros((n, n))
    risky = [a for a in assets if a != "cash"]
    for i, a in enumerate(risky):
        for j, b in enumerate(risky):
            if a in cov_df.columns and b in cov_df.columns:
                aug[i, j] = cov_df.loc[a, b]
    ci = assets.index("cash")
    aug[ci, ci] = CASH_VOL ** 2
    return pd.DataFrame(aug, index=assets, columns=assets)


def run_mvo(mu_vec, sigma, lam, max_w):
    n = len(mu_vec)

    def neg_u(w):
        return -(w @ mu_vec - 0.5 * lam * w @ sigma @ w)

    def neg_u_jac(w):
        return -(mu_vec - lam * sigma @ w)

    ub = max_w if max_w else 1.0
    bounds = [(0, ub)] * n
    # Find cash index and set its lower bound to 2%
    # Cash is last in ASSETS order
    cash_idx = n - 1  # will be set correctly below
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    x0 = np.full(n, 1.0 / n)
    res = minimize(neg_u, x0, jac=neg_u_jac, bounds=bounds,
                   constraints=constraints, method="SLSQP",
                   options={"maxiter": 300, "ftol": 1e-12})
    if not res.success:
        return None
    w = np.maximum(res.x, 0)
    # Enforce cash floor
    return w


def run_backtest(lam, max_w, s_monthly, preloaded):
    z_lagged = preloaded["z_lagged"]
    asset_ret = preloaded["asset_ret"]
    asset_ret_fwd = preloaded["asset_ret_fwd"]
    prices_df = preloaded["prices_df"]
    bt_dates = preloaded["bt_dates"]
    config = preloaded["config"]

    exclude = CONFIG["exclude_recent_months"]
    top_pct = CONFIG["relevance_top_pct"]
    min_hist = CONFIG["min_history_months"]
    baseline_w = CONFIG["baseline_weights"]

    rets = {}
    wrows = []
    broken_trend_held = 0  # months where optimizer holds a 0-1/3 asset
    total_months = 0
    ex_ante_vols = []
    realized_rets_for_vol = []

    for dt in bt_dates:
        if dt not in z_lagged.index:
            continue
        avail = [a for a in ASSETS if a in asset_ret.columns and pd.notna(asset_ret.loc[dt, a])]
        if len([a for a in avail if a != "cash"]) < 2:
            continue
        actual = {a: asset_ret.loc[dt, a] for a in avail}

        faber_scores = compute_faber_scores(prices_df, dt)

        cutoff = dt - pd.DateOffset(months=exclude)
        candidates = z_lagged[z_lagged.index <= cutoff]

        if len(candidates) < min_hist:
            rets[dt] = sum(baseline_w.get(a, 0) * actual.get(a, 0) for a in avail)
            continue

        current_z = z_lagged.loc[dt].values
        omega, omega_inv, mean_vec = compute_indicator_covariance(
            z_lagged, dt, CONFIG["covariance_regularization"])
        relevance = compute_relevance(current_z, candidates, omega_inv, mean_vec)
        rel_dates, rel_scores = select_relevant_subsample(relevance, top_pct)
        rel_weights = compute_relevance_weights(relevance, rel_scores)
        fwd_dates = pd.DatetimeIndex([d + pd.DateOffset(months=1) for d in rel_dates])
        fwd_w = pd.Series(rel_weights.values, index=fwd_dates)
        fwd_w = fwd_w[~fwd_w.index.duplicated(keep="first")]

        avail_risky = [a for a in avail if a != "cash" and a in asset_ret_fwd.columns]
        if not avail_risky:
            rets[dt] = actual.get("cash", 0)
            continue

        cond_cov = conditioned_covariance_matrix(
            asset_ret_fwd[avail_risky], fwd_w.index, fwd_w)

        # Build vol-scaled conviction
        mu_dict = build_mu(faber_scores, cond_cov, s_monthly)

        # Augment cov with cash
        cov_aug = _nearest_pd(augment_cov(cond_cov))

        asset_order = [a for a in ASSETS if a in avail]
        mu_vec = np.array([mu_dict.get(a, 0) for a in asset_order])
        sigma = cov_aug.reindex(index=asset_order, columns=asset_order).fillna(0).values

        w_opt = run_mvo(mu_vec, sigma, lam, max_w)
        if w_opt is None:
            rets[dt] = sum(baseline_w.get(a, 0) * actual.get(a, 0) for a in avail)
            continue

        w_dict = {a: float(w_opt[i]) for i, a in enumerate(asset_order)}
        ret = sum(w_dict.get(a, 0) * actual.get(a, 0) for a in avail)
        rets[dt] = ret
        wrows.append({"date": dt, **w_dict})

        # Ex-ante vol
        w_arr = np.array([w_dict.get(a, 0) for a in asset_order])
        ex_ante_var = w_arr @ sigma @ w_arr
        ex_ante_vols.append(np.sqrt(max(ex_ante_var, 0)) * np.sqrt(12))
        realized_rets_for_vol.append(ret)

        # Track broken-trend holdings
        total_months += 1
        for a in avail:
            if a == "cash":
                continue
            if faber_scores.get(a, 0) <= 1 and w_dict.get(a, 0) > 0.02:
                broken_trend_held += 1
                break

    if not rets:
        return None

    s = pd.Series(rets).sort_index()
    ar = s.mean() * 12; av = s.std() * np.sqrt(12)
    sh = ar / av if av > 0 else 0
    neg = s[s < 0]; ds = neg.std() * np.sqrt(12) if len(neg) > 10 else av
    so = ar / ds if ds > 0 else 0
    cum = (1 + s).cumprod()
    dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
    final = cum.iloc[-1]
    wdf = pd.DataFrame(wrows).set_index("date") if wrows else pd.DataFrame()

    return {
        "ar": ar, "av": av, "sh": sh, "so": so, "dd": dd, "final": final,
        "series": s, "weights": wdf,
        "broken_trend_pct": broken_trend_held / total_months if total_months > 0 else 0,
        "ex_ante_vols": ex_ante_vols,
        "realized_rets": realized_rets_for_vol,
    }


def main():
    lines = []
    def pr(s=""): print(s); lines.append(s)

    config_obj, raw_macro, z_data, asset_ret = load_data()
    z_cols = [c for c in z_data.columns if c.endswith("_z")]
    z_lagged = z_data[z_cols].dropna().shift(1).dropna()
    asset_ret_fwd = asset_ret.shift(-1)
    prices_df = load_prices()

    bt_start = pd.Timestamp(CONFIG["backtest_start"])
    bt_end = pd.Timestamp(CONFIG["backtest_end"])
    bt_dates = z_lagged.index[(z_lagged.index >= bt_start) & (z_lagged.index <= bt_end)]
    bt_dates = bt_dates[bt_dates.isin(asset_ret.index)]

    preloaded = {
        "config": config_obj, "z_lagged": z_lagged, "asset_ret": asset_ret,
        "asset_ret_fwd": asset_ret_fwd, "prices_df": prices_df, "bt_dates": bt_dates,
    }

    pr("=" * 80)
    pr("  VOL-SCALED CONVICTION MVO")
    pr("=" * 80)
    pr(f"\nBacktest: {bt_dates.min().date()} to {bt_dates.max().date()} ({len(bt_dates)} months)")
    pr(f"μ_i = S_monthly × σ_i_conditioned × trend_modifier")
    pr(f"S_annual = 0.85 → S_monthly = {0.85/np.sqrt(12):.3f}")

    S_BASE = 0.85 / np.sqrt(12)
    caps = [("Unconstrained", None), ("60% Cap", 0.60), ("40% Cap", 0.40)]

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1: Lambda sweep
    # ══════════════════════════════════════════════════════════════════════════

    pr(f"\n\n{'='*80}")
    pr(f"  PHASE 1: LAMBDA SWEEP (S=0.85 annual)")
    pr(f"{'='*80}")

    lambdas = list(range(1, 51))
    sweep = {}
    best_lam = {}

    for cap_name, max_w in caps:
        pr(f"\n  {cap_name}:")
        pr(f"  {'λ':>4} {'Vol':>7} {'Return':>8} {'Sharpe':>8} {'MaxDD':>8}")
        pr(f"  {'-'*4} {'-'*7} {'-'*8} {'-'*8} {'-'*8}")

        best, best_vd = 10, 999
        for lam in lambdas:
            r = run_backtest(lam, max_w, S_BASE, preloaded)
            if r is None: continue
            sweep[(cap_name, lam)] = r
            pr(f"  {lam:>4} {r['av']:>6.1%} {r['ar']:>7.1%} {r['sh']:>8.3f} {r['dd']:>7.1%}")
            vd = abs(r["av"] - 0.10)
            if vd < best_vd:
                best_vd = vd; best = lam

        best_lam[cap_name] = best
        br = sweep.get((cap_name, best))
        pr(f"\n  → Best λ for ~10% vol: {best} (vol={br['av']:.1%}, Sharpe={br['sh']:.3f})")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2: S sensitivity
    # ══════════════════════════════════════════════════════════════════════════

    pr(f"\n\n{'='*80}")
    pr(f"  PHASE 2: S PARAMETER SENSITIVITY")
    pr(f"{'='*80}")

    # Use best cap (highest Sharpe at best lambda)
    best_cap_name = max(caps, key=lambda c: sweep.get((c[0], best_lam[c[0]]), {}).get("sh", 0))[0]
    best_cap_w = {"Unconstrained": None, "60% Cap": 0.60, "40% Cap": 0.40}[best_cap_name]
    bl = best_lam[best_cap_name]

    pr(f"\n  Using {best_cap_name} at λ={bl}")
    pr(f"\n  {'S_ann':>7} {'S_mon':>7} {'Return':>8} {'Vol':>7} {'Sharpe':>8} {'MaxDD':>8}")
    pr(f"  {'-'*7} {'-'*7} {'-'*8} {'-'*7} {'-'*8} {'-'*8}")

    for s_ann in [0.50, 0.70, 0.85, 1.00, 1.20]:
        s_mon = s_ann / np.sqrt(12)
        r = run_backtest(bl, best_cap_w, s_mon, preloaded)
        if r is None: continue
        pr(f"  {s_ann:>7.2f} {s_mon:>7.3f} {r['ar']:>7.1%} {r['av']:>6.1%} {r['sh']:>8.3f} {r['dd']:>7.1%}")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 3: Full comparison
    # ══════════════════════════════════════════════════════════════════════════

    pr(f"\n\n{'='*80}")
    pr(f"  PHASE 3: FULL COMPARISON")
    pr(f"{'='*80}")

    # Run reference strategies
    baseline_w = CONFIG["baseline_weights"]
    exclude = CONFIG["exclude_recent_months"]
    min_hist = CONFIG["min_history_months"]

    ref_rets = {"Faber-Only": {}, "Harvey": {}, "IVV B&H": {}, "60/40": {}}

    for dt in bt_dates:
        if dt not in z_lagged.index: continue
        avail = [a for a in ASSETS if a in asset_ret.columns and pd.notna(asset_ret.loc[dt, a])]
        if len([a for a in avail if a != "cash"]) < 2: continue
        actual = {a: asset_ret.loc[dt, a] for a in avail}

        fs = compute_faber_scores(prices_df, dt)
        fw, pool, eligible = apply_faber_filter(fs, baseline_w, CONFIG["partial_multiplier"])

        wfo = dict(fw); wfo["cash"] = wfo.get("cash", 0) + pool
        ref_rets["Faber-Only"][dt] = sum(wfo.get(a, 0) * actual.get(a, 0) for a in avail)
        ref_rets["IVV B&H"][dt] = actual.get("IVV", 0)
        ref_rets["60/40"][dt] = 0.60 * actual.get("IVV", 0) + 0.40 * actual.get("VGLT", 0) if "VGLT" in actual else 0.60 * actual.get("IVV", 0)

        cutoff = dt - pd.DateOffset(months=exclude)
        cand = z_lagged[z_lagged.index <= cutoff]
        if len(cand) < min_hist:
            ref_rets["Harvey"][dt] = sum(baseline_w.get(a, 0) * actual.get(a, 0) for a in avail)
            continue
        try:
            sim = compute_distances(z_lagged, dt, config_obj)
            her = {}
            for a in avail:
                if a not in asset_ret_fwd.columns: her[a] = 0; continue
                r = [asset_ret_fwd.loc[d, a] for d in sim.similar_dates
                     if d in asset_ret_fwd.index and pd.notna(asset_ret_fwd.loc[d, a])]
                her[a] = np.mean(r) if r else 0
            hs = {a: her.get(a, 0) for a in eligible if a != "cash" and her.get(a, 0) > 0}
            wh = distribute_pool(fw, pool, hs, eligible, 0.40)
            ref_rets["Harvey"][dt] = sum(wh.get(a, 0) * actual.get(a, 0) for a in avail)
        except ValueError:
            ref_rets["Harvey"][dt] = ref_rets["Faber-Only"][dt]

    def perf(d):
        s = pd.Series(d).sort_index()
        ar = s.mean() * 12; av = s.std() * np.sqrt(12)
        sh = ar / av if av > 0 else 0
        neg = s[s < 0]; ds = neg.std() * np.sqrt(12) if len(neg) > 10 else av
        so = ar / ds if ds > 0 else 0
        cum = (1 + s).cumprod(); dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
        return {"ar": ar, "av": av, "sh": sh, "so": so, "dd": dd, "final": cum.iloc[-1], "series": s}

    all_p = {name: perf(d) for name, d in ref_rets.items()}

    pr(f"\n  {'Strategy':<30} {'Return':>8} {'Vol':>7} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'Terminal':>10}")
    pr(f"  {'-'*30} {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")

    for name in ["Faber-Only", "Harvey", "IVV B&H", "60/40"]:
        p = all_p[name]
        pr(f"  {name:<30} {p['ar']:>7.1%} {p['av']:>6.1%} {p['sh']:>8.3f} {p['so']:>8.3f} {p['dd']:>7.1%} ${p['final']:>9.2f}")

    for cap_name, max_w in caps:
        lam = best_lam[cap_name]
        r = sweep.get((cap_name, lam))
        if r is None: continue
        label = f"MVO {cap_name} (λ={lam})"
        pr(f"  {label:<30} {r['ar']:>7.1%} {r['av']:>6.1%} {r['sh']:>8.3f} {r['so']:>8.3f} {r['dd']:>7.1%} ${r['final']:>9.2f}")
        all_p[label] = r

    # Crisis
    pr(f"\n\nCRISIS ANALYSIS")
    pr("-" * 70)
    for cname, cs, ce in [("GFC", "2008-09", "2009-03"), ("COVID", "2020-02", "2020-04"), ("2022 Bear", "2022-01", "2022-10")]:
        pr(f"\n  {cname}:")
        pr(f"  {'Strategy':<30} {'Return':>10}")
        pr(f"  {'-'*30} {'-'*10}")
        for name, p in all_p.items():
            sr = p.get("series")
            if sr is None: continue
            c = sr[(sr.index >= pd.Timestamp(cs)) & (sr.index <= pd.Timestamp(ce))]
            if len(c) > 0:
                pr(f"  {name:<30} {(1+c).prod()-1:>+9.1%}")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 4: Diagnostics
    # ══════════════════════════════════════════════════════════════════════════

    pr(f"\n\n{'='*80}")
    pr(f"  PHASE 4: DIAGNOSTICS")
    pr(f"{'='*80}")

    # Pick best MVO config
    best_mvo_key = max(
        [(cn, best_lam[cn]) for cn, _ in caps if (cn, best_lam[cn]) in sweep],
        key=lambda k: sweep[k]["sh"]
    )
    br = sweep[best_mvo_key]
    pr(f"\n  Best MVO: {best_mvo_key[0]} at λ={best_mvo_key[1]}")

    if len(br.get("weights", pd.DataFrame())) > 0:
        wdf = br["weights"]

        pr(f"\n  Weight statistics:")
        pr(f"  {'Asset':<8} {'Mean':>7} {'Std':>7} {'Min':>7} {'Max':>7} {'%Zero':>7}")
        pr(f"  {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
        for a in ASSETS:
            if a not in wdf.columns: continue
            c = wdf[a]
            pr(f"  {a:<8} {c.mean():>6.0%} {c.std():>6.0%} {c.min():>6.0%} {c.max():>6.0%} {(c < 0.005).mean():>6.0%}")

        risky_cols = [a for a in ASSETS if a != "cash" and a in wdf.columns]
        if risky_cols:
            max_any = wdf[risky_cols].max().max()
            avg_max = wdf[risky_cols].max(axis=1).mean()
            pr(f"\n  Max weight ever (single asset): {max_any:.0%}")
            pr(f"  Avg max weight across months:   {avg_max:.0%}")

        diffs = wdf.diff().abs().sum(axis=1).dropna() / 2
        pr(f"\n  Monthly turnover:")
        pr(f"    Mean:   {diffs.mean():.1%}")
        pr(f"    Median: {diffs.median():.1%}")
        pr(f"    Max:    {diffs.max():.1%}")

    # Broken trend holdings
    pr(f"\n  Broken-trend asset held (0-1/3 Faber, >2% weight):")
    pr(f"    {br['broken_trend_pct']*100:.1f}% of months")
    if br["broken_trend_pct"] > 0.05:
        pr(f"    → Covariance matrix IS overriding trend signal for diversification")
    else:
        pr(f"    → Trend signal dominates — covariance adds little beyond trend")

    # Ex-ante vs realized vol
    if br["ex_ante_vols"]:
        ea = np.array(br["ex_ante_vols"])
        # Realized: rolling 12-month vol of returns
        ret_s = br["series"]
        rv = ret_s.rolling(12).std() * np.sqrt(12)
        rv = rv.dropna()
        # Align
        ea_s = pd.Series(ea[:len(rv)], index=rv.index[:len(ea)])
        common = ea_s.index.intersection(rv.index)
        if len(common) > 12:
            ea_c = ea_s.reindex(common)
            rv_c = rv.reindex(common)
            corr = ea_c.corr(rv_c)
            pr(f"\n  Ex-ante vs realized portfolio vol:")
            pr(f"    Correlation:          {corr:.3f}")
            pr(f"    Mean ex-ante:         {ea_c.mean():.1%}")
            pr(f"    Mean realized (12m):  {rv_c.mean():.1%}")
            pr(f"    Mean abs error:       {(ea_c - rv_c).abs().mean():.1%}")

    # Save
    rp = OUTPUT / "vol_scaled_mvo_report.txt"
    with open(rp, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Report saved: {rp}")


if __name__ == "__main__":
    main()
