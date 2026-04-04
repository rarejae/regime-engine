"""Faber signal + Kritzman covariance mean-variance synthesis.

Uses Faber trend scores as conviction-based expected returns and
Kritzman's relevance-weighted conditioned covariance matrix as the
risk model in a mean-variance optimizer.
"""

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
    conditioned_covariance_matrix, conditioned_expected_returns,
    conditioned_volatilities, _nearest_pd,
)
from experiments.kritzman_relevance.allocation import apply_faber_filter, distribute_pool
from experiments.kritzman_relevance.backtest import load_data, compute_faber_scores, load_prices

OUTPUT = Path("experiments/kritzman_relevance/output")
OUTPUT.mkdir(parents=True, exist_ok=True)

ASSETS = CONFIG["assets"]  # IVV, QQQ, VGLT, IAU, DBC, cash

CONVICTION_MAPS = {
    "A (base)":   {3: 1.0,  2: 0.5,  1: -0.5, 0: -0.5},
    "B (wide)":   {3: 2.0,  2: 0.5,  1: -1.0, 0: -1.0},
    "C (narrow)": {3: 0.6,  2: 0.4,  1: -0.2, 0: -0.2},
    "D (no neg)": {3: 1.0,  2: 0.5,  1: 0.0,  0: 0.0},
    "E (binary)": {3: 1.0,  2: 1.0,  1: -1.0, 0: -1.0},
}

CASH_VOL = 0.001  # monthly vol for cash (~0.35% ann)


def build_conviction_vector(faber_scores: dict, conv_map: dict) -> dict:
    """Map Faber trend scores to conviction proxies."""
    mu = {}
    for a in ASSETS:
        if a == "cash":
            mu[a] = 0.0
        else:
            score = faber_scores.get(a, 0)
            mu[a] = conv_map.get(score, -0.5)
    return mu


def augment_cov_with_cash(cov_df: pd.DataFrame) -> pd.DataFrame:
    """Add cash row/col to covariance matrix with near-zero variance."""
    assets = list(cov_df.columns) + (["cash"] if "cash" not in cov_df.columns else [])
    n = len(assets)
    aug = np.zeros((n, n))
    risky = [a for a in assets if a != "cash"]
    for i, a in enumerate(risky):
        for j, b in enumerate(risky):
            if a in cov_df.columns and b in cov_df.columns:
                aug[i, j] = cov_df.loc[a, b]
    cash_idx = assets.index("cash")
    aug[cash_idx, cash_idx] = CASH_VOL ** 2
    return pd.DataFrame(aug, index=assets, columns=assets)


def run_mvo(mu_vec: np.ndarray, sigma: np.ndarray, lam: float,
            max_weight: float | None) -> np.ndarray | None:
    """Solve MVO: max w'mu - (lam/2) w'Sigma w, s.t. constraints."""
    n = len(mu_vec)

    def neg_utility(w):
        return -(w @ mu_vec - 0.5 * lam * w @ sigma @ w)

    def neg_utility_jac(w):
        return -(mu_vec - lam * sigma @ w)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    ub = max_weight if max_weight else 1.0
    bounds = [(0.0, ub)] * (n - 1) + [(0.02, ub)]  # last = cash, min 2%
    x0 = np.full(n, 1.0 / n)

    res = minimize(neg_utility, x0, jac=neg_utility_jac, bounds=bounds,
                   constraints=constraints, method="SLSQP",
                   options={"maxiter": 300, "ftol": 1e-12})
    if not res.success:
        return None
    return np.maximum(res.x, 0)


def run_full_backtest(lam: float, max_weight: float | None,
                      conv_map: dict, preloaded: dict) -> dict | None:
    """Run single backtest with given lambda, cap, and conviction map."""
    config = preloaded["config"]
    z_lagged = preloaded["z_lagged"]
    asset_ret = preloaded["asset_ret"]
    asset_ret_fwd = preloaded["asset_ret_fwd"]
    prices_df = preloaded["prices_df"]
    bt_dates = preloaded["bt_dates"]

    exclude_months = CONFIG["exclude_recent_months"]
    top_pct = CONFIG["relevance_top_pct"]
    min_hist = CONFIG["min_history_months"]
    baseline_w = CONFIG["baseline_weights"]

    returns = {}
    weight_rows = []

    for dt in bt_dates:
        if dt not in z_lagged.index:
            continue
        avail = [a for a in ASSETS if a in asset_ret.columns and pd.notna(asset_ret.loc[dt, a])]
        if len([a for a in avail if a != "cash"]) < 2:
            continue
        actual = {a: asset_ret.loc[dt, a] for a in avail}

        # Faber scores
        faber_scores = compute_faber_scores(prices_df, dt)

        # Conviction vector
        mu_dict = build_conviction_vector(faber_scores, conv_map)
        asset_order = [a for a in ASSETS if a in avail]
        mu_vec = np.array([mu_dict.get(a, 0) for a in asset_order])

        # Kritzman conditioned covariance
        cutoff = dt - pd.DateOffset(months=exclude_months)
        candidates = z_lagged[z_lagged.index <= cutoff]

        if len(candidates) < min_hist:
            # Not enough history — use baseline weights
            returns[dt] = sum(baseline_w.get(a, 0) * actual.get(a, 0) for a in avail)
            continue

        current_z = z_lagged.loc[dt].values
        omega, omega_inv, mean_vec = compute_indicator_covariance(
            z_lagged, dt, CONFIG["covariance_regularization"])
        relevance = compute_relevance(current_z, candidates, omega_inv, mean_vec)
        rel_dates, rel_scores = select_relevant_subsample(relevance, top_pct)
        rel_weights = compute_relevance_weights(relevance, rel_scores)

        fwd_dates = pd.DatetimeIndex([d + pd.DateOffset(months=1) for d in rel_dates])
        fwd_weights = pd.Series(rel_weights.values, index=fwd_dates)
        fwd_weights = fwd_weights[~fwd_weights.index.duplicated(keep="first")]

        avail_risky = [a for a in asset_order if a != "cash" and a in asset_ret_fwd.columns]
        if not avail_risky:
            returns[dt] = actual.get("cash", 0)
            continue

        cond_cov = conditioned_covariance_matrix(
            asset_ret_fwd[avail_risky], fwd_weights.index, fwd_weights)
        cov_aug = augment_cov_with_cash(cond_cov)
        cov_aug = _nearest_pd(cov_aug)

        # Align to asset_order
        sigma = cov_aug.reindex(index=asset_order, columns=asset_order).fillna(0).values

        # Optimize
        w_opt = run_mvo(mu_vec, sigma, lam, max_weight)
        if w_opt is None:
            returns[dt] = sum(baseline_w.get(a, 0) * actual.get(a, 0) for a in avail)
            continue

        w_dict = {a: float(w_opt[i]) for i, a in enumerate(asset_order)}
        ret = sum(w_dict.get(a, 0) * actual.get(a, 0) for a in avail)
        returns[dt] = ret
        weight_rows.append({"date": dt, **w_dict})

    if not returns:
        return None

    s = pd.Series(returns).sort_index()
    ar = s.mean() * 12; av = s.std() * np.sqrt(12)
    sh = ar / av if av > 0 else 0
    neg = s[s < 0]; ds = neg.std() * np.sqrt(12) if len(neg) > 10 else av
    so = ar / ds if ds > 0 else 0
    cum = (1 + s).cumprod()
    dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
    final = cum.iloc[-1]

    wdf = pd.DataFrame(weight_rows).set_index("date") if weight_rows else pd.DataFrame()

    return {"ar": ar, "av": av, "sh": sh, "so": so, "dd": dd, "final": final,
            "series": s, "weights": wdf, "n_months": len(s)}


def main():
    lines = []
    def pr(s=""): print(s); lines.append(s)

    # ── Preload data ──────────────────────────────────────────────────────────
    config, raw_macro, z_data, asset_ret = load_data()
    z_cols = [c for c in z_data.columns if c.endswith("_z")]
    z_lagged = z_data[z_cols].dropna().shift(1).dropna()
    asset_ret_fwd = asset_ret.shift(-1)
    prices_df = load_prices()

    bt_start = pd.Timestamp(CONFIG["backtest_start"])
    bt_end = pd.Timestamp(CONFIG["backtest_end"])
    bt_dates = z_lagged.index[(z_lagged.index >= bt_start) & (z_lagged.index <= bt_end)]
    bt_dates = bt_dates[bt_dates.isin(asset_ret.index)]

    preloaded = {
        "config": config, "z_lagged": z_lagged, "asset_ret": asset_ret,
        "asset_ret_fwd": asset_ret_fwd, "prices_df": prices_df, "bt_dates": bt_dates,
    }

    pr("=" * 80)
    pr("  FABER SIGNAL + KRITZMAN COVARIANCE MVO SYNTHESIS")
    pr("=" * 80)
    pr(f"\nBacktest: {bt_dates.min().date()} to {bt_dates.max().date()} ({len(bt_dates)} months)")

    # ── Phase 1: Lambda sweep ─────────────────────────────────────────────────
    pr(f"\n\n{'='*80}")
    pr(f"  PHASE 1: RISK AVERSION (λ) CALIBRATION SWEEP")
    pr(f"{'='*80}")

    conv_map_base = CONVICTION_MAPS["A (base)"]
    cap_configs = [("Unconstrained", None), ("60% Cap", 0.60), ("40% Cap", 0.40)]
    lambdas = np.arange(0.5, 20.5, 0.5)

    sweep_results = {}  # (cap_name, lam) -> metrics dict
    best_lambda = {}  # cap_name -> best lam (targeting ~10% vol)

    for cap_name, max_w in cap_configs:
        pr(f"\n  {cap_name}:")
        pr(f"  {'λ':>6} {'Vol':>7} {'Return':>8} {'Sharpe':>8} {'MaxDD':>8}")
        pr(f"  {'-'*6} {'-'*7} {'-'*8} {'-'*8} {'-'*8}")

        best_lam, best_vol_diff = 5.0, 999
        for lam in lambdas:
            res = run_full_backtest(lam, max_w, conv_map_base, preloaded)
            if res is None:
                continue
            sweep_results[(cap_name, lam)] = res
            pr(f"  {lam:>6.1f} {res['av']:>6.1%} {res['ar']:>7.1%} {res['sh']:>8.3f} {res['dd']:>7.1%}")

            vol_diff = abs(res["av"] - 0.10)
            if vol_diff < best_vol_diff:
                best_vol_diff = vol_diff
                best_lam = lam

        best_lambda[cap_name] = best_lam
        pr(f"\n  → Best λ for ~10% vol: {best_lam:.1f} (realized vol: {sweep_results.get((cap_name, best_lam), {}).get('av', 0):.1%})")

    # ── Phase 2: Run reference strategies ─────────────────────────────────────

    # Faber-only
    faber_rets = {}
    harvey_rets = {}
    krp_rets = {}
    ivv_rets = {}
    sixfour_rets = {}

    baseline_w = CONFIG["baseline_weights"]
    exclude_months = CONFIG["exclude_recent_months"]
    top_pct = CONFIG["relevance_top_pct"]
    min_hist = CONFIG["min_history_months"]

    for dt in bt_dates:
        if dt not in z_lagged.index: continue
        avail = [a for a in ASSETS if a in asset_ret.columns and pd.notna(asset_ret.loc[dt, a])]
        if len([a for a in avail if a != "cash"]) < 2: continue
        actual = {a: asset_ret.loc[dt, a] for a in avail}

        faber_scores = compute_faber_scores(prices_df, dt)
        faber_w, pool, eligible = apply_faber_filter(faber_scores, baseline_w, CONFIG["partial_multiplier"])

        # Faber-only
        w_fo = dict(faber_w); w_fo["cash"] = w_fo.get("cash", 0) + pool
        faber_rets[dt] = sum(w_fo.get(a, 0) * actual.get(a, 0) for a in avail)

        # IVV B&H
        ivv_rets[dt] = actual.get("IVV", 0)

        # 60/40
        sixfour_rets[dt] = 0.60 * actual.get("IVV", 0) + 0.40 * actual.get("VGLT", 0) if "VGLT" in actual else 0.60 * actual.get("IVV", 0)

        # Harvey
        cutoff = dt - pd.DateOffset(months=exclude_months)
        candidates = z_lagged[z_lagged.index <= cutoff]
        if len(candidates) < min_hist:
            harvey_rets[dt] = sum(baseline_w.get(a, 0) * actual.get(a, 0) for a in avail)
            krp_rets[dt] = harvey_rets[dt]
            continue
        try:
            sim = compute_distances(z_lagged, dt, config)
            harvey_er = {}
            for a in avail:
                if a not in asset_ret_fwd.columns: harvey_er[a] = 0; continue
                r = [asset_ret_fwd.loc[d, a] for d in sim.similar_dates
                     if d in asset_ret_fwd.index and pd.notna(asset_ret_fwd.loc[d, a])]
                harvey_er[a] = np.mean(r) if r else 0
            h_scores = {a: harvey_er.get(a, 0) for a in eligible if a != "cash" and harvey_er.get(a, 0) > 0}
            w_h = distribute_pool(faber_w, pool, h_scores, eligible, 0.40)
            harvey_rets[dt] = sum(w_h.get(a, 0) * actual.get(a, 0) for a in avail)
        except ValueError:
            harvey_rets[dt] = sum(w_fo.get(a, 0) * actual.get(a, 0) for a in avail)

        # Kritzman-RP
        current_z = z_lagged.loc[dt].values
        omega, omega_inv, mean_vec = compute_indicator_covariance(z_lagged, dt, CONFIG["covariance_regularization"])
        relevance = compute_relevance(current_z, candidates, omega_inv, mean_vec)
        rel_dates, rel_scores = select_relevant_subsample(relevance, top_pct)
        rel_weights = compute_relevance_weights(relevance, rel_scores)
        fwd_dates = pd.DatetimeIndex([d + pd.DateOffset(months=1) for d in rel_dates])
        fwd_w = pd.Series(rel_weights.values, index=fwd_dates)
        fwd_w = fwd_w[~fwd_w.index.duplicated(keep="first")]
        avail_risky = [a for a in avail if a != "cash" and a in asset_ret_fwd.columns]
        if avail_risky:
            from experiments.kritzman_relevance.allocation import allocate_risk_parity
            cc = conditioned_covariance_matrix(asset_ret_fwd[avail_risky], fwd_w.index, fwd_w)
            rp_raw = allocate_risk_parity(cc, eligible, 0.40)
            rp_scores = {a: max(rp_raw.get(a, 0), 0) for a in eligible if a != "cash"}
            w_rp = distribute_pool(faber_w, pool, rp_scores, eligible, 0.40)
            krp_rets[dt] = sum(w_rp.get(a, 0) * actual.get(a, 0) for a in avail)
        else:
            krp_rets[dt] = faber_rets[dt]

    def perf(d):
        s = pd.Series(d).sort_index()
        ar = s.mean() * 12; av = s.std() * np.sqrt(12)
        sh = ar / av if av > 0 else 0
        neg = s[s < 0]; ds = neg.std() * np.sqrt(12) if len(neg) > 10 else av
        so = ar / ds if ds > 0 else 0
        cum = (1 + s).cumprod(); dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
        return {"ar": ar, "av": av, "sh": sh, "so": so, "dd": dd, "final": cum.iloc[-1], "series": s}

    ref = {
        "Faber-Only": perf(faber_rets),
        "Harvey": perf(harvey_rets),
        "Kritzman-RP": perf(krp_rets),
        "IVV B&H": perf(ivv_rets),
        "60/40": perf(sixfour_rets),
    }

    # ── Phase 3: Comparison table ─────────────────────────────────────────────
    pr(f"\n\n{'='*80}")
    pr(f"  PHASE 2: COMPARISON AT OPTIMAL λ (~10% vol target)")
    pr(f"{'='*80}")

    pr(f"\n  {'Strategy':<28} {'Return':>8} {'Vol':>7} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'Terminal':>10}")
    pr(f"  {'-'*28} {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")

    all_rows = {}
    for name, p in ref.items():
        pr(f"  {name:<28} {p['ar']:>7.1%} {p['av']:>6.1%} {p['sh']:>8.3f} {p['so']:>8.3f} {p['dd']:>7.1%} ${p['final']:>9.2f}")
        all_rows[name] = p

    for cap_name, max_w in cap_configs:
        lam = best_lambda[cap_name]
        key = (cap_name, lam)
        if key not in sweep_results: continue
        r = sweep_results[key]
        label = f"MVO {cap_name} (λ={lam:.1f})"
        pr(f"  {label:<28} {r['ar']:>7.1%} {r['av']:>6.1%} {r['sh']:>8.3f} {r['so']:>8.3f} {r['dd']:>7.1%} ${r['final']:>9.2f}")
        all_rows[label] = r

    # ── Phase 4: Crisis analysis ──────────────────────────────────────────────
    pr(f"\n\nCRISIS ANALYSIS")
    pr("-" * 70)

    for cname, cs, ce in [("GFC", "2008-09", "2009-03"), ("COVID", "2020-02", "2020-04"), ("2022 Bear", "2022-01", "2022-10")]:
        pr(f"\n  {cname}:")
        pr(f"  {'Strategy':<28} {'Return':>10}")
        pr(f"  {'-'*28} {'-'*10}")
        for name, p in all_rows.items():
            sr = p.get("series")
            if sr is None: continue
            c = sr[(sr.index >= pd.Timestamp(cs)) & (sr.index <= pd.Timestamp(ce))]
            if len(c) > 0:
                pr(f"  {name:<28} {(1+c).prod()-1:>+9.1%}")

    # ── Phase 5: Conviction sensitivity ───────────────────────────────────────
    pr(f"\n\n{'='*80}")
    pr(f"  PHASE 3: CONVICTION SCORE SENSITIVITY")
    pr(f"{'='*80}")

    # Use best cap config (highest Sharpe among MVO rows)
    best_mvo = max(
        [(cap_name, best_lambda[cap_name], sweep_results.get((cap_name, best_lambda[cap_name])))
         for cap_name, _ in cap_configs if (cap_name, best_lambda[cap_name]) in sweep_results],
        key=lambda x: x[2]["sh"] if x[2] else 0
    )
    best_cap, best_lam_val, _ = best_mvo
    best_max_w = {"Unconstrained": None, "60% Cap": 0.60, "40% Cap": 0.40}[best_cap]

    pr(f"\n  Using {best_cap} with λ={best_lam_val:.1f}")
    pr(f"\n  {'Conviction Map':<16} {'Return':>8} {'Vol':>7} {'Sharpe':>8} {'MaxDD':>8}")
    pr(f"  {'-'*16} {'-'*8} {'-'*7} {'-'*8} {'-'*8}")

    for conv_name, conv_map in CONVICTION_MAPS.items():
        res = run_full_backtest(best_lam_val, best_max_w, conv_map, preloaded)
        if res is None: continue
        pr(f"  {conv_name:<16} {res['ar']:>7.1%} {res['av']:>6.1%} {res['sh']:>8.3f} {res['dd']:>7.1%}")

    # ── Phase 6: Diagnostics for best MVO ─────────────────────────────────────
    pr(f"\n\n{'='*80}")
    pr(f"  PHASE 4: DIAGNOSTICS ({best_cap}, λ={best_lam_val:.1f})")
    pr(f"{'='*80}")

    best_res = sweep_results.get((best_cap, best_lam_val))
    if best_res and len(best_res.get("weights", pd.DataFrame())) > 0:
        wdf = best_res["weights"]

        pr(f"\n  Weight statistics:")
        pr(f"  {'Asset':<8} {'Mean':>7} {'Std':>7} {'Min':>7} {'Max':>7} {'%Zero':>7}")
        pr(f"  {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
        for a in ASSETS:
            if a not in wdf.columns: continue
            c = wdf[a]
            pr(f"  {a:<8} {c.mean():>6.0%} {c.std():>6.0%} {c.min():>6.0%} {c.max():>6.0%} {(c < 0.005).mean():>6.0%}")

        # Max single weight
        max_any = wdf.drop(columns=["cash"], errors="ignore").max().max()
        avg_max = wdf.drop(columns=["cash"], errors="ignore").max(axis=1).mean()
        pct_over60 = (wdf.drop(columns=["cash"], errors="ignore").max(axis=1) > 0.60).mean()
        pr(f"\n  Max weight ever (single asset): {max_any:.0%}")
        pr(f"  Avg max weight across months:   {avg_max:.0%}")
        pr(f"  Months with >60% in one asset:  {pct_over60:.0%}")

        # Turnover
        diffs = wdf.diff().abs().sum(axis=1).dropna() / 2
        pr(f"\n  Monthly turnover:")
        pr(f"    Mean:   {diffs.mean():.1%}")
        pr(f"    Median: {diffs.median():.1%}")
        pr(f"    Max:    {diffs.max():.1%}")

    # Save report
    rp = OUTPUT / "faber_kritzman_mvo_report.txt"
    with open(rp, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Report saved: {rp}")


if __name__ == "__main__":
    main()
