"""Full backtest harness: Kritzman relevance-weighted allocation.

Runs 4 strategies behind the same Faber gate:
1. Harvey-Mulliner (Euclidean, equal-weight) — baseline
2. Kritzman relevance (inverse-vol) — direct comparison
3. Kritzman relevance (mean-variance)
4. Kritzman relevance (risk parity)

All at 1x (no leverage).
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from dotenv import load_dotenv; load_dotenv()

import warnings; warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

import numpy as np
import pandas as pd

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
    conditioned_expected_returns, conditioned_covariance_matrix,
    conditioned_volatilities, conditioned_correlations,
)
from experiments.kritzman_relevance.allocation import (
    allocate_inverse_vol, allocate_mean_variance, allocate_risk_parity,
    apply_faber_filter, distribute_pool,
)

OUTPUT = Path("experiments/kritzman_relevance/output")
OUTPUT.mkdir(parents=True, exist_ok=True)

STRATEGIES = ["Faber-Only", "Harvey", "Kritzman-InvVol", "Kritzman-MV", "Kritzman-RP", "IVV B&H", "60/40"]


def load_data():
    """Load all data needed for the backtest."""
    config = RegimeConfig()
    raw_macro = fetch_monthly_history(config)
    transformed = transform_variables(raw_macro, config)
    z_data = get_valid_zscored(transformed, config)

    asset_ret = pd.read_parquet("data/macro/roth_asset_returns.parquet")
    for c in list(asset_ret.columns):
        if c not in CONFIG["assets"]:
            asset_ret = asset_ret.drop(columns=[c])

    return config, raw_macro, z_data, asset_ret


def compute_faber_scores(prices_df: pd.DataFrame, as_of_date: pd.Timestamp) -> dict:
    """Compute Faber trend strength (0-3) for each asset using monthly SMAs."""
    monthly = prices_df[prices_df.index < as_of_date].resample("MS").last()
    scores = {}
    for asset in prices_df.columns:
        if asset == "cash":
            continue
        score = 0
        for period in CONFIG["sma_periods"]:
            if len(monthly) < period:
                continue
            sma = monthly[asset].iloc[-period:].mean()
            current = monthly[asset].iloc[-1]
            if pd.notna(sma) and pd.notna(current) and current > sma:
                score += 1
        scores[asset] = score
    return scores


def load_prices():
    """Load daily ETF prices for Faber SMA computation."""
    import yfinance as yf
    ep = {}
    for our, ticker in [("IVV", "SPY"), ("QQQ", "QQQ"), ("VGLT", "TLT"), ("IAU", "GLD"), ("DBC", "DBC")]:
        d = yf.download(ticker, start="1998-01-01", progress=False)
        if d is not None and not d.empty:
            p = d["Close"]
            if hasattr(p, "columns"): p = p.iloc[:, 0]
            p.index = pd.to_datetime(p.index).tz_localize(None)
            ep[our] = p
    return pd.DataFrame(ep).sort_index()


def run_backtest():
    """Run full monthly backtest."""
    print("Loading data...")
    config, raw_macro, z_data, asset_ret = load_data()

    # z_data has columns like sp500_z, yield_curve_z, etc.
    z_cols = [c for c in z_data.columns if c.endswith("_z")]
    z_clean = z_data[z_cols].dropna()

    # Shift z-scores by 1 month for signal alignment (month T uses T-1 z-scores)
    z_lagged = z_clean.shift(1).dropna()

    # Forward returns: month AFTER the indicator date
    asset_ret_fwd = asset_ret.shift(-1)

    print("Loading prices for Faber...")
    prices_df = load_prices()

    # Determine backtest dates
    bt_start = pd.Timestamp(CONFIG["backtest_start"])
    bt_end = pd.Timestamp(CONFIG["backtest_end"])
    bt_dates = z_lagged.index[(z_lagged.index >= bt_start) & (z_lagged.index <= bt_end)]
    bt_dates = bt_dates[bt_dates.isin(asset_ret.index)]

    print(f"Backtest: {bt_dates.min().date()} to {bt_dates.max().date()} ({len(bt_dates)} months)")

    # Results storage
    results = {s: {} for s in STRATEGIES}
    weight_log = {s: [] for s in STRATEGIES}
    diagnostics = {
        "n_relevant": [],
        "cond_number": [],
        "corr_ivv_vglt": [],
        "relevance_spread": [],
    }

    exclude_months = CONFIG["exclude_recent_months"]
    top_pct = CONFIG["relevance_top_pct"]
    min_hist = CONFIG["min_history_months"]
    baseline_w = CONFIG["baseline_weights"]
    max_single = CONFIG["max_single_asset"]

    for dt in bt_dates:
        # ── Signal alignment assertions ───────────────────────────────────────
        # z_lagged[dt] uses macro data from month dt-1 (already shifted)
        assert dt in z_lagged.index, f"{dt} not in z_lagged"

        avail = [a for a in CONFIG["assets"] if a in asset_ret.columns and pd.notna(asset_ret.loc[dt, a])]
        avail_risky = [a for a in avail if a != "cash"]
        if len(avail_risky) < 2:
            continue

        actual = {a: asset_ret.loc[dt, a] for a in avail}

        # ── Faber filter (same for all strategies) ────────────────────────────
        faber_scores = compute_faber_scores(prices_df, dt)
        faber_w, pool, eligible = apply_faber_filter(faber_scores, baseline_w, CONFIG["partial_multiplier"])

        # ── Faber-Only: pool goes to cash ────────────────────────────────────
        w_faber_only = dict(faber_w)
        w_faber_only["cash"] = w_faber_only.get("cash", 0) + pool
        results["Faber-Only"][dt] = sum(w_faber_only.get(a, 0) * actual.get(a, 0) for a in avail)

        # ── Benchmarks ────────────────────────────────────────────────────────
        results["IVV B&H"][dt] = actual.get("IVV", 0)
        r_6040 = 0.60 * actual.get("IVV", 0) + 0.40 * actual.get("VGLT", 0) if "VGLT" in actual else actual.get("IVV", 0) * 0.60
        results["60/40"][dt] = r_6040

        # ── Candidate pool for similarity (exclude recent 36 months) ──────────
        cutoff = dt - pd.DateOffset(months=exclude_months)
        candidates = z_lagged[z_lagged.index <= cutoff]

        if len(candidates) < min_hist:
            # Not enough history — use static weights for macro strategies
            for strat in ["Harvey", "Kritzman-InvVol", "Kritzman-MV", "Kritzman-RP"]:
                results[strat][dt] = sum(baseline_w.get(a, 0) * actual.get(a, 0) for a in avail)
            continue

        # ── Harvey-Mulliner (Euclidean, equal-weight) ─────────────────────────
        try:
            sim = compute_distances(z_lagged, dt, config)
            harvey_er = {}
            for a in avail:
                if a not in asset_ret_fwd.columns:
                    harvey_er[a] = 0; continue
                rets = [asset_ret_fwd.loc[d, a] for d in sim.similar_dates
                        if d in asset_ret_fwd.index and pd.notna(asset_ret_fwd.loc[d, a])]
                harvey_er[a] = np.mean(rets) if rets else 0

            # Allocate: Faber gate + Harvey pool direction (inverse-vol with trailing vol)
            harvey_scores = {}
            for a in eligible:
                if a == "cash": continue
                er = harvey_er.get(a, 0)
                if er > 0:
                    harvey_scores[a] = er  # simplified: just use ER as score
            w_harvey = distribute_pool(faber_w, pool, harvey_scores, eligible, max_single)
        except ValueError:
            w_harvey = dict(faber_w)
            w_harvey["cash"] = w_harvey.get("cash", 0) + pool

        results["Harvey"][dt] = sum(w_harvey.get(a, 0) * actual.get(a, 0) for a in avail)
        weight_log["Harvey"].append({"date": dt, **w_harvey})

        # ── Kritzman relevance engine ─────────────────────────────────────────
        current_z = z_lagged.loc[dt].values

        # Assert no look-ahead: covariance uses data strictly before dt
        omega, omega_inv, mean_vec = compute_indicator_covariance(
            z_lagged, dt, CONFIG["covariance_regularization"]
        )

        # Compute relevance for all candidates
        relevance = compute_relevance(current_z, candidates, omega_inv, mean_vec)

        # Select top 20%
        rel_dates, rel_scores = select_relevant_subsample(relevance, top_pct)
        rel_weights = compute_relevance_weights(relevance, rel_scores)

        # Forward returns from month AFTER each relevant date
        fwd_dates = pd.DatetimeIndex([d + pd.DateOffset(months=1) for d in rel_dates])

        # Assert signal alignment: forward dates must be after indicator dates
        assert all(fd > rd for fd, rd in zip(fwd_dates, rel_dates)), "Forward date <= indicator date"

        # Map weights to forward dates
        fwd_weights = pd.Series(rel_weights.values, index=fwd_dates)
        fwd_weights = fwd_weights[~fwd_weights.index.duplicated(keep="first")]

        # Conditioned estimates
        avail_cols = [a for a in avail if a in asset_ret_fwd.columns]
        cond_er = conditioned_expected_returns(asset_ret_fwd[avail_cols], fwd_weights.index, fwd_weights)
        cond_cov = conditioned_covariance_matrix(asset_ret_fwd[avail_cols], fwd_weights.index, fwd_weights, cond_er)
        cond_vol = conditioned_volatilities(cond_cov)
        cond_corr = conditioned_correlations(cond_cov)

        # Diagnostics
        diagnostics["n_relevant"].append(len(rel_dates))
        diagnostics["cond_number"].append(np.linalg.cond(omega))
        if "IVV" in cond_corr.columns and "VGLT" in cond_corr.columns:
            diagnostics["corr_ivv_vglt"].append(cond_corr.loc["IVV", "VGLT"])
        else:
            diagnostics["corr_ivv_vglt"].append(np.nan)
        diagnostics["relevance_spread"].append(float(rel_scores.max() - rel_scores.min()))

        # ── Kritzman Inverse-Vol ──────────────────────────────────────────────
        kr_iv_alloc = allocate_inverse_vol(cond_er, cond_vol, eligible, max_single)
        kr_iv_scores = {a: cond_er.get(a, 0) / max(cond_vol.get(a, 0.15), 0.01)
                        for a in eligible if a != "cash" and cond_er.get(a, 0) > 0}
        w_kr_iv = distribute_pool(faber_w, pool, kr_iv_scores, eligible, max_single)
        results["Kritzman-InvVol"][dt] = sum(w_kr_iv.get(a, 0) * actual.get(a, 0) for a in avail)
        weight_log["Kritzman-InvVol"].append({"date": dt, **w_kr_iv})

        # ── Kritzman Mean-Variance ────────────────────────────────────────────
        # For MV: compute unconstrained optimal weights, then use as scores for pool
        mv_raw = allocate_mean_variance(cond_er, cond_cov, eligible, risk_aversion=2.0, max_single=max_single)
        mv_scores = {a: max(mv_raw.get(a, 0), 0) for a in eligible if a != "cash"}
        w_kr_mv = distribute_pool(faber_w, pool, mv_scores, eligible, max_single)
        results["Kritzman-MV"][dt] = sum(w_kr_mv.get(a, 0) * actual.get(a, 0) for a in avail)
        weight_log["Kritzman-MV"].append({"date": dt, **w_kr_mv})

        # ── Kritzman Risk Parity ──────────────────────────────────────────────
        rp_raw = allocate_risk_parity(cond_cov, eligible, max_single)
        rp_scores = {a: max(rp_raw.get(a, 0), 0) for a in eligible if a != "cash"}
        w_kr_rp = distribute_pool(faber_w, pool, rp_scores, eligible, max_single)
        results["Kritzman-RP"][dt] = sum(w_kr_rp.get(a, 0) * actual.get(a, 0) for a in avail)
        weight_log["Kritzman-RP"].append({"date": dt, **w_kr_rp})

    # Convert to series
    ret_series = {s: pd.Series(d).sort_index() for s, d in results.items()}
    return ret_series, weight_log, diagnostics, bt_dates


def print_report(ret_series, weight_log, diagnostics, bt_dates):
    """Print performance comparison."""
    lines = []
    def pr(s=""): print(s); lines.append(s)

    pr("=" * 80)
    pr("  COMPONENT VALUE-ADD: KRITZMAN RELEVANCE vs HARVEY-MULLINER")
    pr("=" * 80)
    pr(f"\nBacktest: {bt_dates.min().date()} to {bt_dates.max().date()} ({len(bt_dates)} months)")
    pr(f"Rebalance: Monthly | Leverage: None (1x) | Faber: 6/10/12 SMA")
    pr(f"Universe: {', '.join(CONFIG['assets'])}")

    def perf(s):
        ar = s.mean() * 12; av = s.std() * np.sqrt(12)
        sh = ar / av if av > 0 else 0
        neg = s[s < 0]; ds = neg.std() * np.sqrt(12) if len(neg) > 10 else av
        so = ar / ds if ds > 0 else 0
        cum = (1 + s).cumprod(); dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
        cal = ar / abs(dd) if dd != 0 else 0; final = cum.iloc[-1]
        return {"ar": ar, "av": av, "sh": sh, "so": so, "dd": dd, "cal": cal, "final": final}

    perfs = {s: perf(ret_series[s]) for s in STRATEGIES if len(ret_series[s]) > 12}

    pr(f"\n\nPERFORMANCE (monthly, 1x, behind Faber gate)")
    pr("-" * 85)
    pr(f"  {'Strategy':<20} {'Return':>8} {'Vol':>7} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'Terminal':>10}")
    pr(f"  {'-'*20} {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    for s in STRATEGIES:
        if s not in perfs: continue
        p = perfs[s]
        pr(f"  {s:<20} {p['ar']:>7.1%} {p['av']:>6.1%} {p['sh']:>8.3f} {p['so']:>8.3f} {p['dd']:>7.1%} ${p['final']:>9.2f}")

    # Value-add vs Harvey
    pr(f"\n\nVALUE-ADD vs HARVEY-MULLINER")
    pr("-" * 60)
    ph = perfs.get("Harvey", {})
    pr(f"  {'Strategy':<20} {'ΔSharpe':>10} {'ΔReturn':>10} {'ΔMaxDD':>10}")
    pr(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
    for s in STRATEGIES:
        if s == "Harvey" or s not in perfs: continue
        p = perfs[s]
        pr(f"  {s:<20} {p['sh']-ph.get('sh',0):>+10.3f} {(p['ar']-ph.get('ar',0))*100:>+9.1f}% {(p['dd']-ph.get('dd',0))*100:>+9.1f}%")

    # Crisis analysis
    pr(f"\n\nCRISIS ANALYSIS")
    pr("-" * 70)
    for cname, cs, ce in [("GFC", "2008-09", "2009-03"), ("COVID", "2020-02", "2020-04"), ("2022 Bear", "2022-01", "2022-10")]:
        pr(f"\n  {cname}:")
        pr(f"  {'Strategy':<20} {'Return':>10}")
        pr(f"  {'-'*20} {'-'*10}")
        for s in STRATEGIES:
            sr = ret_series.get(s)
            if sr is None: continue
            c = sr[(sr.index >= pd.Timestamp(cs)) & (sr.index <= pd.Timestamp(ce))]
            if len(c) > 0:
                pr(f"  {s:<20} {(1+c).prod()-1:>+9.1%}")

    # Monthly return correlation between strategies
    pr(f"\n\nMONTHLY RETURN CORRELATION")
    pr("-" * 50)
    ret_df = pd.DataFrame(ret_series)
    corr = ret_df.corr()
    header = f"  {'':>20}"
    for s in STRATEGIES:
        header += f" {s[:12]:>13}"
    pr(header)
    for s1 in STRATEGIES:
        row = f"  {s1:<20}"
        for s2 in STRATEGIES:
            if s1 in corr.index and s2 in corr.columns:
                row += f" {corr.loc[s1, s2]:>13.3f}"
            else:
                row += f" {'N/A':>13}"
        pr(row)

    # Macro engine alpha
    pf = perfs.get("Faber-Only", {})
    if pf:
        pr(f"\n\nMACRO ENGINE ALPHA (Sharpe over Faber-only {pf.get('sh',0):.3f})")
        pr("-" * 50)
        for s in ["Harvey", "Kritzman-InvVol", "Kritzman-MV", "Kritzman-RP"]:
            if s in perfs:
                pr(f"  {s:<20} {perfs[s]['sh'] - pf['sh']:>+.3f}")

    # Diagnostics
    pr(f"\n\nDIAGNOSTICS")
    pr("-" * 50)
    if diagnostics["n_relevant"]:
        pr(f"  Avg relevant months per period:  {np.mean(diagnostics['n_relevant']):.0f}")
        pr(f"  Avg Ω condition number:          {np.mean(diagnostics['cond_number']):.1f}")
        corr_vals = [v for v in diagnostics["corr_ivv_vglt"] if not np.isnan(v)]
        if corr_vals:
            pr(f"  Avg conditioned IVV-VGLT corr:   {np.mean(corr_vals):.3f}")
            pr(f"  Std conditioned IVV-VGLT corr:   {np.std(corr_vals):.3f}")
            pr(f"  Range:                           [{min(corr_vals):.3f}, {max(corr_vals):.3f}]")
        pr(f"  Avg relevance spread:            {np.mean(diagnostics['relevance_spread']):.2f}")

    # Save
    report_path = OUTPUT / "backtest_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Report saved: {report_path}")

    # Save monthly results
    ret_df.to_parquet(OUTPUT / "monthly_returns.parquet")
    print(f"  Monthly returns saved: {OUTPUT / 'monthly_returns.parquet'}")

    return lines


if __name__ == "__main__":
    ret_series, weight_log, diagnostics, bt_dates = run_backtest()
    print_report(ret_series, weight_log, diagnostics, bt_dates)
