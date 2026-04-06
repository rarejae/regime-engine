"""Pro-rata redistribution vs cash baseline experiment.

Three strategies, all using identical Faber multi-SMA trend filter:
1. Faber-Cash: freed capital goes to cash (existing baseline)
2. Faber-ProRata: freed capital redistributed pro-rata across eligible assets
3. Faber-CrossSectional: redistribution weighted by trend strength score

Plus Harvey (behind Faber), IVV B&H, and 60/40 benchmarks.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from dotenv import load_dotenv; load_dotenv()

import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from taa.data import load_monthly_prices, load_monthly_asset_returns
from taa.faber import compute_trend_scores

# ── Configuration ────────────────────────────────────────────────────────────

BASELINE = {"IVV": 0.45, "QQQ": 0.25, "VGLT": 0.05, "IAU": 0.10, "DBC": 0.05, "cash": 0.10}
ASSETS = list(BASELINE.keys())
SMA_PERIODS = [6, 10, 12]
MAX_SINGLE = 0.70  # pro-rata cap per asset
BACKTEST_START = "2002-01-01"

STRATEGIES = [
    "Faber-Cash", "Faber-ProRata", "Faber-CrossSectional",
    "Harvey (behind Faber)", "IVV B&H", "60/40",
]


# ── Faber filter (shared) ───────────────────────────────────────────────────

def apply_faber_filter(trend_scores: dict) -> tuple[dict, float, list]:
    """Apply Faber trend filter to baseline weights.

    Returns (weights_after_filter, pool_size, eligible_assets).
    Score 3/3 = full weight, 2/3 = 70% weight, 0-1/3 = exit.
    """
    w = {}
    pool = 0.0
    eligible = []

    for a, base_w in BASELINE.items():
        if a == "cash":
            w[a] = base_w
            continue
        score = trend_scores.get(a, 0)
        if score >= 3:
            w[a] = base_w
            eligible.append(a)
        elif score == 2:
            w[a] = base_w * 0.70
            pool += base_w * 0.30
            eligible.append(a)
        else:
            w[a] = 0.0
            pool += base_w

    return w, pool, eligible


# ── Capital redistribution strategies ────────────────────────────────────────

def redistribute_cash(w: dict, pool: float, eligible: list, trend_scores: dict) -> dict:
    """Faber-Cash: all freed capital goes to cash."""
    out = dict(w)
    out["cash"] = out.get("cash", 0) + pool
    return out


def redistribute_pro_rata(w: dict, pool: float, eligible: list, trend_scores: dict) -> dict:
    """Faber-ProRata: freed capital redistributed proportionally across eligible assets.

    Redistribution is proportional to each eligible asset's current allocated weight.
    No single asset can exceed MAX_SINGLE of the portfolio.
    If no eligible assets, freed capital goes to cash.
    """
    out = dict(w)

    if pool <= 1e-6:
        return out

    # Eligible risky assets with positive weight
    candidates = {a: out[a] for a in eligible if a != "cash" and out.get(a, 0) > 0}

    if not candidates:
        out["cash"] = out.get("cash", 0) + pool
        return out

    remaining_pool = pool
    for _ in range(5):  # iterative to handle cap overflow
        if remaining_pool <= 1e-6:
            break

        total_w = sum(candidates.values())
        if total_w <= 0:
            out["cash"] = out.get("cash", 0) + remaining_pool
            remaining_pool = 0.0
            break

        overflow = 0.0
        capped = set()
        for a, aw in candidates.items():
            share = remaining_pool * (aw / total_w)
            new_w = out[a] + share
            if new_w > MAX_SINGLE:
                overflow += new_w - MAX_SINGLE
                out[a] = MAX_SINGLE
                capped.add(a)
            else:
                out[a] = new_w

        remaining_pool = overflow
        candidates = {a: out[a] for a in candidates if a not in capped and out[a] < MAX_SINGLE}

    if remaining_pool > 1e-6:
        out["cash"] = out.get("cash", 0) + remaining_pool

    return out


def redistribute_cross_sectional(w: dict, pool: float, eligible: list, trend_scores: dict) -> dict:
    """Faber-CrossSectional: redistribution weighted by trend strength.

    3/3 score gets 2x weight vs 2/3 score when receiving freed capital.
    Same MAX_SINGLE cap as pro-rata.
    """
    out = dict(w)

    if pool <= 1e-6:
        return out

    # Score-weighted candidates: 3/3 gets weight 2, 2/3 gets weight 1
    candidates = {}
    for a in eligible:
        if a == "cash" or out.get(a, 0) <= 0:
            continue
        score = trend_scores.get(a, 0)
        strength = 2.0 if score >= 3 else 1.0
        candidates[a] = out[a] * strength

    if not candidates:
        out["cash"] = out.get("cash", 0) + pool
        return out

    remaining_pool = pool
    for _ in range(5):
        if remaining_pool <= 1e-6:
            break

        total_score = sum(candidates.values())
        if total_score <= 0:
            out["cash"] = out.get("cash", 0) + remaining_pool
            remaining_pool = 0.0
            break

        overflow = 0.0
        capped = set()
        for a, sc in candidates.items():
            share = remaining_pool * (sc / total_score)
            new_w = out[a] + share
            if new_w > MAX_SINGLE:
                overflow += new_w - MAX_SINGLE
                out[a] = MAX_SINGLE
                capped.add(a)
            else:
                out[a] = new_w

        remaining_pool = overflow
        candidates = {a: candidates[a] for a in candidates if a not in capped and out[a] < MAX_SINGLE}

    if remaining_pool > 1e-6:
        out["cash"] = out.get("cash", 0) + remaining_pool

    return out


# ── Harvey capital direction (benchmark) ─────────────────────────────────────

def redistribute_harvey(w: dict, pool: float, eligible: list, trend_scores: dict,
                        harvey_er: dict, rvols: dict) -> dict:
    """Harvey: inverse-vol capital direction for the reallocation pool."""
    out = dict(w)

    if pool <= 1e-6:
        return out

    candidates = {}
    for a in eligible:
        if a == "cash" or out.get(a, 0) <= 0:
            continue
        er = harvey_er.get(a, 0)
        vol = rvols.get(a, 0.15)
        if er > 0 and vol > 0.01:
            candidates[a] = er / vol

    if not candidates:
        out["cash"] = out.get("cash", 0) + pool
        return out

    if len(candidates) == 1:
        a = list(candidates.keys())[0]
        alloc = max(min(pool, 0.40 - out.get(a, 0)), 0)
        out[a] = out.get(a, 0) + alloc
        out["cash"] = out.get("cash", 0) + (pool - alloc)
        return out

    total = sum(candidates.values())
    for a, sc in candidates.items():
        out[a] = out.get(a, 0) + pool * (sc / total)

    return out


# ── Backtest ─────────────────────────────────────────────────────────────────

def run_backtest():
    print("=" * 80)
    print("  PRO-RATA REDISTRIBUTION vs CASH BASELINE")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    monthly_prices = load_monthly_prices()
    asset_ret = load_monthly_asset_returns()
    asset_ret_fwd = asset_ret.shift(-1)
    trend_df = compute_trend_scores(monthly_prices)

    # Harvey similarity engine data
    from taa.data import load_monthly_macro, load_realized_vols
    from taa.harvey import compute_zscore_variables, find_similar_months, compute_expected_returns
    macro = load_monthly_macro()
    z_data = compute_zscore_variables(macro)
    z_cols = [c for c in z_data.columns if c.endswith("_z")]
    z_clean = z_data[z_cols].dropna()
    rvol_monthly = load_realized_vols()

    # Determine backtest dates
    bt_start = pd.Timestamp(BACKTEST_START)
    bt_dates = asset_ret.index[asset_ret.index >= bt_start]
    bt_dates = bt_dates[bt_dates.isin(trend_df.index)]
    bt_dates = bt_dates[bt_dates.isin(z_clean.index)]

    print(f"Backtest: {bt_dates.min().date()} to {bt_dates.max().date()} ({len(bt_dates)} months)")

    # Results
    results = {s: {} for s in STRATEGIES}
    weight_log = {s: [] for s in STRATEGIES[:3]}  # log weights for the 3 test strategies
    alignment_violations = 0

    for dt in bt_dates:
        avail = [a for a in ASSETS if a in asset_ret.columns and pd.notna(asset_ret.loc[dt, a])]
        if len(avail) < 3:
            continue
        actual = {a: float(asset_ret.loc[dt, a]) for a in avail}

        # ── Signal alignment assertion ───────────────────────────────────
        # trend_df is already shifted by 1 in compute_trend_scores (PIT safe)
        # trend_df[dt] uses prices through month dt-1
        # We apply these weights to month dt returns (which are month T+1 relative
        # to the data used for signals at T-1... but in the monthly framework,
        # trend_df[dt] = signal from prices through dt-1, applied to dt returns)
        assert dt in trend_df.index, f"Signal alignment: {dt} not in trend_df"

        # Get trend scores for this month
        cur_trends = {}
        for a in ASSETS:
            if a == "cash":
                continue
            if a in trend_df.columns:
                v = trend_df.loc[dt, a]
                cur_trends[a] = int(v) if pd.notna(v) else 0
            else:
                cur_trends[a] = 0

        # ── Faber filter (identical for all 3 test strategies) ───────────
        faber_w, pool, eligible = apply_faber_filter(cur_trends)

        # ── Strategy 1: Faber-Cash ───────────────────────────────────────
        w_cash = redistribute_cash(dict(faber_w), pool, eligible, cur_trends)
        results["Faber-Cash"][dt] = sum(w_cash.get(a, 0) * actual.get(a, 0) for a in avail)
        weight_log["Faber-Cash"].append({"date": dt, **w_cash})

        # ── Strategy 2: Faber-ProRata ────────────────────────────────────
        w_pr = redistribute_pro_rata(dict(faber_w), pool, eligible, cur_trends)
        results["Faber-ProRata"][dt] = sum(w_pr.get(a, 0) * actual.get(a, 0) for a in avail)
        weight_log["Faber-ProRata"].append({"date": dt, **w_pr})

        # ── Strategy 3: Faber-CrossSectional ─────────────────────────────
        w_cs = redistribute_cross_sectional(dict(faber_w), pool, eligible, cur_trends)
        results["Faber-CrossSectional"][dt] = sum(w_cs.get(a, 0) * actual.get(a, 0) for a in avail)
        weight_log["Faber-CrossSectional"].append({"date": dt, **w_cs})

        # ── Signal alignment: verify weights sum to ~1 ───────────────────
        for label, w in [("Cash", w_cash), ("ProRata", w_pr), ("CrossSectional", w_cs)]:
            total_w = sum(w.get(a, 0) for a in ASSETS)
            assert abs(total_w - 1.0) < 0.01, f"{label} weights sum to {total_w:.4f} at {dt}"

        # ── Harvey (behind Faber) benchmark ──────────────────────────────
        z_prior = z_clean.index[z_clean.index < dt]
        harvey_er = {}
        rvols = {}
        if len(z_prior) > 0:
            try:
                sim, _ = find_similar_months(z_clean, z_prior[-1])
                harvey_er = compute_expected_returns(sim, asset_ret_fwd, avail)
            except ValueError:
                pass

        rv_prior = rvol_monthly.index[rvol_monthly.index <= dt]
        if len(rv_prior) > 0:
            for a in ASSETS:
                if a == "cash":
                    continue
                if a in rvol_monthly.columns:
                    v = rvol_monthly.loc[rv_prior[-1], a]
                    rvols[a] = float(v) if pd.notna(v) and v > 0 else 0.15

        w_harvey = redistribute_harvey(dict(faber_w), pool, eligible, cur_trends, harvey_er, rvols)
        results["Harvey (behind Faber)"][dt] = sum(w_harvey.get(a, 0) * actual.get(a, 0) for a in avail)

        # ── Benchmarks ───────────────────────────────────────────────────
        results["IVV B&H"][dt] = actual.get("IVV", 0)
        r_6040 = 0.60 * actual.get("IVV", 0) + 0.40 * actual.get("VGLT", 0)
        results["60/40"][dt] = r_6040

    # Convert to series
    ret_series = {s: pd.Series(d).sort_index() for s, d in results.items()}
    return ret_series, weight_log, bt_dates


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(s: pd.Series) -> dict:
    ar = s.mean() * 12
    av = s.std() * np.sqrt(12)
    sh = ar / av if av > 0 else 0
    neg = s[s < 0]
    ds = neg.std() * np.sqrt(12) if len(neg) > 10 else av
    sortino = ar / ds if ds > 0 else 0
    cum = (1 + s).cumprod()
    dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
    calmar = ar / abs(dd) if dd != 0 else 0
    final = cum.iloc[-1]
    return {"ar": ar, "av": av, "sh": sh, "sortino": sortino, "dd": dd, "calmar": calmar, "final": final}


# ── Report ───────────────────────────────────────────────────────────────────

def print_report(ret_series, weight_log, bt_dates):
    print(f"\nBacktest: {bt_dates.min().date()} to {bt_dates.max().date()} ({len(bt_dates)} months)")
    print(f"Rebalance: Monthly | Leverage: None (1x) | Faber: 6/10/12 SMA")
    print(f"Baseline: IVV 45%, QQQ 25%, VGLT 5%, IAU 10%, DBC 5%, Cash 10%")
    print(f"Pro-rata cap: {MAX_SINGLE:.0%} per asset")

    perfs = {s: compute_metrics(ret_series[s]) for s in STRATEGIES if len(ret_series[s]) > 12}

    # ── Performance table ────────────────────────────────────────────────
    print(f"\n{'='*95}")
    print(f"  PERFORMANCE")
    print(f"{'='*95}")
    print(f"  {'Strategy':<25} {'Return':>8} {'Vol':>7} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'Calmar':>8} {'Terminal':>10}")
    print(f"  {'-'*25} {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    for s in STRATEGIES:
        if s not in perfs:
            continue
        p = perfs[s]
        print(f"  {s:<25} {p['ar']:>7.1%} {p['av']:>6.1%} {p['sh']:>8.3f} {p['sortino']:>8.3f} {p['dd']:>7.1%} {p['calmar']:>8.2f} ${p['final']:>9.2f}")

    # ── Crisis analysis ──────────────────────────────────────────────────
    print(f"\n{'='*95}")
    print(f"  CRISIS ANALYSIS")
    print(f"{'='*95}")
    for cname, cs, ce in [("GFC (2008-2009)", "2008-09", "2009-03"),
                           ("COVID (2020)", "2020-02", "2020-04"),
                           ("2022 Bear", "2022-01", "2022-10")]:
        print(f"\n  {cname}:")
        print(f"  {'Strategy':<25} {'Return':>10} {'MaxDD':>10}")
        print(f"  {'-'*25} {'-'*10} {'-'*10}")
        for s in STRATEGIES:
            sr = ret_series.get(s)
            if sr is None:
                continue
            c = sr[(sr.index >= pd.Timestamp(cs)) & (sr.index <= pd.Timestamp(ce))]
            if len(c) > 0:
                cum = (1 + c).cumprod()
                mdd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
                print(f"  {s:<25} {(1+c).prod()-1:>+9.1%} {mdd:>9.1%}")

    # ── Alpha summary ────────────────────────────────────────────────────
    print(f"\n{'='*95}")
    print(f"  ALPHA SUMMARY")
    print(f"{'='*95}")
    fc = perfs.get("Faber-Cash", {})
    harvey = perfs.get("Harvey (behind Faber)", {})
    pr = perfs.get("Faber-ProRata", {})
    cs = perfs.get("Faber-CrossSectional", {})

    if fc and pr:
        print(f"\n  Pro-rata alpha over Faber-Cash:           {pr['sh'] - fc['sh']:>+.3f} Sharpe")
    if fc and cs:
        print(f"  Cross-sectional alpha over Faber-Cash:    {cs['sh'] - fc['sh']:>+.3f} Sharpe")
    if harvey and pr:
        print(f"  Pro-rata alpha over Harvey:               {pr['sh'] - harvey['sh']:>+.3f} Sharpe")
    if harvey and cs:
        print(f"  Cross-sectional alpha over Harvey:        {cs['sh'] - harvey['sh']:>+.3f} Sharpe")

    # ── Weight diagnostics ───────────────────────────────────────────────
    print(f"\n{'='*95}")
    print(f"  WEIGHT DIAGNOSTICS")
    print(f"{'='*95}")
    for strat in ["Faber-Cash", "Faber-ProRata", "Faber-CrossSectional"]:
        wl = weight_log.get(strat, [])
        if not wl:
            continue
        wdf = pd.DataFrame(wl).set_index("date")
        print(f"\n  {strat} — Average weights:")
        for a in ASSETS:
            if a in wdf.columns:
                print(f"    {a:<8}: {wdf[a].mean():>6.1%}  (max {wdf[a].max():>5.1%})")

    # ── Months where redistribution matters ──────────────────────────────
    fc_s = ret_series["Faber-Cash"]
    pr_s = ret_series["Faber-ProRata"]
    diff = pr_s - fc_s
    active_months = (diff.abs() > 1e-6).sum()
    print(f"\n  Months where pro-rata differs from cash: {active_months}/{len(fc_s)} ({active_months/len(fc_s):.0%})")
    print(f"  Mean monthly return diff (ProRata - Cash): {diff.mean()*100:+.3f}%")
    print(f"  Std of monthly diff: {diff.std()*100:.3f}%")

    # ── Calendar year returns ────────────────────────────────────────────
    print(f"\n{'='*95}")
    print(f"  CALENDAR YEAR RETURNS")
    print(f"{'='*95}")
    print(f"  {'Year':>6} {'Faber-Cash':>11} {'ProRata':>11} {'CrossSect':>11} {'Harvey':>11} {'IVV':>11}")
    for yr in range(2002, 2027):
        row = f"  {yr:>6}"
        for s in ["Faber-Cash", "Faber-ProRata", "Faber-CrossSectional", "Harvey (behind Faber)", "IVV B&H"]:
            sr = ret_series.get(s, pd.Series(dtype=float))
            y = sr[sr.index.year == yr]
            if len(y) > 3:
                row += f" {(1+y).prod()-1:>+10.1%}"
            else:
                row += f" {'--':>11}"
        print(row)

    print()
    return perfs


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ret_series, weight_log, bt_dates = run_backtest()
    perfs = print_report(ret_series, weight_log, bt_dates)
