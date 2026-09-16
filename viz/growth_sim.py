"""Historical start-date growth simulation.

Maps every monthly start in a daily return series onto a lump + monthly
contribution plan at that window's realized CAGR. Not Monte Carlo.

Used by viz/pages/1_Growth_simulation.py and as a CLI:

  .venv/bin/python -m viz.growth_sim --start 5000 --monthly 300 --years 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TDY = 252
CAGR_PCTS = (0, 5, 10, 25, 50, 75, 90, 95, 100)
CAGR_HIST_EDGES = np.array([-1.0, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 1.0])
CAGR_HIST_LABELS = ["<0%", "0–5%", "5–10%", "10–15%", "15–20%", "20–25%", "25%+"]

# Default after-tax assumptions (taxable account). Sensitivity model, not a tax
# engine. See after_tax_cagr() for the mechanism and the Growth-sim page sidebar
# for the live controls.
DEFAULT_ORDINARY = 0.32   # federal short-term / ordinary rate
DEFAULT_LTCG = 0.15       # federal long-term cap-gains rate
DEFAULT_STATE = 0.05      # state rate (stacks on both)
DEFAULT_STCG_FRACTION = 0.50  # of realized gains, share taxed short-term
DEFAULT_REALIZED_FRACTION = 0.65  # of each year's gain, share realized (turnover)


def blended_gain_rate(
    ordinary: float = DEFAULT_ORDINARY,
    ltcg: float = DEFAULT_LTCG,
    state: float = DEFAULT_STATE,
    stcg_fraction: float = DEFAULT_STCG_FRACTION,
) -> float:
    """Blended marginal rate on a realized dollar of gain (fed STCG/LTCG mix + state).

    Mirrors live.tax.effective_gain_rate so the two dashboards agree.
    """
    fed = stcg_fraction * ordinary + (1.0 - stcg_fraction) * ltcg
    return fed + state


def after_tax_cagr(
    cagr: np.ndarray | float,
    blended_rate: float,
    realized_fraction: float,
) -> np.ndarray | float:
    """Haircut a pre-tax CAGR to an after-tax CAGR.

    Model: each year, a fraction ``realized_fraction`` of that year's gain is
    sold/realized and taxed at ``blended_rate``; the rest defers (compounds
    untaxed — no terminal-liquidation tax modeled). Losses are not credited.

        g_after = g − blended_rate · realized_fraction · max(g, 0)

    Monotone in g, so percentiles are preserved. Applied to each window's
    realized annual return, this is the constant-CAGR analogue of taxing
    realized gains once a year.
    """
    g = np.asarray(cagr, dtype=float)
    out = g - blended_rate * realized_fraction * np.clip(g, 0.0, None)
    return float(out) if np.ndim(cagr) == 0 else out


def window_stats(
    daily: pd.Series,
    years: int,
    bench: pd.Series | None = None,
) -> pd.DataFrame:
    """One row per month-start with a full `years`-year daily path."""
    s = daily.dropna()
    if len(s) < years * TDY + 1:
        return pd.DataFrame(columns=["start", "cagr", "maxdd", "beat_bench"])
    k = years * TDY
    idx = s.index
    vals = s.to_numpy(dtype=float)
    bvals = None
    if bench is not None:
        bvals = bench.reindex(idx).to_numpy(dtype=float)
    month_starts = pd.date_range(idx.min().replace(day=1), idx.max(), freq="MS")
    rows = []
    for ms in month_starts:
        loc = int(idx.searchsorted(ms))
        if loc + k > len(idx):
            continue
        sl = vals[loc : loc + k]
        if not np.isfinite(sl).all():
            continue
        wealth = np.cumprod(1.0 + sl)
        cagr = float(wealth[-1] ** (1.0 / years) - 1.0)
        peak = np.maximum.accumulate(wealth)
        maxdd = float((wealth / peak - 1.0).min())
        beat = None
        if bvals is not None:
            bl = bvals[loc : loc + k]
            if np.isfinite(bl).all():
                beat = bool(wealth[-1] > np.prod(1.0 + bl))
        rows.append(
            {
                "start": idx[loc],
                "cagr": cagr,
                "maxdd": maxdd,
                "beat_bench": beat,
            }
        )
    return pd.DataFrame(rows)


def cagr_percentiles(cagrs: np.ndarray) -> dict[int, float]:
    if len(cagrs) == 0:
        return {p: float("nan") for p in CAGR_PCTS}
    qs = np.percentile(cagrs, CAGR_PCTS)
    return {int(p): float(q) for p, q in zip(CAGR_PCTS, qs)}


def cagr_histogram(cagrs: np.ndarray) -> pd.Series:
    if len(cagrs) == 0:
        return pd.Series(0.0, index=CAGR_HIST_LABELS)
    counts, _ = np.histogram(cagrs, bins=CAGR_HIST_EDGES)
    return pd.Series(counts / len(cagrs) * 100.0, index=CAGR_HIST_LABELS)


def fv(start: float, monthly: float, years: float, cagr: float) -> float:
    """Lump at t=0, then `years*12` months of grow-then-contribute."""
    n = int(round(years * 12))
    rm = (1.0 + cagr) ** (1.0 / 12.0) - 1.0
    if abs(rm) < 1e-12:
        return start + monthly * n
    g = (1.0 + rm) ** n
    return start * g + monthly * ((g - 1.0) / rm)


def wealth_path(start: float, monthly: float, years: int, cagr: float) -> pd.Series:
    """Year-end wealth under constant CAGR (year 0 = starting lump)."""
    vals = [float(start)]
    for y in range(1, years + 1):
        vals.append(fv(start, monthly, y, cagr))
    return pd.Series(vals, index=range(0, years + 1), name="wealth")


def apply_plan(
    stats: pd.DataFrame,
    start: float,
    monthly: float,
    years: int,
    blended_rate: float = 0.0,
    realized_fraction: float = DEFAULT_REALIZED_FRACTION,
) -> dict:
    """Map the window CAGR distribution onto the plan, pre- and after-tax.

    ``blended_rate`` = 0 disables tax (after-tax == pre-tax). Otherwise every
    window's realized CAGR is haircut by after_tax_cagr() before percentiles and
    terminal values are recomputed, so the after-tax distribution is internally
    consistent rather than a flat offset off the median.
    """
    cagrs = stats["cagr"].to_numpy(dtype=float)
    pct = cagr_percentiles(cagrs)
    hist = cagr_histogram(cagrs)
    terms = {p: fv(start, monthly, years, c) for p, c in pct.items()}
    cash_in = start + monthly * 12 * years
    dd = stats["maxdd"].to_numpy(dtype=float)
    beat = stats["beat_bench"].dropna()
    ge10 = float((cagrs >= 0.10).mean() * 100.0) if len(cagrs) else float("nan")

    at_cagrs = after_tax_cagr(cagrs, blended_rate, realized_fraction)
    at_pct = cagr_percentiles(at_cagrs)
    at_terms = {p: fv(start, monthly, years, c) for p, c in at_pct.items()}

    return {
        "n": int(len(stats)),
        "cash_in": cash_in,
        "cagr_pct": pct,
        "terminal": terms,
        "hist_pct": hist,
        "maxdd_pct": {
            0: float(np.percentile(dd, 0)) if len(dd) else float("nan"),
            10: float(np.percentile(dd, 10)) if len(dd) else float("nan"),
            50: float(np.percentile(dd, 50)) if len(dd) else float("nan"),
        },
        "beat_bench_pct": float(beat.mean() * 100.0) if len(beat) else float("nan"),
        "ge10_pct": ge10,
        "path_p10": wealth_path(start, monthly, years, pct[10]),
        "path_median": wealth_path(start, monthly, years, pct[50]),
        "path_p90": wealth_path(start, monthly, years, pct[90]),
        # After-tax mirror
        "blended_rate": blended_rate,
        "realized_fraction": realized_fraction,
        "at_cagr_pct": at_pct,
        "at_terminal": at_terms,
        "at_path_p10": wealth_path(start, monthly, years, at_pct[10]),
        "at_path_median": wealth_path(start, monthly, years, at_pct[50]),
        "at_path_p90": wealth_path(start, monthly, years, at_pct[90]),
    }


def _default_package() -> Path:
    from viz.data import PACKAGES, list_packages

    pkgs = list_packages()
    preferred = PACKAGES / "v19d_marketstack_verification"
    if preferred.exists():
        return preferred
    if not pkgs:
        raise SystemExit("No viz/packages found. Export one first.")
    return pkgs[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=float, default=5_000)
    parser.add_argument("--monthly", type=float, default=300)
    parser.add_argument("--years", type=int, default=10)
    parser.add_argument("--column", default=None, help="Strategy column (default: package primary)")
    parser.add_argument("--package", default=None, help="Path to viz package folder")
    parser.add_argument("--tax", action="store_true", help="Also print after-tax (taxable) estimate")
    parser.add_argument("--realized", type=float, default=DEFAULT_REALIZED_FRACTION,
                        help="Share of each year's gain realized/taxed (0-1)")
    parser.add_argument("--ordinary", type=float, default=DEFAULT_ORDINARY)
    parser.add_argument("--ltcg", type=float, default=DEFAULT_LTCG)
    parser.add_argument("--state", type=float, default=DEFAULT_STATE)
    parser.add_argument("--stcg-frac", type=float, default=DEFAULT_STCG_FRACTION)
    args = parser.parse_args()

    from viz.data import load_package, strategy_map

    pkg_path = Path(args.package) if args.package else _default_package()
    pkg = load_package(str(pkg_path))
    meta = pkg["meta"]
    col = args.column or meta.get("primary_strategy") or pkg["daily"].columns[0]
    bench_col = next(
        (s["id"] for s in meta.get("strategies", []) if s["id"] in ("qqq_bh", "spy_bh") and s["id"] in pkg["daily"].columns),
        None,
    )
    stats = window_stats(
        pkg["daily"][col],
        args.years,
        pkg["daily"][bench_col] if bench_col else None,
    )
    blended = blended_gain_rate(args.ordinary, args.ltcg, args.state, args.stcg_frac) if args.tax else 0.0
    plan = apply_plan(stats, args.start, args.monthly, args.years, blended, args.realized)
    smap = strategy_map(meta)
    name = smap.get(col, {}).get("name", col)
    print(f"{name}  {pkg_path.name}")
    print(f"plan    ${args.start:,.0f} + ${args.monthly:,.0f}/mo × {args.years}y")
    print(f"windows {plan['n']}  cash-in ${plan['cash_in']:,.0f}")
    print(
        f"CAGR    min {plan['cagr_pct'][0]:.1%}  p10 {plan['cagr_pct'][10]:.1%}  "
        f"med {plan['cagr_pct'][50]:.1%}  p90 {plan['cagr_pct'][90]:.1%}"
    )
    print(
        f"end $   worst {plan['terminal'][0]:,.0f}  p10 {plan['terminal'][10]:,.0f}  "
        f"med {plan['terminal'][50]:,.0f}  p90 {plan['terminal'][90]:,.0f}"
    )
    print(
        f"maxDD   worst {plan['maxdd_pct'][0]:.1%}  p10 {plan['maxdd_pct'][10]:.1%}  "
        f"med {plan['maxdd_pct'][50]:.1%}"
    )
    if not np.isnan(plan["beat_bench_pct"]):
        print(f"beat {bench_col} {plan['beat_bench_pct']:.1f}% of windows")

    if args.tax:
        atc = plan["at_cagr_pct"]
        att = plan["at_terminal"]
        print(
            f"\nafter-tax  blended {blended:.1%}  realized {args.realized:.0%}/yr"
            f"  (STCG frac {args.stcg_frac:.0%})"
        )
        print(
            f"CAGR    med {atc[50]:.1%} (−{plan['cagr_pct'][50]-atc[50]:.1%})"
            f"  p10 {atc[10]:.1%}  p90 {atc[90]:.1%}"
        )
        drag = 1 - att[50] / plan["terminal"][50] if plan["terminal"][50] else float("nan")
        print(
            f"end $   med {att[50]:,.0f} (−{drag:.0%} vs pre-tax {plan['terminal'][50]:,.0f})"
            f"  p10 {att[10]:,.0f}  p90 {att[90]:,.0f}"
        )


if __name__ == "__main__":
    main()
