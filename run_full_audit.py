"""Full backtest audit — runs both windows and produces detailed analytics."""

from __future__ import annotations

import pickle
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from statistics import mean, stdev

# Ensure imports work
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from utils.logging import setup_logging
setup_logging("WARNING")  # suppress verbose scoring logs

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING")

import numpy as np
import pandas as pd

from backtest.runner import run_backtest, preflight_checks, BacktestResult
from backtest.trade_simulator import BacktestTrade
from backtest.macro_backfill import load_cached_condition_vectors
from engine.strategy_scorer import rank_strategies, STRATEGY_DISPLAY, score_strategy


# ── Run both windows ─────────────────────────────────────────────────────────

print("=" * 80)
print("  MACRO REGIME ENGINE — FULL BACKTEST AUDIT")
print("=" * 80)
print()

print("Running pre-COVID window (2010-01-04 to 2020-02-19)...")
pre_result = run_backtest(
    start_override=date(2010, 1, 4),
    end_override=date(2020, 2, 19),
)
pre_trades = pre_result.pre_covid.trades if pre_result.pre_covid else []
print(f"  → {len(pre_trades)} trades")

print("Running post-COVID window (2020-02-20 to 2023-12-29)...")
post_result = run_backtest(
    start_override=date(2020, 2, 20),
    end_override=date(2023, 12, 29),
)
post_trades = post_result.pre_covid.trades if post_result.pre_covid else []
print(f"  → {len(post_trades)} trades")

all_trades = pre_trades + post_trades
print(f"\nTotal trades: {len(all_trades)}")

# Save for later analysis
AUDIT_PATH = Path("backtest_results")
AUDIT_PATH.mkdir(exist_ok=True)
with open(AUDIT_PATH / "audit_trades.pickle", "wb") as f:
    pickle.dump({"pre": pre_trades, "post": post_trades}, f)

# Load condition vectors for edge analysis
cvs = load_cached_condition_vectors() or {}


# ── Helper functions ─────────────────────────────────────────────────────────

def _pf(wins_pnl: float, losses_pnl: float) -> float:
    return round(wins_pnl / abs(losses_pnl), 2) if losses_pnl != 0 else float("inf")

def _max_consec(trades: list[BacktestTrade], cond) -> int:
    mx = cur = 0
    for t in sorted(trades, key=lambda x: x.entry_date):
        if cond(t):
            cur += 1
            mx = max(mx, cur)
        else:
            cur = 0
    return mx

def _sharpe(trades: list[BacktestTrade], capital: float = 100_000) -> float:
    if len(trades) < 2:
        return 0.0
    eq = [capital]
    for t in sorted(trades, key=lambda x: x.exit_date or x.entry_date):
        eq.append(eq[-1] + t.pnl * 100)
    rets = pd.Series(eq).pct_change().dropna()
    if rets.std() == 0:
        return 0.0
    avg_hold = mean([t.days_held for t in trades]) or 30
    return float(rets.mean() / rets.std() * np.sqrt(252 / avg_hold))

def _max_dd(trades: list[BacktestTrade], capital: float = 100_000) -> float:
    if not trades:
        return 0.0
    eq = [capital]
    for t in sorted(trades, key=lambda x: x.exit_date or x.entry_date):
        eq.append(eq[-1] + t.pnl * 100)
    s = pd.Series(eq)
    peak = s.expanding().max()
    dd = (s - peak) / peak
    return float(dd.min())

def _equity_curve(trades: list[BacktestTrade], capital: float = 10_000) -> list[float]:
    eq = [capital]
    for t in sorted(trades, key=lambda x: x.exit_date or x.entry_date):
        eq.append(eq[-1] + t.pnl * 100)
    return eq

def _ann_return(trades: list[BacktestTrade], capital: float = 100_000) -> float:
    if not trades:
        return 0.0
    eq = _equity_curve(trades, capital)
    first = sorted(trades, key=lambda x: x.entry_date)[0].entry_date
    last = sorted(trades, key=lambda x: x.exit_date or x.entry_date)[-1].exit_date or first
    days = (last - first).days
    if days <= 0:
        return 0.0
    years = days / 365.25
    return ((eq[-1] / capital) ** (1 / years) - 1) * 100

def _window_stats(trades: list[BacktestTrade], label: str) -> dict:
    if not trades:
        return {}
    wins = [t for t in trades if t.win]
    losses = [t for t in trades if not t.win]
    pnls = [t.pnl for t in trades]
    win_pnls = [t.pnl for t in wins]
    loss_pnls = [t.pnl for t in losses]
    return {
        "label": label,
        "n": len(trades),
        "win_rate": len(wins) / len(trades),
        "total_pnl": sum(pnls),
        "total_pnl_contract": sum(pnls) * 100,
        "avg_pnl": mean(pnls),
        "avg_win": mean(win_pnls) if win_pnls else 0,
        "avg_loss": mean(loss_pnls) if loss_pnls else 0,
        "pf": _pf(sum(win_pnls) if win_pnls else 0, sum(loss_pnls) if loss_pnls else 0),
        "sharpe": _sharpe(trades),
        "max_dd": _max_dd(trades),
        "ann_ret": _ann_return(trades),
        "max_consec_wins": _max_consec(trades, lambda t: t.win),
        "max_consec_losses": _max_consec(trades, lambda t: not t.win),
        "avg_days": mean([t.days_held for t in trades]),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Overall summary per window
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  SECTION 1: OVERALL SUMMARY")
print("=" * 80)

for label, trades in [("PRE-COVID (2010-01 to 2020-02)", pre_trades),
                       ("POST-COVID (2020-02 to 2023-12)", post_trades),
                       ("COMBINED (2010-2023)", all_trades)]:
    s = _window_stats(trades, label)
    if not s:
        print(f"\n{label}: NO TRADES")
        continue
    print(f"\n{label}:")
    print(f"  Total trades:          {s['n']}")
    print(f"  Win rate:              {s['win_rate']:.1%}")
    print(f"  Total P&L/contract:    ${s['total_pnl_contract']:,.0f}")
    print(f"  Annualized return:     {s['ann_ret']:.2f}%")
    print(f"  Sharpe ratio:          {s['sharpe']:.2f}")
    print(f"  Profit factor:         {s['pf']:.2f}")
    print(f"  Max drawdown:          {s['max_dd']:.2%}")
    print(f"  Max consecutive wins:  {s['max_consec_wins']}")
    print(f"  Max consecutive losses:{s['max_consec_losses']}")
    print(f"  Avg days held:         {s['avg_days']:.1f}")
    print(f"  Avg win P&L:           ${s['avg_win']:.2f}/share")
    print(f"  Avg loss P&L:          ${s['avg_loss']:.2f}/share")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Performance by YEAR
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  SECTION 2: PERFORMANCE BY YEAR")
print("=" * 80)

print(f"\n{'Year':>4} {'Trades':>6} {'WinR':>6} {'P&L($)':>9} {'Sharpe':>7} {'PF':>6} {'AvgDays':>7} {'Best Trade':>50} {'Worst Trade':>50}")
print("-" * 160)

for year in range(2010, 2024):
    yt = [t for t in all_trades if t.entry_date.year == year]
    if not yt:
        print(f"{year:>4} {'---':>6}")
        continue
    wins = [t for t in yt if t.win]
    pnls = [t.pnl for t in yt]
    wp = [t.pnl for t in wins]
    lp = [t.pnl for t in yt if not t.win]
    best = max(yt, key=lambda t: t.pnl)
    worst = min(yt, key=lambda t: t.pnl)
    wr = len(wins) / len(yt) if yt else 0
    pf = _pf(sum(wp) if wp else 0, sum(lp) if lp else 0)
    sh = _sharpe(yt)
    ad = mean([t.days_held for t in yt])

    # Dominant regime
    regimes = defaultdict(int)
    for t in yt:
        regimes[t.regime_label] += 1
    dom_regime = max(regimes, key=regimes.get)

    best_s = f"{best.entry_date} {best.strategy[:15]:15} ${best.pnl:+.2f}"
    worst_s = f"{worst.entry_date} {worst.strategy[:15]:15} ${worst.pnl:+.2f}"
    print(f"{year:>4} {len(yt):>6} {wr:>5.0%} {sum(pnls)*100:>+9.0f} {sh:>7.2f} {pf:>6.2f} {ad:>7.1f} {best_s:>50} {worst_s:>50}  [{dom_regime}]")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Performance by STRATEGY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  SECTION 3: PERFORMANCE BY STRATEGY")
print("=" * 80)

print(f"\n{'Strategy':>20} {'N':>5} {'WinR':>6} {'AvgPnL':>8} {'PF':>6} {'AvgDays':>7} {'TotPnL($)':>10} {'BestYr':>8} {'WorstYr':>8}")
print("-" * 90)

for strat in ["bull_put_spread", "bull_call_spread", "bear_call_spread", "bear_put_spread", "iron_condor"]:
    st = [t for t in all_trades if t.strategy == strat]
    if not st:
        print(f"{strat:>20} {'---':>5}")
        continue
    wins = [t for t in st if t.win]
    pnls = [t.pnl for t in st]
    wp = [t.pnl for t in wins]
    lp = [t.pnl for t in st if not t.win]
    wr = len(wins) / len(st)
    pf = _pf(sum(wp) if wp else 0, sum(lp) if lp else 0)
    ad = mean([t.days_held for t in st])

    # Best/worst year
    by_year = defaultdict(list)
    for t in st:
        by_year[t.entry_date.year].append(t.pnl)
    yr_totals = {y: sum(p) for y, p in by_year.items()}
    best_yr = max(yr_totals, key=yr_totals.get) if yr_totals else "N/A"
    worst_yr = min(yr_totals, key=yr_totals.get) if yr_totals else "N/A"

    print(f"{strat:>20} {len(st):>5} {wr:>5.0%} {mean(pnls):>+8.2f} {pf:>6.2f} {ad:>7.1f} {sum(pnls)*100:>+10.0f} {best_yr:>8} {worst_yr:>8}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Performance by REGIME
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  SECTION 4: PERFORMANCE BY REGIME")
print("=" * 80)

for regime in ["GOLDILOCKS", "REFLATION", "STAGFLATION", "DEFLATION", "MIXED"]:
    rt = [t for t in all_trades if t.regime_label == regime]
    if not rt:
        print(f"\n{regime}: NO TRADES")
        continue
    wins = [t for t in rt if t.win]
    pnls = [t.pnl for t in rt]
    wr = len(wins) / len(rt)

    # Strategy breakdown
    strat_counts = defaultdict(int)
    strat_pnl = defaultdict(list)
    for t in rt:
        strat_counts[t.strategy] += 1
        strat_pnl[t.strategy].append(t.pnl)

    most_used = max(strat_counts, key=strat_counts.get)
    best_strat = max(strat_pnl, key=lambda s: sum(strat_pnl[s]) if strat_pnl[s] else -999)

    print(f"\n{regime}: {len(rt)} trades, {wr:.0%} win rate, ${sum(pnls)*100:+,.0f} P&L")
    print(f"  Most used:  {most_used} ({strat_counts[most_used]})")
    print(f"  Best perf:  {best_strat} (${sum(strat_pnl[best_strat])*100:+,.0f})")
    for s in sorted(strat_counts.keys()):
        sp = strat_pnl[s]
        sw = len([p for p in sp if p > 0])
        print(f"    {s:>20}: {strat_counts[s]:>3} trades, {sw/len(sp):.0%} WR, ${sum(sp)*100:+,.0f}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Strategy × Regime cross-tab
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  SECTION 5: STRATEGY × REGIME CROSS-TAB (Win Rate / N trades)")
print("=" * 80)

regimes_list = ["GOLDILOCKS", "REFLATION", "STAGFLATION", "DEFLATION", "MIXED"]
strats_list = ["bull_put_spread", "bull_call_spread", "bear_call_spread", "bear_put_spread", "iron_condor"]

print(f"\n{'':>20}", end="")
for r in regimes_list:
    print(f"  {r:>14}", end="")
print()
print("-" * (20 + 16 * len(regimes_list)))

for strat in strats_list:
    print(f"{strat:>20}", end="")
    for regime in regimes_list:
        cell = [t for t in all_trades if t.strategy == strat and t.regime_label == regime]
        if not cell:
            print(f"  {'---':>14}", end="")
        else:
            w = len([t for t in cell if t.win])
            wr = w / len(cell)
            print(f"  {wr:>5.0%} ({len(cell):>3})", end="  ")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: Drawdown analysis
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  SECTION 6: DRAWDOWN ANALYSIS")
print("=" * 80)

eq = _equity_curve(all_trades, 10_000)
sorted_all = sorted(all_trades, key=lambda x: x.exit_date or x.entry_date)
eq_dates = [sorted_all[0].entry_date] + [t.exit_date or t.entry_date for t in sorted_all]

peak_val = eq[0]
peak_idx = 0
max_dd_val = 0
max_dd_peak_idx = 0
max_dd_trough_idx = 0

for i, val in enumerate(eq):
    if val >= peak_val:
        peak_val = val
        peak_idx = i
    dd = (val - peak_val) / peak_val
    if dd < max_dd_val:
        max_dd_val = dd
        max_dd_peak_idx = peak_idx
        max_dd_trough_idx = i

print(f"\nEquity curve: ${eq[0]:,.0f} → ${eq[-1]:,.0f}")
print(f"Peak: ${max(eq):,.0f}")
print(f"Max drawdown: {max_dd_val:.2%} (${eq[max_dd_trough_idx] - eq[max_dd_peak_idx]:+,.0f})")
if max_dd_peak_idx < len(eq_dates) and max_dd_trough_idx < len(eq_dates):
    print(f"  Peak date:   ~{eq_dates[max_dd_peak_idx]}")
    print(f"  Trough date: ~{eq_dates[max_dd_trough_idx]}")

# Drawdowns > 3%
print(f"\nSignificant drawdowns (> 3%):")
s = pd.Series(eq)
peak_s = s.expanding().max()
dd_s = (s - peak_s) / peak_s
in_dd = False
dd_start = 0
for i in range(len(dd_s)):
    if dd_s.iloc[i] < -0.03 and not in_dd:
        in_dd = True
        dd_start = i
    elif dd_s.iloc[i] >= 0 and in_dd:
        trough_idx = dd_s.iloc[dd_start:i].idxmin()
        d1 = eq_dates[dd_start] if dd_start < len(eq_dates) else "?"
        d2 = eq_dates[trough_idx] if trough_idx < len(eq_dates) else "?"
        d3 = eq_dates[i] if i < len(eq_dates) else "?"
        print(f"  {d1} → {d2}: {dd_s.iloc[trough_idx]:.1%} (recovered by ~{d3})")
        in_dd = False


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: Monthly returns
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  SECTION 7: MONTHLY RETURNS ($/contract)")
print("=" * 80)

monthly = defaultdict(float)
for t in all_trades:
    ed = t.exit_date or t.entry_date
    key = (ed.year, ed.month)
    monthly[key] += t.pnl * 100

print(f"\n{'':>5}", end="")
for m in range(1, 13):
    print(f"  {'JanFebMarAprMayJunJulAugSepOctNovDec'[(m-1)*3:m*3]:>6}", end="")
print(f"  {'TOTAL':>8}")
print("-" * (5 + 8 * 13))

for year in range(2010, 2024):
    print(f"{year:>4} ", end="")
    yr_total = 0
    for m in range(1, 13):
        val = monthly.get((year, m), 0)
        yr_total += val
        if val == 0:
            print(f"  {'--':>6}", end="")
        elif val < -500:
            print(f"  {val:>+6.0f}*", end="")  # flag big losses
        else:
            print(f"  {val:>+6.0f}", end="")
    print(f"  {yr_total:>+8.0f}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: Trade management analysis
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  SECTION 8: TRADE MANAGEMENT ANALYSIS")
print("=" * 80)

reasons = defaultdict(list)
for t in all_trades:
    reasons[t.exit_reason or "unknown"].append(t)

print(f"\n{'Exit Reason':>25} {'N':>5} {'%':>6} {'WinR':>6} {'AvgPnL':>8} {'TotPnL($)':>10}")
print("-" * 72)
for reason in sorted(reasons.keys()):
    rt = reasons[reason]
    wins = [t for t in rt if t.win]
    pnls = [t.pnl for t in rt]
    pct = len(rt) / len(all_trades)
    wr = len(wins) / len(rt) if rt else 0
    print(f"{reason:>25} {len(rt):>5} {pct:>5.0%} {wr:>5.0%} {mean(pnls):>+8.2f} {sum(pnls)*100:>+10.0f}")

# Check credit vs debit split
credit_trades = [t for t in all_trades if t.entry_credit > 0]
debit_trades = [t for t in all_trades if t.entry_credit < 0]
print(f"\nCredit spreads: {len(credit_trades)} trades, "
      f"{len([t for t in credit_trades if t.win])/len(credit_trades):.0%} WR, "
      f"${sum(t.pnl for t in credit_trades)*100:+,.0f}" if credit_trades else "\nCredit spreads: 0")
print(f"Debit spreads:  {len(debit_trades)} trades, "
      f"{len([t for t in debit_trades if t.win])/len(debit_trades):.0%} WR, "
      f"${sum(t.pnl for t in debit_trades)*100:+,.0f}" if debit_trades else "Debit spreads: 0")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: Regime transition analysis
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  SECTION 9: REGIME TRANSITION ANALYSIS")
print("=" * 80)

# Track regime by entry date
sorted_trades = sorted(all_trades, key=lambda x: x.entry_date)
transitions = []
for i in range(1, len(sorted_trades)):
    prev = sorted_trades[i - 1]
    curr = sorted_trades[i]
    if prev.regime_label != curr.regime_label:
        transitions.append((prev.regime_label, curr.regime_label, curr.entry_date))

print(f"\nTotal regime transitions: {len(transitions)}")

# Count transition types
trans_counts = defaultdict(int)
for fr, to, dt in transitions:
    trans_counts[f"{fr} → {to}"] += 1

print(f"\nTransition frequencies:")
for tr in sorted(trans_counts, key=trans_counts.get, reverse=True):
    print(f"  {tr:>30}: {trans_counts[tr]:>3}")

# Regime persistence
regime_runs = []
if sorted_trades:
    cur_regime = sorted_trades[0].regime_label
    cur_start = sorted_trades[0].entry_date
    for t in sorted_trades[1:]:
        if t.regime_label != cur_regime:
            regime_runs.append((cur_regime, (t.entry_date - cur_start).days))
            cur_regime = t.regime_label
            cur_start = t.entry_date
    regime_runs.append((cur_regime, (sorted_trades[-1].entry_date - cur_start).days))

if regime_runs:
    regime_dur = defaultdict(list)
    for r, d in regime_runs:
        regime_dur[r].append(d)
    print(f"\nAverage regime persistence (days):")
    for r in sorted(regime_dur.keys()):
        durs = regime_dur[r]
        print(f"  {r:>15}: avg={mean(durs):.0f}d, median={sorted(durs)[len(durs)//2]}d, n={len(durs)}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10: Edge analysis
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  SECTION 10: EDGE ANALYSIS (Actual Win Rate vs Delta-Implied)")
print("=" * 80)

# For credit spreads: implied win = 1 - |short_delta|
# For debit spreads: implied win = |long_delta| (need to reach max profit)
edge_by_strat = defaultdict(lambda: {"actual_wins": 0, "n": 0, "implied_sum": 0})
edge_by_regime = defaultdict(lambda: {"actual_wins": 0, "n": 0, "implied_sum": 0})

for t in all_trades:
    if t.strategy_type == "credit":
        implied = 1.0 - abs(t.short_delta)
    else:
        implied = abs(t.long_delta)

    implied = max(0.05, min(0.95, implied))  # clamp

    edge_by_strat[t.strategy]["n"] += 1
    edge_by_strat[t.strategy]["implied_sum"] += implied
    if t.win:
        edge_by_strat[t.strategy]["actual_wins"] += 1

    edge_by_regime[t.regime_label]["n"] += 1
    edge_by_regime[t.regime_label]["implied_sum"] += implied
    if t.win:
        edge_by_regime[t.regime_label]["actual_wins"] += 1

print(f"\n{'':>20} {'N':>5} {'Actual WR':>10} {'Implied WR':>12} {'Edge':>8}")
print("-" * 60)

print("BY STRATEGY:")
for strat in strats_list:
    d = edge_by_strat[strat]
    if d["n"] == 0:
        continue
    actual_wr = d["actual_wins"] / d["n"]
    implied_wr = d["implied_sum"] / d["n"]
    edge = actual_wr - implied_wr
    marker = " ✓" if edge > 0 else " ✗"
    print(f"  {strat:>20} {d['n']:>5} {actual_wr:>9.1%} {implied_wr:>11.1%} {edge:>+7.1%}{marker}")

print("\nBY REGIME:")
for regime in regimes_list:
    d = edge_by_regime[regime]
    if d["n"] == 0:
        continue
    actual_wr = d["actual_wins"] / d["n"]
    implied_wr = d["implied_sum"] / d["n"]
    edge = actual_wr - implied_wr
    marker = " ✓" if edge > 0 else " ✗"
    print(f"  {regime:>20} {d['n']:>5} {actual_wr:>9.1%} {implied_wr:>11.1%} {edge:>+7.1%}{marker}")

# Overall edge
total_actual = sum(1 for t in all_trades if t.win)
total_implied = sum(
    (1.0 - abs(t.short_delta)) if t.strategy_type == "credit" else abs(t.long_delta)
    for t in all_trades
)
overall_actual_wr = total_actual / len(all_trades) if all_trades else 0
overall_implied_wr = total_implied / len(all_trades) if all_trades else 0
print(f"\n  {'OVERALL':>20} {len(all_trades):>5} {overall_actual_wr:>9.1%} {overall_implied_wr:>11.1%} {overall_actual_wr - overall_implied_wr:>+7.1%}")


# ═══════════════════════════════════════════════════════════════════════════════
# DIAGNOSIS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  DIAGNOSIS")
print("=" * 80)

print("\nSTRENGTHS:")
for strat in strats_list:
    st = [t for t in all_trades if t.strategy == strat]
    if st and len([t for t in st if t.win]) / len(st) >= 0.60:
        print(f"  - {strat}: {len([t for t in st if t.win])/len(st):.0%} win rate ({len(st)} trades)")

for regime in regimes_list:
    rt = [t for t in all_trades if t.regime_label == regime]
    if rt and len(rt) >= 5:
        wr = len([t for t in rt if t.win]) / len(rt)
        if wr >= 0.60:
            print(f"  - {regime} regime: {wr:.0%} win rate ({len(rt)} trades)")

print("\nWEAKNESSES:")
for strat in strats_list:
    st = [t for t in all_trades if t.strategy == strat]
    if st and len([t for t in st if t.win]) / len(st) < 0.50:
        print(f"  - {strat}: only {len([t for t in st if t.win])/len(st):.0%} win rate ({len(st)} trades)")

for year in range(2010, 2024):
    yt = [t for t in all_trades if t.entry_date.year == year]
    if yt:
        sh = _sharpe(yt)
        if sh < 0:
            print(f"  - {year}: negative Sharpe ({sh:.2f})")

print("\nSPECIFIC CONCERNS:")
# Stop-loss frequency
sl_trades = [t for t in all_trades if t.exit_reason and "stop_loss" in t.exit_reason]
if sl_trades:
    sl_pct = len(sl_trades) / len(all_trades)
    print(f"  - Stop-loss exits: {sl_pct:.0%} of all trades ({len(sl_trades)} trades)")

# Small sample sizes in cross-tab
for strat in strats_list:
    for regime in regimes_list:
        cell = [t for t in all_trades if t.strategy == strat and t.regime_label == regime]
        if 0 < len(cell) < 5:
            print(f"  - Small sample: {strat} × {regime} has only {len(cell)} trades")

# Strategy dominance
strat_counts = defaultdict(int)
for t in all_trades:
    strat_counts[t.strategy] += 1
dom = max(strat_counts, key=strat_counts.get)
dom_pct = strat_counts[dom] / len(all_trades)
if dom_pct > 0.50:
    print(f"  - {dom} dominates with {dom_pct:.0%} of all trades")


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  BACKTEST SUMMARY")
print("=" * 80)

for label, trades in [("PRE-COVID (2010-01 to 2020-02)", pre_trades),
                       ("POST-COVID (2020-02 to 2023-12)", post_trades),
                       ("COMBINED (2010-2023)", all_trades)]:
    s = _window_stats(trades, label)
    if not s:
        continue
    print(f"\n{label}:")
    print(f"  Trades:                {s['n']}")
    print(f"  Win rate:              {s['win_rate']:.1%}")
    print(f"  Sharpe:                {s['sharpe']:.2f}")
    print(f"  Profit factor:         {s['pf']:.2f}")
    print(f"  Total P&L/contract:    ${s['total_pnl_contract']:,.0f}")
    print(f"  Max drawdown:          {s['max_dd']:.2%}")
    print(f"  Annualized return:     {s['ann_ret']:.2f}%")

print()
