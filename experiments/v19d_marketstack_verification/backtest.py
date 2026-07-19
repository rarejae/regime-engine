"""V19d verification backtest on Marketstack data — 2000 to present.

Verification design:
- Runs the EXACT production implementation (run_v19d_full from
  experiments/v19d_final) on independently sourced data. No strategy
  code is reimplemented here.
- Marketstack plan history starts ~2016-07. Pre-splice data comes from
  yfinance (the source used by all prior backtests); post-splice data
  comes from Marketstack. Daily-return correlation on the 2016-2026
  overlap validates the two sources agree before splicing.
- Quality gate: rerun 2002-01 → 2026-03 must reproduce the locked V19d
  numbers (17.27% CAGR, 0.866 Sharpe, -25.1% MaxDD) within tolerance.
- Headline run: 2000-01 → latest Marketstack close.

Proxies (same as all prior experiments): IVV→SPY, IAU→GLD, VGLT→TLT.
GLD inception is Nov 2004 — before that the gold sleeve scores 0 and
sits in cash, consistent with the vault's dot-com-era gold findings.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from dotenv import load_dotenv; load_dotenv()

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from data.sources.marketstack import load_adj_closes
from experiments.v11_beta_scaled.backtest import (
    SMA_PERIODS, cagr, max_dd, sharpe_r, sortino_r, calmar_r, dca_terminal,
)
import experiments.v19d_final.backtest as v19d_mod
from experiments.v19d_final.backtest import run_v19d_full, run_6040

SPLICE_DATE = pd.Timestamp("2016-08-01")
TICKER_MAP = {"IVV": "SPY", "QQQ": "QQQ", "IAU": "GLD", "VGLT": "TLT"}
LEV_TICKERS = ["QLD", "SSO"]
MIN_OVERLAP_CORR = 0.99

# Locked production numbers (2002-01 → 2026-03) from V19D_FINAL_BACKTEST
LOCKED = {"CAGR": 0.1727, "Sharpe": 0.866, "MaxDD": -0.251}


def load_yf_prices(tickers, start="1998-01-01"):
    """yfinance closes with retries and a local parquet cache (yf rate-limits)."""
    import time
    import yfinance as yf
    cache = Path(__file__).resolve().parent / "yf_cache.parquet"
    if cache.exists():
        cached = pd.read_parquet(cache)
        if all(t in cached.columns for t in tickers):
            return cached

    frames = {}
    for t in tickers:
        p = None
        for attempt in range(4):
            d = yf.download(t, start=start, progress=False, auto_adjust=True)
            if d is not None and not d.empty:
                p = d["Close"]
                if hasattr(p, "columns"): p = p.iloc[:, 0]
                p.index = pd.to_datetime(p.index).tz_localize(None)
                break
            time.sleep(3 * (attempt + 1))
        if p is None:
            raise RuntimeError(f"yfinance returned no data for {t} after retries")
        frames[t] = p
    df = pd.DataFrame(frames).sort_index()
    df.to_parquet(cache)
    return df


def splice(yf_p: pd.Series, ms_p: pd.Series, name: str):
    """yfinance before SPLICE_DATE, Marketstack after, ratio-adjusted at the
    boundary so the series is continuous (SMA levels stay consistent)."""
    yf_p = yf_p.dropna(); ms_p = ms_p.dropna()
    if ms_p.empty:
        print(f"  {name}: no Marketstack data — yfinance only")
        return yf_p, np.nan
    overlap = yf_p.index.intersection(ms_p.index)
    overlap = overlap[overlap >= SPLICE_DATE]
    corr = np.nan
    if len(overlap) > 100:
        r_yf = yf_p.loc[overlap].pct_change().replace([np.inf, -np.inf], np.nan)
        r_ms = ms_p.loc[overlap].pct_change().replace([np.inf, -np.inf], np.nan)
        corr = r_yf.corr(r_ms)
    anchor_days = overlap[:5]
    if len(anchor_days) == 0:
        return yf_p, corr
    ratio = (yf_p.loc[anchor_days] / ms_p.loc[anchor_days]).mean()
    ms_scaled = ms_p.loc[anchor_days[0]:] * ratio
    # Fill Marketstack gaps (missing/zero-price days) from yfinance, which
    # is already in the same scale as the ratio-adjusted series.
    ms_scaled = ms_scaled.reindex(yf_p.loc[anchor_days[0]:].index.union(ms_scaled.index))
    filled = int(ms_scaled.isna().sum())
    ms_scaled = ms_scaled.fillna(yf_p)
    if filled:
        print(f"    ({name}: filled {filled} missing Marketstack days from yfinance)")
    spliced = pd.concat([yf_p[yf_p.index < anchor_days[0]], ms_scaled])
    return spliced[~spliced.index.duplicated(keep="last")].sort_index().dropna(), corr


def load_spliced_data():
    all_ms_symbols = sorted(set(TICKER_MAP.values()) | set(LEV_TICKERS))
    print("  Fetching Marketstack EOD data...")
    ms = load_adj_closes(all_ms_symbols, date_from="2015-01-01")
    print("  Fetching yfinance history (pre-splice + overlap)...")
    yf_df = load_yf_prices(all_ms_symbols)

    print("\n  Source agreement (daily-return corr on 2016+ overlap):")
    prices = {}
    corrs = {}
    for our, tk in {**TICKER_MAP, "QLD": "QLD", "SSO": "SSO"}.items():
        p, corr = splice(yf_df[tk], ms[tk] if tk in ms.columns else pd.Series(dtype=float), tk)
        prices[our] = p
        corrs[tk] = corr
        status = "✓" if (np.isnan(corr) or corr > MIN_OVERLAP_CORR) else "✗ DISAGREEMENT"
        print(f"    {tk:<5} corr={corr:.4f} → {status}" if not np.isnan(corr)
              else f"    {tk:<5} corr=n/a")
    bad = [t for t, c in corrs.items() if not np.isnan(c) and c <= MIN_OVERLAP_CORR]
    assert not bad, f"Source disagreement on {bad} — do not trust splice"

    dpdf = pd.DataFrame({k: prices[k] for k in TICKER_MAP}).sort_index()
    daily_ret = dpdf.pct_change()
    daily_smas = {p: dpdf.rolling(p, min_periods=p).mean() for p in SMA_PERIODS}

    actual_lev = {t: prices[t].pct_change().dropna() for t in LEV_TICKERS}
    both_start = max(actual_lev["QLD"].index.min(), actual_lev["SSO"].index.min())

    import os
    from fredapi import Fred
    rfr_daily = pd.Series(0.0, index=daily_ret.index)
    key = os.environ.get("FRED_API_KEY")
    if key:
        tb = Fred(api_key=key).get_series("DTB3", observation_start="1998-01-01")
        tb.index = pd.to_datetime(tb.index)
        rfr_daily = (tb / 100 / 252).reindex(daily_ret.index, method="ffill").fillna(0)

    return daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start


def metrics(s):
    sm = s.resample("MS").apply(lambda x: (1 + x).prod() - 1)
    return {"CAGR": cagr(s), "Vol": s.std() * np.sqrt(252), "Sharpe": sharpe_r(s),
            "Sortino": sortino_r(s), "MaxDD": max_dd(s), "Calmar": calmar_r(s),
            "Terminal": (1 + s).cumprod().iloc[-1], "DCA": dca_terminal(sm)}


def prow(name, m):
    print(f"  {name:<22} {m['CAGR']:>7.2%} {m['Vol']:>7.2%} {m['Sharpe']:>7.3f} "
          f"{m['Sortino']:>8.3f} {m['MaxDD']:>7.1%} {m['Calmar']:>7.2f} "
          f"${m['Terminal']:>8.2f} ${m['DCA']/1e6:>7.2f}M")


def main():
    W = 120
    print("=" * W)
    print("  V19d VERIFICATION BACKTEST — MARKETSTACK DATA, 2000 → PRESENT")
    print("=" * W)

    daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start = load_spliced_data()
    end_date = daily_ret.dropna(how="all").index.max()
    v19d_mod.END_DATE = end_date.strftime("%Y-%m-%d")
    print(f"\n  Data range: {dpdf.dropna(how='all').index.min().date()} → {end_date.date()}")
    print(f"  Actual QLD/SSO from: {both_start.date()} (simulated 2x before)")
    print(f"  Marketstack coverage: {SPLICE_DATE.date()} → {end_date.date()}; yfinance before")

    # ── Signal alignment assertion (CLAUDE.md requirement) ──
    # SMA at day t must use closes through day t only (no look-ahead).
    # Scores at month start use the prior trading day — enforced inside
    # run_v19d_full via prior[-1].
    qqq_px = dpdf["QQQ"].dropna()
    test_day = qqq_px.index[2000]
    manual = qqq_px.loc[:test_day].iloc[-126:].mean()
    sma_val = dpdf["QQQ"].rolling(126, min_periods=126).mean().loc[test_day]
    assert abs(sma_val - manual) < 1e-6, "SMA look-ahead detected"
    print(f"\n  Signal alignment: SMA-126 at {test_day.date()} matches trailing-only calc ✓")

    # ── QUALITY GATE: reproduce locked 2002-2026 numbers ──
    print(f"\n{'=' * W}\n  1. QUALITY GATE — reproduce locked V19d (2002-01 → 2026-03)\n{'=' * W}")
    v19d_mod.END_DATE = "2026-03-31"
    s_lock, cb_lock, ml_lock, _, _ = run_v19d_full(
        daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, "2002-01-01")
    m_lock = metrics(s_lock)
    print(f"\n  {'Metric':<10}{'This run':>12}{'Locked':>12}{'Delta':>10}")
    for k in ["CAGR", "Sharpe", "MaxDD"]:
        print(f"  {k:<10}{m_lock[k]:>12.4f}{LOCKED[k]:>12.4f}{m_lock[k]-LOCKED[k]:>+10.4f}")
    gate_ok = (abs(m_lock["CAGR"] - LOCKED["CAGR"]) < 0.01
               and abs(m_lock["Sharpe"] - LOCKED["Sharpe"]) < 0.03
               and abs(m_lock["MaxDD"] - LOCKED["MaxDD"]) < 0.02)
    print(f"\n  Quality gate: {'PASS ✓' if gate_ok else 'FAIL ✗ — investigate before trusting results'}")

    # ── HEADLINE RUN: 2000 → latest ──
    print(f"\n{'=' * W}\n  2. FULL RUN — 2000-01-01 → {end_date.date()}\n{'=' * W}")
    v19d_mod.END_DATE = end_date.strftime("%Y-%m-%d")
    s, cb, ml, rebal, daily_detail = run_v19d_full(
        daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, "2000-01-01")
    qqq_s = daily_ret["QQQ"].loc["2000-01-01":end_date].dropna()
    ivv_s = daily_ret["IVV"].loc["2000-01-01":end_date].dropna()
    s6040 = run_6040(daily_ret, rfr_daily, "2000-01-01")

    print(f"\n  {'Strategy':<22} {'CAGR':>7} {'Vol':>7} {'Sharpe':>7} {'Sortino':>8} "
          f"{'MaxDD':>7} {'Calmar':>7} {'Term$1':>9} {'DCA':>9}")
    print(f"  {'-'*22} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*7} {'-'*7} {'-'*9} {'-'*9}")
    for nm, series in [("V19d", s), ("QQQ B&H", qqq_s), ("IVV B&H", ivv_s), ("60/40 (TLT)", s6040)]:
        prow(nm, metrics(series))

    cbq = sum(1 for e in cb if e["asset"] == "QQQ")
    cbi = sum(1 for e in cb if e["asset"] == "IVV")
    cba = sum(1 for e in cb if e["asset"] == "IAU")
    print(f"\n  CB events: {len(cb)} total (QQQ {cbq}, IVV {cbi}, IAU {cba}) | rebalances: {rebal}")

    # ── Crisis windows (incl. dot-com, which the locked backtest missed) ──
    print(f"\n{'=' * W}\n  3. CRISIS DRAWDOWNS\n{'=' * W}")
    crises = [("Dot-com 2000-02", "2000-03-01", "2002-12-31"),
              ("GFC 07-09", "2007-11-01", "2009-03-31"),
              ("COVID 2020", "2020-02-01", "2020-04-30"),
              ("2022 bear", "2022-01-01", "2022-12-31"),
              ("2025-26 (recent)", "2025-01-01", end_date.strftime("%Y-%m-%d"))]
    print(f"\n  {'Crisis':<20}{'V19d':>10}{'QQQ B&H':>10}{'IVV B&H':>10}")
    for label, c0, c1 in crises:
        row = f"  {label:<20}"
        for series in [s, qqq_s, ivv_s]:
            w = series[(series.index >= c0) & (series.index <= c1)]
            row += f"{max_dd(w):>10.1%}" if len(w) > 20 else f"{'n/a':>10}"
        print(row)

    # ── Annual returns ──
    print(f"\n{'=' * W}\n  4. ANNUAL RETURNS\n{'=' * W}")
    print(f"\n  {'Year':<6}{'V19d':>9}{'QQQ':>9}{'IVV':>9}")
    for yr in range(2000, end_date.year + 1):
        row = f"  {yr:<6}"
        for series in [s, qqq_s, ivv_s]:
            sp = series[series.index.year == yr]
            row += f"{(1 + sp).prod() - 1:>9.2%}" if len(sp) > 0 else f"{'—':>9}"
        print(row)

    # ── Recent 24 months detail ──
    print(f"\n{'=' * W}\n  5. LAST 24 MONTHS — ALLOCATION AND RETURNS\n{'=' * W}")
    s_monthly = s.resample("MS").apply(lambda x: (1 + x).prod() - 1)
    print(f"\n  {'Month':<10}{'QQQ':>4}{'IVV':>4}{'IAU':>4}{'Pod1':>13}{'Pod2':>13}{'Gold':>6}{'EffEq':>7}{'Ret':>8}")
    for m in ml[-24:]:
        ts = pd.Timestamp(m["month"].year, m["month"].month, 1)
        mr = float(s_monthly.get(ts, np.nan))
        print(f"  {m['month'].strftime('%Y-%m'):<10}{m['qqq_sc']:>4}{m['ivv_sc']:>4}{m['iau_sc']:>4}"
              f"{m['p1_mode']:>13}{m['p2_mode']:>13}{m['gold_mode']:>6}{m['eff_equity']:>6.0%}{mr:>8.2%}")

    # ── Current state snapshot ──
    print(f"\n{'=' * W}\n  6. CURRENT STATE (as of {end_date.date()} close, Marketstack)\n{'=' * W}")
    last = dpdf.dropna(how="all").index[-1]
    print(f"\n  {'Asset':<6}{'Close':>10}{'SMA126':>10}{'SMA200':>10}{'SMA252':>10}{'Score':>7}")
    for a in ["QQQ", "IVV", "IAU"]:
        px = dpdf.loc[last, a]
        smas = [daily_smas[p].loc[last, a] for p in SMA_PERIODS]
        score = sum(px > sv for sv in smas)
        print(f"  {a:<6}{px:>10.2f}" + "".join(f"{sv:>10.2f}" for sv in smas) + f"{score:>7}")
    cur = ml[-1]
    print(f"\n  Current allocation: Pod1={cur['p1_mode']}, Pod2={cur['p2_mode']}, "
          f"Gold={cur['gold_mode']} | eff equity {cur['eff_equity']:.0%}")

    # YTD and trailing 12m
    ytd = s[s.index.year == end_date.year]
    t12 = s.iloc[-252:]
    q_ytd = qqq_s[qqq_s.index.year == end_date.year]
    print(f"\n  YTD {end_date.year}: V19d {(1+ytd).prod()-1:+.2%} vs QQQ {(1+q_ytd).prod()-1:+.2%}")
    print(f"  Trailing 12m: V19d {(1+t12).prod()-1:+.2%} vs QQQ {(1+qqq_s.iloc[-252:]).prod()-1:+.2%}")

    print(f"\n  DONE.")


if __name__ == "__main__":
    main()
