"""V11: Beta-Scaled Dynamic State Architecture.

No fixed baseline weights. Allocation determined by sum of IVV+QQQ Faber scores
with beta-scaled signal-squared composition formula. Asset-specific leverage when
score == 3. Defensive pool: DBMF unconditional + VGLT/IAU/DBC Faber-conditioned.
Daily per-asset circuit breaker identical to V9.

Pass criteria: improve simultaneously on Baseline AND V9 AND beat QQQ from 2013.
"""

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from dotenv import load_dotenv; load_dotenv()

import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

from taa.data import load_daily_etf_returns

SMA_PERIODS = [126, 200, 252]
SSO_EXP = 0.0089
QLD_EXP = 0.0095

# ── V11 allocation table ────────────────────────────────────────────────────
# (ivv_score, qqq_score) → (ivv_w, ivv_lev, qqq_w, qqq_lev, defense_w)
V11_TABLE = {
    (3, 3): (0.40, True,  0.60, True,  0.00),
    (3, 2): (0.69, True,  0.31, False, 0.00),
    (2, 3): (0.23, False, 0.77, True,  0.00),
    (2, 2): (0.35, False, 0.35, False, 0.30),
    (3, 1): (0.65, False, 0.00, False, 0.35),
    (1, 3): (0.00, False, 0.65, False, 0.35),
    (3, 0): (0.26, False, 0.00, False, 0.74),
    (0, 3): (0.00, False, 0.26, False, 0.74),
    (2, 1): (0.26, False, 0.04, False, 0.70),
    (1, 2): (0.06, False, 0.24, False, 0.70),
    (2, 0): (0.10, False, 0.00, False, 0.90),
    (0, 2): (0.00, False, 0.10, False, 0.90),
    (1, 1): (0.06, False, 0.04, False, 0.90),
    (1, 0): (0.00, False, 0.00, False, 1.00),
    (0, 1): (0.00, False, 0.00, False, 1.00),
    (0, 0): (0.00, False, 0.00, False, 1.00),
}

DEFENSIVE_ASSETS = ["DBMF", "VGLT", "IAU", "DBC"]
ALL_ASSETS = ["IVV", "QQQ", "VGLT", "IAU", "DBC", "DBMF", "cash"]


# ── Data loading ────────────────────────────────────────────────────────────

def load_data():
    import yfinance as yf

    daily_ret = load_daily_etf_returns()
    keep_cols = [c for c in daily_ret.columns if c in ["IVV", "QQQ", "VGLT", "IAU", "DBC", "cash"]]
    daily_ret = daily_ret[keep_cols]

    from fredapi import Fred
    rfr_daily = pd.Series(0.0, index=daily_ret.index)
    key = os.environ.get("FRED_API_KEY")
    if key:
        tb = Fred(api_key=key).get_series("DTB3", observation_start="1998-01-01")
        tb.index = pd.to_datetime(tb.index)
        rfr_daily = (tb / 100 / 252).reindex(daily_ret.index, method="ffill").fillna(0)

    ticker_map = {"IVV": "SPY", "QQQ": "QQQ", "VGLT": "TLT", "IAU": "GLD", "DBC": "DBC"}
    dp = {}
    for our, ticker in ticker_map.items():
        d = yf.download(ticker, start="1998-01-01", progress=False)
        if d is not None and not d.empty:
            p = d["Close"]
            if hasattr(p, "columns"): p = p.iloc[:, 0]
            p.index = pd.to_datetime(p.index).tz_localize(None)
            dp[our] = p
    dpdf = pd.DataFrame(dp).sort_index()
    daily_smas = {p: dpdf.rolling(p, min_periods=p).mean() for p in SMA_PERIODS}

    actual_lev = {}
    for ticker in ["SSO", "QLD"]:
        d = yf.download(ticker, start="2006-01-01", progress=False, auto_adjust=True)
        if d is not None and not d.empty:
            p = d["Close"]
            if hasattr(p, "columns"): p = p.iloc[:, 0]
            p.index = pd.to_datetime(p.index).tz_localize(None)
            actual_lev[ticker] = p.pct_change().dropna()
    both_start = max(actual_lev.get("SSO", pd.Series()).index.min(),
                     actual_lev.get("QLD", pd.Series()).index.min()) \
        if "SSO" in actual_lev and "QLD" in actual_lev else pd.Timestamp("2099-01-01")

    # DBMF actual returns (May 2019 onward). Pre-2019: use T-bills as proxy.
    dbmf_ret = pd.Series(dtype=float)
    d = yf.download("DBMF", start="2019-01-01", progress=False, auto_adjust=True)
    if d is not None and not d.empty:
        p = d["Close"]
        if hasattr(p, "columns"): p = p.iloc[:, 0]
        p.index = pd.to_datetime(p.index).tz_localize(None)
        dbmf_ret = p.pct_change().dropna()
    dbmf_inception = dbmf_ret.index.min() if len(dbmf_ret) > 0 else pd.Timestamp("2019-05-08")

    return daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, dbmf_ret, dbmf_inception


def asset_score(day, asset, dpdf, smas):
    if asset not in dpdf.columns: return 0
    p = dpdf.loc[:day, asset]
    if len(p) == 0 or pd.isna(p.iloc[-1]): return 0
    price = p.iloc[-1]; sc = 0
    for per in SMA_PERIODS:
        s = smas[per].loc[:day, asset]
        if len(s) > 0 and pd.notna(s.iloc[-1]) and price > s.iloc[-1]: sc += 1
    return sc


def check_breach(day, asset, dpdf, smas):
    if asset not in dpdf.columns: return False
    p = dpdf.loc[:day, asset]
    if len(p) == 0: return False
    price = p.iloc[-1]; b = 0
    for per in SMA_PERIODS:
        s = smas[per].loc[:day, asset]
        if len(s) > 0 and pd.notna(s.iloc[-1]) and price < s.iloc[-1]: b += 1
    return b >= 3


def lev_ret(underlying_ret, rfr, expense, day, actual_lev, ticker, both_start):
    if day >= both_start and ticker in actual_lev:
        r = float(actual_lev[ticker].get(day, np.nan))
        if not np.isnan(r): return r
    return 2.0 * underlying_ret - rfr - expense / 252


# ── V11 runner ──────────────────────────────────────────────────────────────

def run_v11(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
            dbmf_ret, dbmf_inception, start_date, capture_diag=False):
    bt_start = pd.Timestamp(start_date)
    common_start = max(daily_ret.dropna(how="all").index.min(), bt_start)
    trading_days = daily_ret.loc[common_start:pd.Timestamp("2026-03-31")].index

    port = {}
    cb_count = 0
    diag = {"states": [], "monthly": []}  # For diagnostics

    # Per-month state
    ivv_w = qqq_w = def_w = 0.0
    ivv_lev = qqq_lev = False
    delev_ivv = delev_qqq = False
    def_assets = []  # active defensive list
    cur_scores = {}
    sum_score = 0
    month_start_date = None
    month_returns_accum = 1.0

    for day in trading_days:
        dr = daily_ret.loc[day]

        is_ms = (day == trading_days[0] or day.month != trading_days[trading_days.get_loc(day) - 1].month)

        if is_ms:
            # Save prior month diagnostics
            if capture_diag and month_start_date is not None:
                diag["monthly"].append({
                    "month": month_start_date,
                    "ivv_score": cur_scores.get("IVV", 0),
                    "qqq_score": cur_scores.get("QQQ", 0),
                    "vglt_score": cur_scores.get("VGLT", 0),
                    "iau_score": cur_scores.get("IAU", 0),
                    "dbc_score": cur_scores.get("DBC", 0),
                    "sum": sum_score,
                    "ivv_w": ivv_w, "qqq_w": qqq_w, "def_w": def_w,
                    "ivv_lev_start": ivv_lev or delev_ivv,  # was leveraged at start
                    "qqq_lev_start": qqq_lev or delev_ivv,
                    "defs": list(def_assets),
                    "ret": month_returns_accum - 1.0,
                })

            # Reset CB flags + fresh allocation
            delev_ivv = delev_qqq = False
            month_returns_accum = 1.0
            month_start_date = day
            prior = trading_days[trading_days < day]
            sd = prior[-1] if len(prior) > 0 else day

            # Compute scores
            cur_scores = {}
            for a in ["IVV", "QQQ", "VGLT", "IAU", "DBC"]:
                cur_scores[a] = asset_score(sd, a, dpdf, daily_smas)
            sum_score = cur_scores["IVV"] + cur_scores["QQQ"]

            # Allocation from table
            ivv_w, ivv_lev, qqq_w, qqq_lev, def_w = V11_TABLE[(cur_scores["IVV"], cur_scores["QQQ"])]

            # Defensive pool: DBMF always, VGLT/IAU/DBC if score >= 2
            def_assets = ["DBMF"]
            if cur_scores["VGLT"] >= 2: def_assets.append("VGLT")
            if cur_scores["IAU"] >= 2: def_assets.append("IAU")
            if cur_scores["DBC"] >= 2: def_assets.append("DBC")

            if capture_diag:
                diag["states"].append({"month": day, "sum": sum_score,
                                       "ivv": cur_scores["IVV"], "qqq": cur_scores["QQQ"],
                                       "ivv_w_alloc": ivv_w, "qqq_w_alloc": qqq_w})

        # Daily circuit breaker (per asset)
        if ivv_lev and not delev_ivv:
            if check_breach(day, "IVV", dpdf, daily_smas):
                ivv_lev = False; delev_ivv = True; cb_count += 1
        if qqq_lev and not delev_qqq:
            if check_breach(day, "QQQ", dpdf, daily_smas):
                qqq_lev = False; delev_qqq = True; cb_count += 1

        # Compute daily return
        rfr = float(rfr_daily.get(day, 0.0))
        ret = 0.0

        # Equity sleeve
        if ivv_w > 0:
            ivv_underlying = float(dr.get("IVV", 0.0)) if pd.notna(dr.get("IVV", np.nan)) else 0.0
            if ivv_lev:
                ret += ivv_w * lev_ret(ivv_underlying, rfr, SSO_EXP, day, actual_lev, "SSO", both_start)
            else:
                ret += ivv_w * ivv_underlying
        if qqq_w > 0:
            qqq_underlying = float(dr.get("QQQ", 0.0)) if pd.notna(dr.get("QQQ", np.nan)) else 0.0
            if qqq_lev:
                ret += qqq_w * lev_ret(qqq_underlying, rfr, QLD_EXP, day, actual_lev, "QLD", both_start)
            else:
                ret += qqq_w * qqq_underlying

        # Defensive pool
        if def_w > 0 and len(def_assets) > 0:
            per = def_w / len(def_assets)
            for da in def_assets:
                if da == "DBMF":
                    if day >= dbmf_inception and day in dbmf_ret.index:
                        r = float(dbmf_ret.loc[day])
                    else:
                        r = rfr  # T-bill proxy pre-inception
                    ret += per * r
                else:
                    r = float(dr.get(da, 0.0)) if pd.notna(dr.get(da, np.nan)) else 0.0
                    ret += per * r

        port[day] = ret
        month_returns_accum *= (1 + ret)

    return pd.Series(port).sort_index(), cb_count, diag


# ── Baseline runner (Faber-Sweep-40 production v5) ──────────────────────────

def run_baseline(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                 dbmf_ret, dbmf_inception, start_date):
    """Production v5 from project status: Faber-Sweep-40 + DBMF 50/50 cash sub."""
    BL = {"IVV": 0.45, "QQQ": 0.25, "VGLT": 0.05, "IAU": 0.10, "DBC": 0.05, "cash": 0.10}
    bt_start = pd.Timestamp(start_date)
    common_start = max(daily_ret.dropna(how="all").index.min(), bt_start)
    trading_days = daily_ret.loc[common_start:pd.Timestamp("2026-03-31")].index

    port = {}; cb_count = 0
    weights = {}; freed_eq = 0.0; lev = False; delevered = False
    cur_scores = {}

    for day in trading_days:
        dr = daily_ret.loc[day]
        is_ms = (day == trading_days[0] or day.month != trading_days[trading_days.get_loc(day) - 1].month)

        if is_ms:
            delevered = False
            prior = trading_days[trading_days < day]
            sd = prior[-1] if len(prior) > 0 else day
            cur_scores = {a: asset_score(sd, a, dpdf, daily_smas) for a in ["IVV", "QQQ", "VGLT", "IAU", "DBC"]}

            weights = {}; freed_eq = 0.0; freed_cash = 0.0
            for a in ["IVV", "QQQ"]:
                bw = BL[a]; sc = cur_scores[a]
                if sc >= 3: weights[a] = bw
                elif sc == 2: weights[a] = bw * 0.70; freed_eq += bw * 0.30
                else: weights[a] = 0; freed_eq += bw
            for a in ["VGLT", "IAU", "DBC"]:
                bw = BL[a]; sc = cur_scores[a]
                if sc >= 3: weights[a] = bw
                elif sc == 2: weights[a] = bw * 0.70; freed_cash += bw * 0.30
                else: weights[a] = 0; freed_cash += bw
            weights["cash"] = BL["cash"] + freed_cash + freed_eq * 0.5
            weights["DBMF"] = freed_eq * 0.5

            lev = cur_scores["IVV"] >= 3 and cur_scores["QQQ"] >= 3

        if lev and not delevered:
            if check_breach(day, "IVV", dpdf, daily_smas) or check_breach(day, "QQQ", dpdf, daily_smas):
                lev = False; delevered = True; cb_count += 1

        rfr = float(rfr_daily.get(day, 0.0))
        ret = 0.0
        for a, w in weights.items():
            if w <= 0.0001: continue
            if a == "cash":
                ret += w * rfr
            elif a == "DBMF":
                if day >= dbmf_inception and day in dbmf_ret.index:
                    ret += w * float(dbmf_ret.loc[day])
                else:
                    ret += w * rfr
            elif a == "IVV":
                ivv_u = float(dr.get("IVV", 0.0)) if pd.notna(dr.get("IVV", np.nan)) else 0.0
                if lev:
                    ret += w * lev_ret(ivv_u, rfr, SSO_EXP, day, actual_lev, "SSO", both_start)
                else:
                    ret += w * ivv_u
            elif a == "QQQ":
                qqq_u = float(dr.get("QQQ", 0.0)) if pd.notna(dr.get("QQQ", np.nan)) else 0.0
                if lev:
                    ret += w * lev_ret(qqq_u, rfr, QLD_EXP, day, actual_lev, "QLD", both_start)
                else:
                    ret += w * qqq_u
            else:
                r = float(dr.get(a, 0.0)) if pd.notna(dr.get(a, np.nan)) else 0.0
                ret += w * r
        port[day] = ret

    return pd.Series(port).sort_index(), cb_count


# ── V9 runner: QLD + IVV guard ──────────────────────────────────────────────

def run_v9(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, start_date):
    bt_start = pd.Timestamp(start_date)
    common_start = max(daily_ret.dropna(how="all").index.min(), bt_start)
    trading_days = daily_ret.loc[common_start:pd.Timestamp("2026-03-31")].index

    port = {}; cb_count = 0
    weights = {}; lev = False; delevered = False

    for day in trading_days:
        dr = daily_ret.loc[day]
        is_ms = (day == trading_days[0] or day.month != trading_days[trading_days.get_loc(day) - 1].month)

        if is_ms:
            delevered = False
            prior = trading_days[trading_days < day]
            sd = prior[-1] if len(prior) > 0 else day
            sc_q = asset_score(sd, "QQQ", dpdf, daily_smas)
            sc_i = asset_score(sd, "IVV", dpdf, daily_smas)
            if sc_q >= 3:
                if sc_i <= 1:
                    weights = {"QQQ": 1.0}; lev = False
                else:
                    weights = {"QLD": 1.0}; lev = True
            elif sc_q == 2:
                weights = {"QQQ": 0.70, "cash": 0.30}; lev = False
            else:
                weights = {"cash": 1.0}; lev = False

        if lev and not delevered:
            if check_breach(day, "QQQ", dpdf, daily_smas):
                lev = False; delevered = True; cb_count += 1
                weights = {"QQQ": 1.0}

        rfr = float(rfr_daily.get(day, 0.0))
        ret = 0.0
        for a, w in weights.items():
            if a == "cash": ret += w * rfr
            elif a == "QQQ":
                qu = float(dr.get("QQQ", 0.0)) if pd.notna(dr.get("QQQ", np.nan)) else 0.0
                ret += w * qu
            elif a == "QLD":
                qu = float(dr.get("QQQ", 0.0)) if pd.notna(dr.get("QQQ", np.nan)) else 0.0
                ret += w * lev_ret(qu, rfr, QLD_EXP, day, actual_lev, "QLD", both_start)
        port[day] = ret

    return pd.Series(port).sort_index(), cb_count


# ── Metrics ─────────────────────────────────────────────────────────────────

def cagr(s):
    if len(s) < 20: return np.nan
    return (1 + s).prod() ** (252 / len(s)) - 1

def max_dd(s):
    cum = (1 + s).cumprod()
    return ((cum - cum.expanding().max()) / cum.expanding().max()).min()

def sharpe_r(s, rf=0.0):
    ar = s.mean() * 252; av = s.std() * np.sqrt(252)
    return (ar - rf) / av if av > 0 else 0

def sortino_r(s):
    ar = s.mean() * 252; neg = s[s < 0]
    ds = neg.std() * np.sqrt(252) if len(neg) > 10 else s.std() * np.sqrt(252)
    return ar / ds if ds > 0 else 0

def calmar_r(s):
    ar = s.mean() * 252; dd = max_dd(s)
    return ar / abs(dd) if dd != 0 else 0

def dca_terminal(s_monthly, start=21000, contrib=700):
    val = start
    for i, r in enumerate(s_monthly):
        if i > 0: val = val * (1 + r) + contrib
        else: val += contrib
    return val


def metrics_row(name, s, cbc=None):
    c = cagr(s); v = s.std() * np.sqrt(252)
    sh = sharpe_r(s); so = sortino_r(s); dd = max_dd(s); cl = calmar_r(s)
    t = (1 + s).cumprod().iloc[-1]
    sm = s.resample("MS").apply(lambda x: (1 + x).prod() - 1)
    dca = dca_terminal(sm)
    cb_str = f"{cbc:>3}" if cbc is not None else "  -"
    return f"  {name:<22} {c:>6.2%} {v:>6.2%} {sh:>7.3f} {so:>8.3f} {dd:>6.1%} {cl:>7.2f} ${t:>8.2f} ${dca/1e6:>7.2f}M {cb_str}"


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 140)
    print("  V11 BETA-SCALED DYNAMIC STATE — FULL BACKTEST")
    print("=" * 140)

    print("\n  Loading data...")
    daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, dbmf_ret, dbmf_inception = load_data()
    print(f"  Daily returns: {daily_ret.index.min().date()} → {daily_ret.index.max().date()}")
    print(f"  DBMF inception: {dbmf_inception.date()} (pre-inception → T-bills)")
    print(f"  SSO/QLD actual: {both_start.date()}")

    # Run all strategies from 2002
    print("\n  Running V11...")
    v11_full, v11_cb, v11_diag = run_v11(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                                          dbmf_ret, dbmf_inception, "2002-01-01", capture_diag=True)
    print("  Running Baseline (Faber-Sweep-40 v5)...")
    bl_full, bl_cb = run_baseline(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                                   dbmf_ret, dbmf_inception, "2002-01-01")
    print("  Running V9 (QLD+IVVguard)...")
    v9_full, v9_cb = run_v9(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, "2002-01-01")

    qqq_full = daily_ret["QQQ"].loc["2002-01-01":"2026-03-31"].dropna()
    ivv_full = daily_ret["IVV"].loc["2002-01-01":"2026-03-31"].dropna()

    # ── TABLE 1: Core metrics ──
    print(f"\n{'=' * 140}")
    print("  TABLE 1: CORE METRICS (2002-2026)")
    print(f"{'=' * 140}")
    print(f"\n  {'Strategy':<22} {'CAGR':>7} {'Vol':>7} {'Sharpe':>7} {'Sortino':>8} "
          f"{'MaxDD':>7} {'Calmar':>7} {'Term$1':>9} {'DCA$700':>9} {'CB':>4}")
    print(f"  {'-' * 22} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 8} {'-' * 7} {'-' * 7} {'-' * 9} {'-' * 9} {'-' * 4}")
    print(metrics_row("V11 Beta-Scaled", v11_full, v11_cb))
    print(metrics_row("Baseline (Sweep-40)", bl_full, bl_cb))
    print(metrics_row("V9 QLD+IVVguard", v9_full, v9_cb))
    print(metrics_row("QQQ B&H", qqq_full))
    print(metrics_row("IVV B&H", ivv_full))

    # ── TABLE 2: Start-date sensitivity ──
    print(f"\n{'=' * 140}")
    print("  TABLE 2: CAGR BY START DATE")
    print(f"{'=' * 140}")
    start_dates = ["2002-01-01", "2007-01-01", "2010-01-01", "2013-01-01", "2019-01-01"]
    print(f"\n  {'Strategy':<22}" + "".join(f"{sd[:4]:>10}" for sd in start_dates))
    print(f"  {'-' * 22}" + "".join(f" {'-' * 9}" for _ in start_dates))

    cagr_2013 = {}
    for name, runner_args in [
        ("V11 Beta-Scaled", ("v11",)),
        ("Baseline", ("bl",)),
        ("V9 QLD+IVVguard", ("v9",)),
    ]:
        row = f"  {name:<22}"
        for sd in start_dates:
            if runner_args[0] == "v11":
                s, _, _ = run_v11(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                                  dbmf_ret, dbmf_inception, sd)
            elif runner_args[0] == "bl":
                s, _ = run_baseline(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start,
                                    dbmf_ret, dbmf_inception, sd)
            else:
                s, _ = run_v9(daily_ret, dpdf, daily_smas, rfr_daily, actual_lev, both_start, sd)
            c = cagr(s)
            row += f"{c:>10.2%}"
            if sd == "2013-01-01": cagr_2013[name] = (s, c)
        print(row)

    # QQQ row
    row = f"  {'QQQ B&H':<22}"
    for sd in start_dates:
        qs = qqq_full[qqq_full.index >= pd.Timestamp(sd)]
        row += f"{cagr(qs):>10.2%}"
    print(row)
    qqq_2013_cagr = cagr(qqq_full[qqq_full.index >= pd.Timestamp("2013-01-01")])

    # ── TABLE 3: Sub-period CAGR ──
    print(f"\n{'=' * 140}")
    print("  TABLE 3: SUB-PERIOD CAGR")
    print(f"{'=' * 140}")
    sub = [
        ("Dot-com 02-03/03",   "2002-01-01", "2003-03-31"),
        ("Pre-GFC 03/04-07/10","2003-04-01", "2007-10-31"),
        ("GFC 07/11-09/03",    "2007-11-01", "2009-03-31"),
        ("Recovery 09-12",     "2009-04-01", "2012-12-31"),
        ("Bull 13-21",         "2013-01-01", "2021-12-31"),
        ("2022 bear",          "2022-01-01", "2022-12-31"),
        ("2023-26",            "2023-01-01", "2026-03-31"),
    ]
    print(f"\n  {'Period':<22}{'V11':>10}{'Baseline':>11}{'V9':>10}{'QQQ B&H':>11}")
    print(f"  {'-' * 22}{'-' * 10:>10} {'-' * 10:>10} {'-' * 9:>9} {'-' * 10:>10}")
    for label, cs, ce in sub:
        row = f"  {label:<22}"
        for s in [v11_full, bl_full, v9_full, qqq_full]:
            sp = s[(s.index >= pd.Timestamp(cs)) & (s.index <= pd.Timestamp(ce))]
            row += f"{cagr(sp):>10.2%} " if len(sp) > 20 else f"{'N/A':>10} "
        print(row)

    # ── TABLE 4: State occupancy ──
    print(f"\n{'=' * 140}")
    print("  TABLE 4: V11 STATE OCCUPANCY (% of months by sum-score)")
    print(f"{'=' * 140}")
    states_df = pd.DataFrame(v11_diag["states"])
    total_m = len(states_df)
    print(f"\n  Total months: {total_m}")
    print(f"\n  {'Sum-score':<12}{'Months':>10}{'Pct':>10}{'Description':>40}")
    descs = {6: "full conviction (both 3/3)", 5: "partial leveraged (one 2)",
             4: "delevered equity", 3: "mostly defensive", 2: "near exit",
             1: "full defensive", 0: "full defensive"}
    for s in [6, 5, 4, 3, 2, 1, 0]:
        n = (states_df["sum"] == s).sum()
        print(f"  Sum {s:<8}{n:>10}{n / total_m:>10.1%}    {descs[s]:>36}")

    # ── TABLE 5: Defensive utilization ──
    print(f"\n{'=' * 140}")
    print("  TABLE 5: DEFENSIVE POOL UTILIZATION (months where def_w > 0)")
    print(f"{'=' * 140}")
    def_months = [m for m in v11_diag["monthly"] if m["def_w"] > 0]
    if def_months:
        print(f"\n  Defensive-active months: {len(def_months)}")
        print(f"  DBMF active:  {sum(1 for m in def_months if 'DBMF' in m['defs'])}/{len(def_months)} (always)")
        print(f"  VGLT active:  {sum(1 for m in def_months if 'VGLT' in m['defs'])}/{len(def_months)}")
        print(f"  IAU  active:  {sum(1 for m in def_months if 'IAU' in m['defs'])}/{len(def_months)}")
        print(f"  DBC  active:  {sum(1 for m in def_months if 'DBC' in m['defs'])}/{len(def_months)}")

    # ── TABLE 6: 2022 month-by-month ──
    print(f"\n{'=' * 140}")
    print("  TABLE 6: 2022 MONTH-BY-MONTH DETAIL")
    print(f"{'=' * 140}")
    bl_2022 = bl_full[(bl_full.index >= "2022-01-01") & (bl_full.index <= "2022-12-31")].resample("MS").apply(lambda x: (1+x).prod()-1)
    v9_2022 = v9_full[(v9_full.index >= "2022-01-01") & (v9_full.index <= "2022-12-31")].resample("MS").apply(lambda x: (1+x).prod()-1)
    print(f"\n  {'Month':<10}{'IVV':>5}{'QQQ':>5}{'Sum':>5}{'Eq%':>7}{'Defensives':>22}{'V11ret':>9}{'BLret':>9}{'V9ret':>9}")
    print(f"  {'-'*10}{'-'*4:>5}{'-'*4:>5}{'-'*4:>5}{'-'*6:>7}{'-'*21:>22}{'-'*8:>9}{'-'*8:>9}{'-'*8:>9}")
    for m in v11_diag["monthly"]:
        if m["month"].year == 2022:
            ts = pd.Timestamp(m["month"].year, m["month"].month, 1)
            blr = bl_2022.get(ts, np.nan); v9r = v9_2022.get(ts, np.nan)
            eqp = m["ivv_w"] + m["qqq_w"]
            defs = "+".join(m["defs"]) if m["defs"] else "-"
            print(f"  {m['month'].strftime('%Y-%m'):<10}{m['ivv_score']:>5}{m['qqq_score']:>5}"
                  f"{m['sum']:>5}{eqp:>7.0%}{defs:>22}{m['ret']:>8.2%} {blr:>8.2%} {v9r:>8.2%}")

    # ── TABLE 7: DCA dollar gap year-by-year ──
    print(f"\n{'=' * 140}")
    print("  TABLE 7: DCA TERMINAL VALUE BY YEAR-END (2013 start, $21K + $700/mo)")
    print(f"{'=' * 140}")
    print(f"\n  {'Year':<6}{'V11':>14}{'Baseline':>14}{'V9':>14}{'QQQ B&H':>14}{'V11-QQQ':>14}{'BL-QQQ':>14}")
    print(f"  {'-' * 6}{'-' * 13:>14}{'-' * 13:>14}{'-' * 13:>14}{'-' * 13:>14}{'-' * 13:>14}{'-' * 13:>14}")
    for yr in range(2013, 2027):
        end = f"{yr}-12-31"
        vals = {}
        for nm, s in [("V11", v11_full), ("BL", bl_full), ("V9", v9_full)]:
            sp = s[(s.index >= "2013-01-01") & (s.index <= end)]
            sm = sp.resample("MS").apply(lambda x: (1+x).prod()-1)
            vals[nm] = dca_terminal(sm)
        qs = qqq_full[(qqq_full.index >= "2013-01-01") & (qqq_full.index <= end)]
        qm = qs.resample("MS").apply(lambda x: (1+x).prod()-1)
        vals["QQQ"] = dca_terminal(qm)
        print(f"  {yr:<6}${vals['V11']/1e3:>12.0f}K ${vals['BL']/1e3:>12.0f}K "
              f"${vals['V9']/1e3:>12.0f}K ${vals['QQQ']/1e3:>12.0f}K "
              f"${(vals['V11']-vals['QQQ'])/1e3:>12.0f}K ${(vals['BL']-vals['QQQ'])/1e3:>12.0f}K")

    # ── TABLE 8: Beta tilt validation ──
    print(f"\n{'=' * 140}")
    print("  TABLE 8: BETA TILT VALIDATION")
    print(f"{'=' * 140}")
    sum6 = states_df[states_df["sum"] == 6]
    sum2 = states_df[states_df["sum"] == 2]
    if len(sum6) > 0:
        avg_q6 = sum6["qqq_w_alloc"].mean(); avg_i6 = sum6["ivv_w_alloc"].mean()
        print(f"\n  Sum=6 months ({len(sum6)}): avg QQQ alloc {avg_q6:.0%}, avg IVV alloc {avg_i6:.0%} "
              f"→ {'QQQ tilted ✓' if avg_q6 > avg_i6 else 'NOT tilted ✗'}")
    if len(sum2) > 0:
        avg_q2 = sum2["qqq_w_alloc"].mean(); avg_i2 = sum2["ivv_w_alloc"].mean()
        print(f"  Sum=2 months ({len(sum2)}): avg QQQ alloc {avg_q2:.1%}, avg IVV alloc {avg_i2:.1%} "
              f"→ {'IVV tilted ✓' if avg_i2 > avg_q2 else 'NOT tilted ✗'}")

    # ── TABLE 9: Pass/fail verdict ──
    print(f"\n{'=' * 140}")
    print("  TABLE 9: PASS / FAIL CRITERIA")
    print(f"{'=' * 140}")

    v11_2013_s, v11_2013_c = cagr_2013["V11 Beta-Scaled"]
    bl_2013_s, bl_2013_c = cagr_2013["Baseline"]
    v9_2013_s, v9_2013_c = cagr_2013["V9 QLD+IVVguard"]

    v11_dd = max_dd(v11_full); v9_dd = max_dd(v9_full); bl_dd = max_dd(bl_full)
    v11_sh = sharpe_r(v11_full); v9_sh = sharpe_r(v9_full); bl_sh = sharpe_r(bl_full)
    v11_t = (1+v11_full).cumprod().iloc[-1]; bl_t = (1+bl_full).cumprod().iloc[-1]

    # DCA gap vs QQQ — peak negative
    def peak_dca_gap(s, qqq):
        worst = 0
        for yr in range(2013, 2027):
            end = f"{yr}-12-31"
            sp = s[(s.index >= "2013-01-01") & (s.index <= end)]
            sm = sp.resample("MS").apply(lambda x: (1+x).prod()-1)
            sd = dca_terminal(sm)
            qs = qqq[(qqq.index >= "2013-01-01") & (qqq.index <= end)]
            qm = qs.resample("MS").apply(lambda x: (1+x).prod()-1)
            qd = dca_terminal(qm)
            gap = sd - qd
            if gap < worst: worst = gap
        return worst
    v11_gap = peak_dca_gap(v11_full, qqq_full)
    bl_gap = peak_dca_gap(bl_full, qqq_full)

    print(f"\n  vs Baseline:")
    print(f"    CAGR from 2013:    V11 {v11_2013_c:.2%} vs Baseline {bl_2013_c:.2%} → {'PASS ✓' if v11_2013_c > bl_2013_c else 'FAIL ✗'}")
    print(f"    Terminal $1:       V11 ${v11_t:.2f} vs Baseline ${bl_t:.2f} → {'PASS ✓' if v11_t >= bl_t else 'FAIL ✗'}")
    print(f"    Peak DCA gap vs QQQ: V11 ${v11_gap/1e3:.0f}K vs BL ${bl_gap/1e3:.0f}K → {'PASS ✓' if v11_gap > bl_gap else 'FAIL ✗'}")

    print(f"\n  vs V9:")
    print(f"    Max DD:            V11 {v11_dd:.1%} vs V9 {v9_dd:.1%} → {'PASS ✓' if v11_dd > v9_dd else 'FAIL ✗'}")
    print(f"    Sharpe:            V11 {v11_sh:.3f} vs V9 {v9_sh:.3f} → {'PASS ✓' if v11_sh > v9_sh else 'FAIL ✗'}")

    print(f"\n  vs QQQ B&H:")
    print(f"    CAGR from 2013:    V11 {v11_2013_c:.2%} vs QQQ {qqq_2013_cagr:.2%} → {'PASS ✓' if v11_2013_c > qqq_2013_cagr else 'FAIL ✗'}")
    qqq_dd = max_dd(qqq_full)
    print(f"    Max DD:            V11 {v11_dd:.1%} vs QQQ {qqq_dd:.1%} → {'PASS ✓' if v11_dd > qqq_dd else 'FAIL ✗'}")

    pareto_baseline = (v11_2013_c > bl_2013_c) and (v11_t >= bl_t) and (v11_gap > bl_gap)
    pareto_v9 = (v11_dd > v9_dd) and (v11_sh > v9_sh)
    pareto_qqq = (v11_2013_c > qqq_2013_cagr) and (v11_dd > qqq_dd)

    print(f"\n  OVERALL: vs Baseline {'PASS' if pareto_baseline else 'FAIL'} | "
          f"vs V9 {'PASS' if pareto_v9 else 'FAIL'} | vs QQQ {'PASS' if pareto_qqq else 'FAIL'}")
    if pareto_baseline and pareto_v9 and pareto_qqq:
        print("  → V11 IS A PARETO IMPROVEMENT ON ALL THREE")
    else:
        print("  → V11 is NOT a strict Pareto improvement. Honest tradeoff required.")

    print()

    # Save results object for vault export
    return {
        "v11_full": v11_full, "bl_full": bl_full, "v9_full": v9_full,
        "qqq_full": qqq_full, "ivv_full": ivv_full,
        "v11_cb": v11_cb, "bl_cb": bl_cb, "v9_cb": v9_cb,
        "v11_diag": v11_diag, "states_df": states_df,
        "cagr_2013": cagr_2013, "qqq_2013_cagr": qqq_2013_cagr,
        "v11_gap": v11_gap, "bl_gap": bl_gap,
    }


if __name__ == "__main__":
    main()
