"""SPY vs QQQ relative-spread mean reversion — diagnostic + V19d overlay.

Test 1: Does the QQQ/SPY log-ratio mean-revert? If so, does buying the
        relatively cheaper index earn subsequent excess return?
Test 2: Overlay that signal on V19d: when both pods are on, tilt 15pp
        toward the laggard. Faber / CB / gold unchanged.

Uses local CSVs (no network). SPY stands in for IVV; GLD for IAU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

CSV_DIR = ROOT / "data/raw/yfinance/sfe_csv"
OUT_JSON = Path(__file__).resolve().parent / "results.json"

SMA_PERIODS = [126, 200, 252]
SSO_EXP = 0.0089
QLD_EXP = 0.0095
TILT_PP = 0.15  # 45/45 → 30/60 toward laggard
Z_LOOKBACK = 252
Z_THRESH = 1.0


def load_csv(name: str) -> pd.Series:
    p = pd.read_csv(CSV_DIR / f"{name}.csv", parse_dates=["Date"], index_col="Date")["adj_close"]
    p.index = pd.to_datetime(p.index).tz_localize(None)
    return p.sort_index().rename(name)


def load_prices() -> pd.DataFrame:
    spy = load_csv("SPY")
    qqq = load_csv("QQQ")
    gld = load_csv("GLD")
    sso = load_csv("SSO")
    qld = load_csv("QLD")
    ivv = load_csv("IVV")
    # Prefer IVV when available; splice SPY before IVV inception
    ivv_full = spy.copy().rename("IVV")
    overlap = ivv.index.intersection(ivv_full.index)
    scale = float(ivv_full.loc[overlap[0]] / ivv.loc[overlap[0]]) if len(overlap) else 1.0
    ivv_full.loc[ivv.index] = ivv * scale
    df = pd.DataFrame({"IVV": ivv_full, "QQQ": qqq, "IAU": gld.rename("IAU")}).sort_index()
    return df.dropna(subset=["IVV", "QQQ"]), sso.pct_change().dropna(), qld.pct_change().dropna()


def load_rfr(index: pd.DatetimeIndex) -> pd.Series:
    tb = pd.read_parquet(ROOT / "data/raw/fred/DTB3.parquet")
    if isinstance(tb, pd.DataFrame):
        col = tb.columns[0]
        tb = tb[col]
    tb.index = pd.to_datetime(tb.index)
    return (tb / 100 / 252).reindex(index, method="ffill").fillna(0.0)


def cagr(s: pd.Series) -> float:
    if len(s) < 20:
        return float("nan")
    return float((1 + s).prod() ** (252 / len(s)) - 1)


def max_dd(s: pd.Series) -> float:
    cum = (1 + s).cumprod()
    return float(((cum - cum.expanding().max()) / cum.expanding().max()).min())


def sharpe_r(s: pd.Series) -> float:
    av = s.std() * np.sqrt(252)
    return float((s.mean() * 252) / av) if av > 0 else 0.0


def terminal(s: pd.Series) -> float:
    return float((1 + s).cumprod().iloc[-1])


def metrics(s: pd.Series) -> dict:
    return {
        "cagr": cagr(s),
        "vol": float(s.std() * np.sqrt(252)),
        "sharpe": sharpe_r(s),
        "maxdd": max_dd(s),
        "term": terminal(s),
        "n": int(len(s)),
    }


def ou_halflife(y: pd.Series) -> float:
    y = y.dropna()
    dy = y.diff().dropna()
    x = y.shift(1).loc[dy.index]
    var = float(x.var())
    if var == 0:
        return float("nan")
    lam = float(np.cov(dy, x)[0, 1] / var)
    if lam >= 0:
        return float("inf")
    return float(-np.log(2) / lam)


def asset_score(day, asset, dpdf, smas) -> int:
    p = dpdf.loc[:day, asset]
    if len(p) == 0 or pd.isna(p.iloc[-1]):
        return 0
    price = p.iloc[-1]
    sc = 0
    for per in SMA_PERIODS:
        s = smas[per].loc[:day, asset]
        if len(s) > 0 and pd.notna(s.iloc[-1]) and price > s.iloc[-1]:
            sc += 1
    return sc


def check_breach(day, asset, dpdf, smas) -> bool:
    p = dpdf.loc[:day, asset]
    if len(p) == 0:
        return False
    price = p.iloc[-1]
    b = 0
    for per in SMA_PERIODS:
        s = smas[per].loc[:day, asset]
        if len(s) > 0 and pd.notna(s.iloc[-1]) and price < s.iloc[-1]:
            b += 1
    return b >= 3


def lev_ret(u, rfr, exp, day, actual, ticker, both_start):
    if day >= both_start and ticker in actual:
        r = float(actual[ticker].get(day, np.nan))
        if not np.isnan(r):
            return r
    return 2.0 * u - rfr - exp / 252


def zscore_log_ratio(qqq: pd.Series, ivv: pd.Series, lookback: int) -> pd.Series:
    lr = np.log(qqq / ivv)
    mu = lr.rolling(lookback, min_periods=lookback).mean()
    sd = lr.rolling(lookback, min_periods=lookback).std()
    return (lr - mu) / sd


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(1, index=index)
    return s.resample("ME").last().index.intersection(index)


def test1_spread(dpdf: pd.DataFrame) -> dict:
    qqq, ivv = dpdf["QQQ"], dpdf["IVV"]
    lr = np.log(qqq / ivv).dropna()
    z = zscore_log_ratio(qqq, ivv, Z_LOOKBACK).dropna()
    ret_q = qqq.pct_change()
    ret_i = ivv.pct_change()
    rel = (ret_q - ret_i).dropna()

    # Structural drift of the ratio
    ratio = (qqq / ivv).dropna()
    ratio_start = float(ratio.iloc[0])
    ratio_end = float(ratio.iloc[-1])

    # Half-lives
    hl_raw = ou_halflife(lr)
    hl_z = ou_halflife(z)
    hl_rel_21 = ou_halflife(rel.rolling(21).sum())

    me = month_ends(z.index)
    me = me[(me >= z.index.min()) & (me <= z.index.max())]

    horizons = {"1m": 21, "3m": 63, "6m": 126, "12m": 252}
    buckets = [
        ("QQQ rich z>+1.5", lambda v: v > 1.5),
        ("QQQ rich +1<z≤1.5", lambda v: (v > 1.0) & (v <= 1.5)),
        ("mild |z|≤1", lambda v: v.abs() <= 1.0),
        ("QQQ cheap −1.5≤z<−1", lambda v: (v < -1.0) & (v >= -1.5)),
        ("QQQ cheap z<−1.5", lambda v: v < -1.5),
    ]

    bucket_rows = []
    for name, pred in buckets:
        mask = pred(z.loc[me])
        dates = z.loc[me][mask].index
        row = {"bucket": name, "n": int(len(dates))}
        for hname, h in horizons.items():
            fwd = []
            hits = 0
            for d in dates:
                loc = rel.index.get_loc(d)
                if isinstance(loc, slice):
                    continue
                if loc + h >= len(rel):
                    continue
                # subsequent relative return QQQ - IVV
                r = float(rel.iloc[loc + 1 : loc + 1 + h].sum())
                fwd.append(r)
                # laggard wins if: QQQ rich (z>0) and r<0 (IVV catches up)
                #               or QQQ cheap (z<0) and r>0 (QQQ catches up)
                zv = float(z.loc[d])
                if (zv > 0 and r < 0) or (zv < 0 and r > 0):
                    hits += 1
            row[f"{hname}_mean_qqq_minus_ivv"] = float(np.mean(fwd)) if fwd else None
            row[f"{hname}_hit_laggard"] = (hits / len(fwd)) if fwd else None
            row[f"{hname}_n"] = len(fwd)
        bucket_rows.append(row)

    # Thresholded laggard hit rates
    thresh_rows = []
    for thresh in [0.5, 1.0, 1.5, 2.0]:
        dates = z.loc[me][z.loc[me].abs() >= thresh].index
        row = {"thresh": thresh, "n_months": int(len(dates))}
        for hname, h in horizons.items():
            hits = 0
            n = 0
            pnls = []  # laggard minus leader over horizon
            for d in dates:
                loc = rel.index.get_loc(d)
                if isinstance(loc, slice) or loc + h >= len(rel):
                    continue
                r = float(rel.iloc[loc + 1 : loc + 1 + h].sum())
                zv = float(z.loc[d])
                # pnl of long laggard / short leader = -sign(z) * (qqq-ivv)
                pnl = -np.sign(zv) * r
                pnls.append(pnl)
                n += 1
                if pnl > 0:
                    hits += 1
            row[f"{hname}_hit"] = (hits / n) if n else None
            row[f"{hname}_mean_pnl"] = float(np.mean(pnls)) if pnls else None
            row[f"{hname}_n"] = n
        thresh_rows.append(row)

    # Correlation: current z vs subsequent relative return (should be negative if MR)
    corr_rows = []
    z_me = z.loc[me]
    for hname, h in horizons.items():
        xs, ys = [], []
        for d in z_me.index:
            loc = rel.index.get_loc(d)
            if isinstance(loc, slice) or loc + h >= len(rel):
                continue
            xs.append(float(z_me.loc[d]))
            ys.append(float(rel.iloc[loc + 1 : loc + 1 + h].sum()))
        if len(xs) > 10:
            corr = float(np.corrcoef(xs, ys)[0, 1])
        else:
            corr = None
        corr_rows.append({"horizon": hname, "corr_z_vs_fwd_qqq_minus_ivv": corr, "n": len(xs)})

    # Standalone 50/50 vs laggard-tilt (unlevered, always invested)
    common = qqq.pct_change().dropna().index.intersection(ivv.pct_change().dropna().index)
    common = common.intersection(z.index)
    rq = qqq.pct_change().loc[common]
    ri = ivv.pct_change().loc[common]
    zz = z.reindex(common).ffill()

    ew = 0.5 * rq + 0.5 * ri
    # monthly signal, applied next day through month
    sig = pd.Series(0.0, index=common)  # +1 = overweight QQQ (QQQ cheap)
    last = 0.0
    for i, d in enumerate(common):
        if i == 0 or d.month != common[i - 1].month:
            zv = zz.loc[d]
            if pd.notna(zv) and zv <= -Z_THRESH:
                last = 1.0
            elif pd.notna(zv) and zv >= Z_THRESH:
                last = -1.0
            else:
                last = 0.0
        sig.iloc[i] = last
    w_q = 0.5 + TILT_PP * sig
    tilt = w_q * rq + (1 - w_q) * ri
    # long-short dollar-neutral: +1 laggard, -1 leader
    ls = -np.sign(zz.shift(1).fillna(0)) * (rq - ri)
    ls_gated = ls.where(zz.shift(1).abs() >= Z_THRESH, 0.0)

    # Subperiods for 50/50 vs tilt
    periods = [
        ("1999–2002 dot-com", "1999-03-10", "2002-12-31"),
        ("2003–2007 bull", "2003-01-01", "2007-10-31"),
        ("GFC", "2007-11-01", "2009-03-31"),
        ("2009–2012", "2009-04-01", "2012-12-31"),
        ("2013–2021 bull", "2013-01-01", "2021-12-31"),
        ("2022 bear", "2022-01-01", "2022-12-31"),
        ("2023–2026", "2023-01-01", "2026-08-14"),
        ("full", None, None),
    ]
    period_rows = []
    for label, a, b in periods:
        if a is None:
            e, t, l = ew, tilt, ls_gated
        else:
            e = ew.loc[a:b]
            t = tilt.loc[a:b]
            l = ls_gated.loc[a:b]
        if len(e) < 40:
            continue
        period_rows.append({
            "period": label,
            "ew": metrics(e),
            "tilt": metrics(t),
            "ls_ann": float(l.mean() * 252),
            "ls_sharpe": sharpe_r(l) if l.std() > 0 else 0.0,
            "tilt_minus_ew_cagr": cagr(t) - cagr(e),
        })

    # Yearly ratio path (for chart) — year-end z and ratio indexed to 1
    yearly = []
    ratio_idx = ratio / ratio.iloc[0]
    for yr, g in ratio_idx.groupby(ratio_idx.index.year):
        z_yr = z.loc[: g.index[-1]]
        yearly.append({
            "year": int(yr),
            "ratio_idx": round(float(g.iloc[-1]), 3),
            "z": round(float(z_yr.iloc[-1]), 2) if len(z_yr) else None,
        })

    occ = {
        "qqq_cheap": float((sig == 1).mean()),
        "qqq_rich": float((sig == -1).mean()),
        "neutral": float((sig == 0).mean()),
    }

    return {
        "sample": {
            "start": str(qqq.dropna().index.min().date()),
            "end": str(qqq.dropna().index.max().date()),
            "n_days": int(len(common)),
            "ratio_start": ratio_start,
            "ratio_end": ratio_end,
            "ratio_multiple": ratio_end / ratio_start,
            "corr_daily": float(rq.corr(ri)),
        },
        "halflife_days": {
            "log_ratio_raw": hl_raw,
            "zscore_252": hl_z,
            "rel_ret_21d": hl_rel_21,
        },
        "buckets": bucket_rows,
        "thresholds": thresh_rows,
        "corr_z_fwd": corr_rows,
        "standalone": {
            "ew": metrics(ew),
            "tilt": metrics(tilt),
            "ls_gated": {
                "ann": float(ls_gated.mean() * 252),
                "sharpe": sharpe_r(ls_gated),
                "hit_daily": float((ls_gated[ls_gated != 0] > 0).mean()) if (ls_gated != 0).any() else None,
            },
            "occupancy": occ,
            "periods": period_rows,
        },
        "yearly_ratio": yearly,
        "z_path_yearly": yearly,
    }


def below_all_smas(dpdf, smas, asset) -> pd.Series:
    p = dpdf[asset]
    mask = pd.Series(True, index=p.index)
    for per in SMA_PERIODS:
        mask &= p < smas[per][asset]
    return mask.fillna(False)


def run_v19d(dpdf, daily_ret, rfr_daily, actual_lev, both_start, start_date,
             mode="base", smas=None, z=None, cb=None):
    """mode: base | lag | mom
    lag = tilt toward laggard; mom = tilt toward leader (sanity / prior reject).
    """
    bt_start = pd.Timestamp(start_date)
    trading_days = daily_ret.loc[bt_start:].index
    if smas is None:
        smas = {p: dpdf.rolling(p, min_periods=p).mean() for p in SMA_PERIODS}
    if z is None:
        z = zscore_log_ratio(dpdf["QQQ"], dpdf["IVV"], Z_LOOKBACK)
    if cb is None:
        cb = {a: below_all_smas(dpdf, smas, a) for a in ["QQQ", "IVV", "IAU"]}

    nav1 = 0.45
    nav2 = 0.45
    nav_g = 0.10
    p1_mode = "cash"
    p1_lev = False
    p1_delev = False
    p2_mode = "cash"
    p2_lev = False
    p2_delev = False
    gold_mode = "cash"
    gold_delev = False
    scores = {"QQQ": 0, "IVV": 0, "IAU": 0}
    w1, w2, wg = 0.45, 0.45, 0.10
    port = {}
    tilt_log = []

    for i, day in enumerate(trading_days):
        dr = daily_ret.loc[day]
        is_ms = (i == 0 or day.month != trading_days[i - 1].month)
        if is_ms:
            p1_delev = p2_delev = gold_delev = False
            prior = trading_days[:i]
            sd = prior[-1] if len(prior) else day
            scores = {a: asset_score(sd, a, dpdf, smas) for a in ["QQQ", "IVV", "IAU"]}
            sc_q, sc_i, sc_a = scores["QQQ"], scores["IVV"], scores["IAU"]

            if sc_q >= 3:
                if sc_i <= 1:
                    p1_mode, p1_lev = "qqq", False
                else:
                    p1_mode, p1_lev = "qld", True
            elif sc_q == 2:
                p1_mode, p1_lev = "qqq_partial", False
            else:
                p1_mode, p1_lev = "cash", False

            if sc_i >= 3:
                p2_mode, p2_lev = "sso", True
            elif sc_i == 2:
                p2_mode, p2_lev = "ivv_partial", False
            else:
                p2_mode, p2_lev = "cash", False
            gold_mode = "iau" if sc_a >= 3 else "cash"

            both_on = (sc_q >= 3 and sc_i >= 3)
            zv = float(z.loc[sd]) if sd in z.index and pd.notna(z.loc[sd]) else 0.0
            w1, w2, wg = 0.45, 0.45, 0.10
            tilt_dir = "neutral"
            if mode != "base" and both_on and abs(zv) >= Z_THRESH:
                qqq_cheap = zv < 0
                toward_qqq = qqq_cheap if mode == "lag" else (not qqq_cheap)
                if toward_qqq:
                    w1, w2 = 0.45 + TILT_PP, 0.45 - TILT_PP
                    tilt_dir = "qqq"
                else:
                    w1, w2 = 0.45 - TILT_PP, 0.45 + TILT_PP
                    tilt_dir = "ivv"
            tilt_log.append({"month": str(day.date()), "z": round(zv, 2),
                             "both_on": both_on, "tilt": tilt_dir,
                             "w1": w1, "w2": w2})

            if i > 0:
                total = nav1 + nav2 + nav_g
                if total > 0:
                    drift = max(abs(nav1 / total - w1), abs(nav2 / total - w2), abs(nav_g / total - wg))
                    if drift > 0.05:
                        nav1, nav2, nav_g = total * w1, total * w2, total * wg

        if p1_lev and not p1_delev and bool(cb["QQQ"].get(day, False)):
            p1_lev, p1_delev, p1_mode = False, True, "cash"
        if p2_lev and not p2_delev and bool(cb["IVV"].get(day, False)):
            p2_lev, p2_delev, p2_mode = False, True, "cash"
        if gold_mode == "iau" and not gold_delev and bool(cb["IAU"].get(day, False)):
            gold_mode, gold_delev = "cash", True

        rfr = float(rfr_daily.get(day, 0.0))
        qqq_u = float(dr.get("QQQ", 0.0)) if pd.notna(dr.get("QQQ", np.nan)) else 0.0
        ivv_u = float(dr.get("IVV", 0.0)) if pd.notna(dr.get("IVV", np.nan)) else 0.0
        iau_u = float(dr.get("IAU", 0.0)) if pd.notna(dr.get("IAU", np.nan)) else 0.0

        if p1_mode == "qld":
            r1 = lev_ret(qqq_u, rfr, QLD_EXP, day, actual_lev, "QLD", both_start) if p1_lev else qqq_u
        elif p1_mode == "qqq":
            r1 = qqq_u
        elif p1_mode == "qqq_partial":
            r1 = 0.70 * qqq_u + 0.30 * rfr
        else:
            r1 = rfr

        if p2_mode == "sso":
            r2 = lev_ret(ivv_u, rfr, SSO_EXP, day, actual_lev, "SSO", both_start) if p2_lev else ivv_u
        elif p2_mode == "ivv":
            r2 = ivv_u
        elif p2_mode == "ivv_partial":
            r2 = 0.70 * ivv_u + 0.30 * rfr
        else:
            r2 = rfr
        rg = iau_u if gold_mode == "iau" else rfr

        nav1 *= 1 + r1
        nav2 *= 1 + r2
        nav_g *= 1 + rg
        port[day] = nav1 + nav2 + nav_g

    nav = pd.Series(port).sort_index()
    rets = nav.pct_change().dropna()
    return rets, tilt_log


def crisis_dd(s: pd.Series, a: str, b: str) -> float:
    sp = s.loc[a:b]
    if len(sp) < 5:
        return float("nan")
    return max_dd(sp)


def main():
    dpdf, sso_r, qld_r = load_prices()
    dpdf = dpdf.loc["1999-03-10":]
    daily_ret = dpdf.pct_change()
    rfr = load_rfr(daily_ret.index)
    daily_ret["cash"] = rfr
    actual = {"SSO": sso_r, "QLD": qld_r}
    both_start = max(sso_r.index.min(), qld_r.index.min())

    print("=" * 80)
    print("TEST 1 — SPY/QQQ relative spread mean reversion")
    print("=" * 80)
    t1 = test1_spread(dpdf)
    s = t1["sample"]
    print(f"  Sample: {s['start']} → {s['end']}  ({s['n_days']} days)")
    print(f"  QQQ/SPY ratio: {s['ratio_start']:.3f} → {s['ratio_end']:.3f}  ({s['ratio_multiple']:.2f}x)")
    print(f"  Daily corr QQQ vs SPY: {s['corr_daily']:.3f}")
    hl = t1["halflife_days"]
    print(f"  OU half-life raw log-ratio: {hl['log_ratio_raw']:.0f} days")
    print(f"  OU half-life 252d z-score:  {hl['zscore_252']:.1f} days")

    print("\n  Conditional forward QQQ−IVV return by z bucket (month-end):")
    print(f"  {'bucket':<24}{'n':>5}  {'1m':>8} {'3m':>8} {'6m':>8} {'12m':>8}  hit1m hit12m")
    for r in t1["buckets"]:
        def fmt(k):
            v = r.get(k)
            return f"{v:>+7.2%}" if v is not None else f"{'—':>8}"
        h1 = r.get("1m_hit_laggard")
        h12 = r.get("12m_hit_laggard")
        h1s = f"{h1:>5.0%}" if h1 is not None else "   —"
        h12s = f"{h12:>6.0%}" if h12 is not None else "    —"
        print(f"  {r['bucket']:<24}{r['n']:>5}  {fmt('1m_mean_qqq_minus_ivv')} {fmt('3m_mean_qqq_minus_ivv')} "
              f"{fmt('6m_mean_qqq_minus_ivv')} {fmt('12m_mean_qqq_minus_ivv')}  {h1s} {h12s}")

    print("\n  Long-laggard / short-leader hit rate when |z| ≥ thresh:")
    print(f"  {'|z|≥':>6}{'n':>6}  {'1m hit':>8}{'1m pnl':>9}  {'3m hit':>8}{'3m pnl':>9}  {'12m hit':>8}{'12m pnl':>9}")
    for r in t1["thresholds"]:
        print(f"  {r['thresh']:>6.1f}{r['n_months']:>6}  "
              f"{r['1m_hit']:>7.0%} {r['1m_mean_pnl']:>+8.2%}  "
              f"{r['3m_hit']:>7.0%} {r['3m_mean_pnl']:>+8.2%}  "
              f"{r['12m_hit']:>7.0%} {r['12m_mean_pnl']:>+8.2%}")

    print("\n  Corr(z, subsequent QQQ−IVV): negative ⇒ mean reversion")
    for r in t1["corr_z_fwd"]:
        print(f"    {r['horizon']:<4}  {r['corr_z_vs_fwd_qqq_minus_ivv']:+.3f}  n={r['n']}")

    st = t1["standalone"]
    print("\n  Always-invested 50/50 vs 15pp laggard tilt:")
    print(f"    50/50   CAGR {st['ew']['cagr']:.2%}  Sharpe {st['ew']['sharpe']:.3f}  MaxDD {st['ew']['maxdd']:.1%}  Term ${st['ew']['term']:.2f}")
    print(f"    Laggard CAGR {st['tilt']['cagr']:.2%}  Sharpe {st['tilt']['sharpe']:.3f}  MaxDD {st['tilt']['maxdd']:.1%}  Term ${st['tilt']['term']:.2f}")
    print(f"    LS gated ann {st['ls_gated']['ann']:+.2%}  Sharpe {st['ls_gated']['sharpe']:.3f}")
    print(f"    Occupancy cheap/rich/neutral: {st['occupancy']['qqq_cheap']:.0%}/{st['occupancy']['qqq_rich']:.0%}/{st['occupancy']['neutral']:.0%}")
    print(f"\n  {'period':<22}{'50/50':>8}{'lag tilt':>10}{'Δ CAGR':>9}{'LS ann':>9}")
    for p in st["periods"]:
        print(f"  {p['period']:<22}{p['ew']['cagr']:>8.2%}{p['tilt']['cagr']:>10.2%}{p['tilt_minus_ew_cagr']:>+9.2%}{p['ls_ann']:>+9.2%}")

    print("\n" + "=" * 80)
    print("TEST 2 — V19d overlay (2002-01 → latest), 15pp tilt when both pods 3/3 and |z|≥1")
    print("=" * 80)
    start = "2002-01-02"
    smas = {p: dpdf.rolling(p, min_periods=p).mean() for p in SMA_PERIODS}
    z_all = zscore_log_ratio(dpdf["QQQ"], dpdf["IVV"], Z_LOOKBACK)
    cb_all = {a: below_all_smas(dpdf, smas, a) for a in ["QQQ", "IVV", "IAU"]}
    kw = dict(smas=smas, z=z_all, cb=cb_all)
    base, _ = run_v19d(dpdf, daily_ret, rfr, actual, both_start, start, "base", **kw)
    lag, lag_log = run_v19d(dpdf, daily_ret, rfr, actual, both_start, start, "lag", **kw)
    mom, _ = run_v19d(dpdf, daily_ret, rfr, actual, both_start, start, "mom", **kw)

    def row(name, s):
        m = metrics(s)
        print(f"  {name:<22} {m['cagr']:>6.2%} {m['vol']:>6.2%} {m['sharpe']:>7.3f} {m['maxdd']:>7.1%} ${m['term']:>8.2f}")
        return m

    print(f"  {'strategy':<22} {'CAGR':>7} {'Vol':>7} {'Sharpe':>7} {'MaxDD':>7} {'Term$1':>9}")
    mb = row("V19d 45/45/10", base)
    ml = row("V19d laggard tilt", lag)
    mm = row("V19d leader tilt", mom)

    crises = [
        ("GFC", "2007-11-01", "2009-03-31"),
        ("COVID", "2020-02-01", "2020-04-30"),
        ("2022", "2022-01-01", "2022-12-31"),
    ]
    crisis_out = []
    print("\n  Crisis MaxDD:")
    for label, a, b in crises:
        db, dl, dm = crisis_dd(base, a, b), crisis_dd(lag, a, b), crisis_dd(mom, a, b)
        print(f"    {label:<8} base {db:.1%}  lag {dl:.1%}  mom {dm:.1%}")
        crisis_out.append({"crisis": label, "base": db, "lag": dl, "mom": dm})

    n_tilt = sum(1 for x in lag_log if x["tilt"] != "neutral")
    n_both = sum(1 for x in lag_log if x["both_on"])
    n_q = sum(1 for x in lag_log if x["tilt"] == "qqq")
    n_i = sum(1 for x in lag_log if x["tilt"] == "ivv")
    print(f"\n  Months both-on: {n_both}/{len(lag_log)}  tilted: {n_tilt}  (QQQ laggard {n_q}, IVV laggard {n_i})")

    # Subperiod CAGRs for V19d
    v19_periods = []
    for label, a, b in [
        ("2002–2007", "2002-01-02", "2007-10-31"),
        ("GFC", "2007-11-01", "2009-03-31"),
        ("2009–2012", "2009-04-01", "2012-12-31"),
        ("2013–2021", "2013-01-01", "2021-12-31"),
        ("2022", "2022-01-01", "2022-12-31"),
        ("2023–2026", "2023-01-01", "2026-08-14"),
    ]:
        v19_periods.append({
            "period": label,
            "base_cagr": cagr(base.loc[a:b]),
            "lag_cagr": cagr(lag.loc[a:b]),
            "mom_cagr": cagr(mom.loc[a:b]),
        })
        print(f"  {label:<12} base {cagr(base.loc[a:b]):.2%}  lag {cagr(lag.loc[a:b]):.2%}  mom {cagr(mom.loc[a:b]):.2%}")

    out = {
        "test1": t1,
        "test2": {
            "base": mb,
            "lag": ml,
            "mom": mm,
            "delta_lag_vs_base": {
                "cagr_pp": (ml["cagr"] - mb["cagr"]) * 100,
                "sharpe": ml["sharpe"] - mb["sharpe"],
                "maxdd_pp": (ml["maxdd"] - mb["maxdd"]) * 100,
                "term": ml["term"] - mb["term"],
            },
            "delta_mom_vs_base": {
                "cagr_pp": (mm["cagr"] - mb["cagr"]) * 100,
                "sharpe": mm["sharpe"] - mb["sharpe"],
                "maxdd_pp": (mm["maxdd"] - mb["maxdd"]) * 100,
            },
            "crises": crisis_out,
            "occupancy": {
                "months": len(lag_log),
                "both_on": n_both,
                "tilted": n_tilt,
                "qqq_laggard": n_q,
                "ivv_laggard": n_i,
            },
            "periods": v19_periods,
            "note": "Local CSVs; SPY spliced under IVV; GLD as IAU. Same-engine comparison, not locked V19d numbers.",
        },
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n  Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
