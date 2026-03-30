"""Roth IRA ETF universe — regime similarity allocation backtest."""

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv; load_dotenv()

import numpy as np
import pandas as pd

from regime.config import RegimeConfig
from regime.data import fetch_monthly_history
from regime.transform import transform_variables, get_valid_zscored
from regime.similarity import compute_distances
from regime.multi_asset import allocate_similarity_weighted

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ROTH_PARQUET = Path("data/macro/roth_asset_returns.parquet")
OUTPUT = Path("regime/output")
TC = 0.0010  # 10bps round-trip

ETF_NAMES = ["IVV", "QQQ", "VXUS", "VGLT", "IAU", "DBC", "VNQ", "cash"]


# ── Data fetching ─────────────────────────────────────────────────────────────

def _fred(sid, start="1955-01-01"):
    try:
        from fredapi import Fred
        key = os.environ.get("FRED_API_KEY")
        if key:
            return Fred(api_key=key).get_series(sid, observation_start=start).dropna()
    except Exception:
        pass
    try:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd={start}"
        df = pd.read_csv(url, index_col=0, parse_dates=True)
        return pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
    except Exception:
        return None


def _yf_monthly(ticker, start="1985-01-01"):
    try:
        import yfinance as yf
        d = yf.download(ticker, start=start, interval="1mo", progress=False)
        if d is None or d.empty:
            return None
        p = d["Close"]
        if hasattr(p, "columns"):
            p = p.iloc[:, 0]
        p.index = pd.to_datetime(p.index).tz_localize(None).to_period("M").to_timestamp()
        return p.pct_change().dropna()
    except Exception:
        return None


def _yf_daily(ticker, start="1985-01-01"):
    try:
        import yfinance as yf
        d = yf.download(ticker, start=start, progress=False)
        if d is None or d.empty:
            return None
        p = d["Close"]
        if hasattr(p, "columns"):
            p = p.iloc[:, 0]
        p.index = pd.to_datetime(p.index).tz_localize(None)
        return p
    except Exception:
        return None


def _splice(pre, etf):
    """Splice pre-ETF proxy with ETF returns."""
    if etf is not None and len(etf) > 0:
        cutoff = etf.index.min()
        return pd.concat([pre[pre.index < cutoff], etf]).sort_index().pipe(
            lambda s: s[~s.index.duplicated(keep="last")]
        )
    return pre


def fetch_roth_returns(force=False):
    if ROTH_PARQUET.exists() and not force:
        logger.info("Using cached Roth asset returns")
        return pd.read_parquet(ROTH_PARQUET)

    ROTH_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    frames = {}

    # 1. IVV — S&P 500
    sp = _fred("SPASTT01USM661N", "1957-01-01")
    if sp is not None:
        pre = sp.resample("MS").last().pct_change().dropna()
        etf = _yf_monthly("IVV", "2000-05-01")
        frames["IVV"] = _splice(pre, etf)
    logger.info(f"  IVV: {frames.get('IVV', pd.Series()).notna().sum()} months")

    # 2. QQQ — Nasdaq 100
    ndx = _yf_daily("^NDX", "1985-01-01")
    if ndx is not None:
        pre = ndx.resample("MS").last().pct_change().dropna()
        pre.index = pre.index.to_period("M").to_timestamp()
        etf = _yf_monthly("QQQ", "1999-03-01")
        frames["QQQ"] = _splice(pre, etf)
    logger.info(f"  QQQ: {frames.get('QQQ', pd.Series()).notna().sum()} months")

    # 3. VXUS — Intl equity
    # Layer 1: Nikkei+FTSE avg from 1984 as rough EAFE proxy
    nikkei = _yf_daily("^N225", "1984-01-01")
    ftse = _yf_daily("^FTSE", "1984-01-01")
    intl_parts = []
    if nikkei is not None and ftse is not None:
        n_m = nikkei.resample("MS").last().pct_change().dropna()
        f_m = ftse.resample("MS").last().pct_change().dropna()
        combined_idx = n_m.index.intersection(f_m.index)
        intl_early = (n_m.reindex(combined_idx) + f_m.reindex(combined_idx)) / 2
        intl_parts.append(intl_early)
    elif ftse is not None:
        intl_parts.append(ftse.resample("MS").last().pct_change().dropna())

    # Layer 2: FRED Eurozone shares from 1986
    ez = _fred("SPASTT01EZM661N", "1986-01-01")
    if ez is not None and len(ez) > 100:
        ez_m = ez.resample("MS").last().pct_change().dropna()
        intl_parts.append(ez_m)

    # Layer 3: EFA from daily
    efa_d = _yf_daily("EFA", "2001-08-01")
    if efa_d is not None:
        efa_m = efa_d.resample("MS").last().pct_change().dropna()
        efa_m.index = efa_m.index.to_period("M").to_timestamp()
        intl_parts.append(efa_m)

    if intl_parts:
        # Splice: earliest first, each successive layer overwrites
        vxus = intl_parts[0].copy()
        for layer in intl_parts[1:]:
            cutoff = layer.index.min()
            vxus = pd.concat([vxus[vxus.index < cutoff], layer]).sort_index()
            vxus = vxus[~vxus.index.duplicated(keep="last")]
        frames["VXUS"] = vxus
    logger.info(f"  VXUS: {frames.get('VXUS', pd.Series()).notna().sum()} months")

    # 4. VGLT — Long-term treasuries (17yr duration)
    gs10 = _fred("GS10", "1960-01-01")
    if gs10 is not None:
        gs10_m = gs10.resample("MS").last().dropna()
        dy = gs10_m.diff()
        coupon = gs10_m.shift(1) / 1200
        pre = (-17.0 * dy / 100 + coupon).dropna()
        etf = _yf_monthly("VGLT", "2009-11-01")
        if etf is None or len(etf) < 10:
            etf = _yf_monthly("TLT", "2002-07-01")
        frames["VGLT"] = _splice(pre, etf)
    logger.info(f"  VGLT: {frames.get('VGLT', pd.Series()).notna().sum()} months")

    # 5. IAU — Gold
    # Primary: FRED WPU1017 (PPI Precious Metals) from 1960
    gold_ppi = _fred("WPU1017", "1960-01-01")
    gold_pre = None
    if gold_ppi is not None and len(gold_ppi) > 200:
        gold_pre = gold_ppi.resample("MS").last().pct_change().dropna()
    else:
        # Fallback: yfinance GC=F daily from 2000
        gc = _yf_daily("GC=F", "2000-01-01")
        if gc is not None:
            gold_pre = gc.resample("MS").last().pct_change().dropna()

    if gold_pre is not None:
        # Splice with GLD/IAU ETF from 2004+
        etf = _yf_monthly("GLD", "2004-11-01")
        if etf is None or len(etf) < 10:
            etf = _yf_monthly("IAU", "2005-01-01")
        frames["IAU"] = _splice(gold_pre, etf)
    logger.info(f"  IAU: {frames.get('IAU', pd.Series()).notna().sum()} months")

    # 6. DBC — Broad commodities (PPI proxy)
    ppi = _fred("PPIACO", "1960-01-01")
    if ppi is not None:
        pre = ppi.resample("MS").last().pct_change().dropna()
        etf = _yf_monthly("DBC", "2006-02-01")
        frames["DBC"] = _splice(pre, etf)
    logger.info(f"  DBC: {frames.get('DBC', pd.Series()).notna().sum()} months")

    # 7. VNQ — REITs
    reit = _fred("WILLREITIND", "1971-01-01")
    if reit is not None:
        pre = reit.resample("MS").last().pct_change().dropna()
        etf = _yf_monthly("VNQ", "2004-09-01")
        frames["VNQ"] = _splice(pre, etf)
    else:
        etf = _yf_monthly("VNQ", "2004-09-01")
        if etf is not None:
            frames["VNQ"] = etf
    logger.info(f"  VNQ: {frames.get('VNQ', pd.Series()).notna().sum()} months")

    # 8. Cash
    tb = _fred("TB3MS", "1960-01-01")
    if tb is not None:
        frames["cash"] = (tb.resample("MS").last() / 1200).dropna()
    logger.info(f"  cash: {frames.get('cash', pd.Series()).notna().sum()} months")

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.to_parquet(ROTH_PARQUET)
    logger.info(f"Saved: {df.shape}")
    return df


# ── Backtest ──────────────────────────────────────────────────────────────────

def main(force=False):
    config = RegimeConfig()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  ROTH IRA ETF REGIME ALLOCATION BACKTEST")
    print("=" * 80)

    raw_macro = fetch_monthly_history(config)
    transformed = transform_variables(raw_macro, config)
    z_data = get_valid_zscored(transformed, config)

    asset_ret = fetch_roth_returns(force=force)
    print(f"\nAsset returns: {asset_ret.shape[0]} months × {asset_ret.shape[1]} ETFs")
    for c in asset_ret.columns:
        n = asset_ret[c].notna().sum()
        fv = asset_ret[c].first_valid_index()
        print(f"  {c:>6}: {n:>4} months (from {fv.date() if fv else 'N/A'})")

    start = pd.Timestamp(config.backtest_start)
    end = pd.Timestamp(config.backtest_end)
    bt_dates = z_data.index[(z_data.index >= start) & (z_data.index <= end)]
    bt_dates = bt_dates[bt_dates.isin(asset_ret.index)]
    print(f"\nBacktest: {len(bt_dates)} months ({bt_dates.min().date()} to {bt_dates.max().date()})")

    etfs = [c for c in asset_ret.columns if c != "cash"]

    # Run backtest
    strats = {"regime": {}, "bench_6040": {}, "bench_ew": {}, "bench_ivv": {}}
    weight_rows = []
    prev_w = {s: None for s in strats}

    for dt in bt_dates:
        avail = [a for a in asset_ret.columns if pd.notna(asset_ret.loc[dt, a])]
        avail_etfs = [a for a in avail if a != "cash"]
        if len(avail_etfs) < 2:
            continue

        actual = {a: asset_ret.loc[dt, a] for a in avail}

        try:
            sim = compute_distances(z_data, dt, config)
        except ValueError:
            continue

        # Expected returns from similar months
        sim_er = {}
        for a in avail:
            rets = [asset_ret.loc[d, a] for d in sim.similar_dates
                    if d in asset_ret.index and pd.notna(asset_ret.loc[d, a])]
            sim_er[a] = np.mean(rets) if rets else 0.0

        w = allocate_similarity_weighted(sim_er, max_single=0.40)

        # Benchmarks
        w_6040 = {a: 0.0 for a in avail}
        if "IVV" in avail:
            w_6040["IVV"] = 0.60
        if "VGLT" in avail:
            w_6040["VGLT"] = 0.40
        w_ew = {a: 1.0 / len(avail_etfs) if a in avail_etfs else 0.0 for a in avail}
        w_ivv = {a: (1.0 if a == "IVV" else 0.0) for a in avail}

        all_w = {"regime": w, "bench_6040": w_6040, "bench_ew": w_ew, "bench_ivv": w_ivv}

        row = {"date": dt}
        for a in avail:
            row[a] = w.get(a, 0.0)
        weight_rows.append(row)

        for sn, sw in all_w.items():
            ret = sum(sw.get(a, 0) * actual.get(a, 0) for a in avail)
            # Transaction costs
            if prev_w[sn] is not None:
                to = sum(abs(sw.get(a, 0) - prev_w[sn].get(a, 0)) for a in avail) / 2
                ret -= to * TC
            strats[sn][dt] = ret
            prev_w[sn] = sw

    results = {s: pd.Series(d).sort_index() for s, d in strats.items() if d}
    weights_df = pd.DataFrame(weight_rows).set_index("date")
    weights_df.to_parquet(OUTPUT / "roth_weights.parquet")

    # ── Report ────────────────────────────────────────────────────────────────

    ivv_rets = results.get("bench_ivv")

    print(f"\n{'='*80}")
    print(f"  PERFORMANCE SUMMARY (1985-2024)")
    print(f"{'='*80}")
    print(f"\n  {'Strategy':>14} {'AnnRet':>7} {'AnnVol':>7} {'Sharpe':>7} {'MaxDD':>8} {'HitR':>5} {'CorrIVV':>8}")
    print(f"  {'-'*58}")

    for sn in ["regime", "bench_6040", "bench_ew", "bench_ivv"]:
        s = results.get(sn)
        if s is None or len(s) < 12:
            continue
        ar = s.mean() * 12
        av = s.std() * np.sqrt(12)
        sh = ar / av if av > 0 else 0
        cum = (1 + s).cumprod()
        dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
        hit = (s > 0).mean()
        corr = s.corr(ivv_rets) if ivv_rets is not None else 0
        label = {"regime": "Regime Alloc", "bench_6040": "60/40", "bench_ew": "Equal Wt", "bench_ivv": "IVV B&H"}[sn]
        print(f"  {label:>14} {ar:>6.1%} {av:>6.1%} {sh:>7.2f} {dd:>7.1%} {hit:>5.0%} {corr:>7.2f}")

    # Decades
    print(f"\n{'='*80}")
    print(f"  BY DECADE")
    print(f"{'='*80}")
    for dname, ds, de in [("1985-94","1985","1994"),("1995-04","1995","2004"),("2005-14","2005","2014"),("2015-24","2015","2024")]:
        print(f"\n  {dname}:")
        print(f"  {'':>14} {'AnnRet':>7} {'Sharpe':>7}")
        for sn in ["regime", "bench_6040", "bench_ivv"]:
            s = results.get(sn)
            if s is None: continue
            m = (s.index >= pd.Timestamp(ds)) & (s.index < pd.Timestamp(f"{int(de)+1}"))
            d = s[m]
            if len(d) < 12: continue
            label = {"regime":"Regime","bench_6040":"60/40","bench_ivv":"IVV"}[sn]
            print(f"  {label:>14} {d.mean()*12:>6.1%} {(d.mean()/d.std()*np.sqrt(12)) if d.std()>0 else 0:>7.2f}")

    # Crises
    print(f"\n{'='*80}")
    print(f"  CRISIS DRAWDOWNS")
    print(f"{'='*80}")
    for cn, cs, ce in [("GFC","2008-09","2009-03"),("COVID","2020-02","2020-03"),("2022","2022-01","2022-10")]:
        print(f"\n  {cn} ({cs} to {ce}):")
        for sn in ["regime","bench_6040","bench_ivv"]:
            s = results.get(sn)
            if s is None: continue
            m = (s.index >= pd.Timestamp(cs)) & (s.index <= pd.Timestamp(ce))
            c = s[m]
            if len(c)==0: continue
            label = {"regime":"Regime","bench_6040":"60/40","bench_ivv":"IVV"}[sn]
            print(f"    {label:>10}: {((1+c).prod()-1):>+7.1%}")

    # Calendar years
    print(f"\n{'='*80}")
    print(f"  CALENDAR YEAR RETURNS")
    print(f"{'='*80}")
    print(f"  {'Year':>6} {'Regime':>8} {'60/40':>8} {'EqWt':>8} {'IVV':>8}")
    print(f"  {'-'*42}")
    for yr in range(1985, 2025):
        row = f"  {yr:>6}"
        for sn in ["regime","bench_6040","bench_ew","bench_ivv"]:
            s = results.get(sn)
            if s is None: row += f" {'--':>8}"; continue
            y = s[s.index.year == yr]
            if len(y)==0: row += f" {'--':>8}"
            else: row += f" {(1+y).prod()-1:>+7.1%}"
        print(row)

    # Final values
    print(f"\n{'='*80}")
    print(f"  FINAL VALUES ($1 invested)")
    print(f"{'='*80}")
    for sn in ["regime","bench_6040","bench_ew","bench_ivv"]:
        s = results.get(sn)
        if s is None or len(s)<12: continue
        label = {"regime":"Regime","bench_6040":"60/40","bench_ew":"EqWt","bench_ivv":"IVV"}[sn]
        print(f"  {label:>10}: ${(1+s).cumprod().iloc[-1]:>8.2f}")

    # ── Allocation diagnostics ────────────────────────────────────────────────

    print(f"\n{'='*80}")
    print(f"  ALLOCATION DIAGNOSTICS")
    print(f"{'='*80}")

    print(f"\n  Average weights:")
    print(f"  {'ETF':>6} {'Mean':>7} {'Median':>7} {'%Zero':>7}")
    print(f"  {'-'*30}")
    for a in sorted(weights_df.columns):
        c = weights_df[a]
        print(f"  {a:>6} {c.mean():>6.1%} {c.median():>6.1%} {(c<0.005).mean():>6.0%}")

    # Crisis weights
    for cn, cs, ce in [("GFC","2007-07","2009-06"),("COVID","2020-01","2020-06"),("2022","2022-01","2022-12")]:
        m = (weights_df.index >= pd.Timestamp(cs)) & (weights_df.index <= pd.Timestamp(ce))
        p = weights_df[m]
        if len(p)==0: continue
        print(f"\n  {cn} weights ({cs} to {ce}):")
        hdr = f"    {'Date':>7}"
        for a in sorted(weights_df.columns):
            hdr += f" {a[:5]:>5}"
        print(hdr)
        for dt, row in p.iterrows():
            line = f"    {dt.strftime('%Y-%m'):>7}"
            for a in sorted(weights_df.columns):
                v = row.get(a, 0)
                line += f" {v:>4.0%}" if v > 0.005 else f" {'--':>4}"
            print(line)

    # Turnover
    to_list = []
    prev = None
    for _, row in weights_df.iterrows():
        if prev is not None:
            to_list.append(sum(abs(row.get(a,0) - prev.get(a,0)) for a in weights_df.columns) / 2)
        prev = row
    if to_list:
        print(f"\n  Turnover: avg={np.mean(to_list):.1%}, median={np.median(to_list):.1%}, max={max(to_list):.1%}")

    print()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    main(force=p.parse_args().force)
