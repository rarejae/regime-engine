"""Unconstrained Harvey engine backtest — 7-ETF universe with IWM."""

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
from regime.run_daily_backtest import fetch_daily_etf_returns as _fetch_base_daily

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

OUTPUT = Path("experiments/signal_development/output")
OUTPUT.mkdir(parents=True, exist_ok=True)
PARQUET = Path("data/macro/roth_v2_asset_returns.parquet")
TC = 0.0010
UNIVERSE = ["IVV", "QQQ", "IWM", "VGLT", "IAU", "DBC", "cash"]


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


def _yf_daily(ticker, start="1979-01-01"):
    try:
        import yfinance as yf
        d = yf.download(ticker, start=start, progress=False)
        if d is None or d.empty:
            return None
        p = d["Close"]
        if hasattr(p, "columns"): p = p.iloc[:, 0]
        p.index = pd.to_datetime(p.index).tz_localize(None)
        return p
    except Exception:
        return None


def _splice(pre, etf_ret):
    if etf_ret is not None and len(etf_ret) > 0:
        cutoff = etf_ret.index.min()
        return pd.concat([pre[pre.index < cutoff], etf_ret]).sort_index().pipe(
            lambda s: s[~s.index.duplicated(keep="last")])
    return pre


def fetch_monthly_returns(force=False):
    if PARQUET.exists() and not force:
        return pd.read_parquet(PARQUET)

    PARQUET.parent.mkdir(parents=True, exist_ok=True)
    frames = {}

    # IVV — S&P 500
    sp = _fred("SPASTT01USM661N", "1957-01-01")
    if sp is not None:
        pre = sp.resample("MS").last().pct_change().dropna()
        import yfinance as yf
        d = yf.download("SPY", start="1993-01-01", interval="1mo", progress=False)
        etf = None
        if d is not None and not d.empty:
            p = d["Close"]
            if hasattr(p, "columns"): p = p.iloc[:, 0]
            p.index = pd.to_datetime(p.index).tz_localize(None).to_period("M").to_timestamp()
            etf = p.pct_change().dropna()
        frames["IVV"] = _splice(pre, etf)
    logger.info(f"  IVV: {len(frames.get('IVV', []))} months")

    # QQQ — Nasdaq 100
    ndx = _yf_daily("^NDX", "1985-01-01")
    if ndx is not None:
        pre = ndx.resample("MS").last().pct_change().dropna()
        pre.index = pre.index.to_period("M").to_timestamp()
        import yfinance as yf
        d = yf.download("QQQ", start="1999-03-01", interval="1mo", progress=False)
        etf = None
        if d is not None and not d.empty:
            p = d["Close"]
            if hasattr(p, "columns"): p = p.iloc[:, 0]
            p.index = pd.to_datetime(p.index).tz_localize(None).to_period("M").to_timestamp()
            etf = p.pct_change().dropna()
        frames["QQQ"] = _splice(pre, etf)
    logger.info(f"  QQQ: {len(frames.get('QQQ', []))} months")

    # IWM — Russell 2000
    rut = _yf_daily("^RUT", "1979-01-01")
    if rut is not None:
        pre = rut.resample("MS").last().pct_change().dropna()
        pre.index = pre.index.to_period("M").to_timestamp()
        import yfinance as yf
        d = yf.download("IWM", start="2000-05-01", interval="1mo", progress=False)
        etf = None
        if d is not None and not d.empty:
            p = d["Close"]
            if hasattr(p, "columns"): p = p.iloc[:, 0]
            p.index = pd.to_datetime(p.index).tz_localize(None).to_period("M").to_timestamp()
            etf = p.pct_change().dropna()
        frames["IWM"] = _splice(pre, etf)
    logger.info(f"  IWM: {len(frames.get('IWM', []))} months")

    # VGLT — Long-term treasuries (17yr duration)
    gs10 = _fred("GS10", "1960-01-01")
    if gs10 is not None:
        gs10_m = gs10.resample("MS").last().dropna()
        bond_ret = (-17.0 * gs10_m.diff() / 100 + gs10_m.shift(1) / 1200).dropna()
        import yfinance as yf
        d = yf.download("TLT", start="2002-07-01", interval="1mo", progress=False)
        etf = None
        if d is not None and not d.empty:
            p = d["Close"]
            if hasattr(p, "columns"): p = p.iloc[:, 0]
            p.index = pd.to_datetime(p.index).tz_localize(None).to_period("M").to_timestamp()
            etf = p.pct_change().dropna()
        frames["VGLT"] = _splice(bond_ret, etf)
    logger.info(f"  VGLT: {len(frames.get('VGLT', []))} months")

    # IAU — Gold (PPI Precious Metals from 1960)
    gold = _fred("WPU1017", "1960-01-01")
    if gold is not None:
        pre = gold.resample("MS").last().pct_change().dropna()
        import yfinance as yf
        d = yf.download("GLD", start="2004-11-01", interval="1mo", progress=False)
        etf = None
        if d is not None and not d.empty:
            p = d["Close"]
            if hasattr(p, "columns"): p = p.iloc[:, 0]
            p.index = pd.to_datetime(p.index).tz_localize(None).to_period("M").to_timestamp()
            etf = p.pct_change().dropna()
        frames["IAU"] = _splice(pre, etf)
    logger.info(f"  IAU: {len(frames.get('IAU', []))} months")

    # DBC — Commodities (PPI proxy)
    ppi = _fred("PPIACO", "1960-01-01")
    if ppi is not None:
        pre = ppi.resample("MS").last().pct_change().dropna()
        import yfinance as yf
        d = yf.download("DBC", start="2006-02-01", interval="1mo", progress=False)
        etf = None
        if d is not None and not d.empty:
            p = d["Close"]
            if hasattr(p, "columns"): p = p.iloc[:, 0]
            p.index = pd.to_datetime(p.index).tz_localize(None).to_period("M").to_timestamp()
            etf = p.pct_change().dropna()
        frames["DBC"] = _splice(pre, etf)
    logger.info(f"  DBC: {len(frames.get('DBC', []))} months")

    # Cash
    tb = _fred("TB3MS", "1960-01-01")
    if tb is not None:
        frames["cash"] = (tb.resample("MS").last() / 1200).dropna()
    logger.info(f"  cash: {len(frames.get('cash', []))} months")

    df = pd.DataFrame(frames).sort_index()
    df.to_parquet(PARQUET)
    logger.info(f"Saved: {df.shape}")
    return df


def fetch_daily_returns():
    """Fetch daily returns for all 7 ETFs including IWM."""
    import yfinance as yf
    frames = {}
    for our, ticker in [("IVV","SPY"),("QQQ","QQQ"),("IWM","IWM"),("VGLT","TLT"),("IAU","GLD"),("DBC","DBC")]:
        d = yf.download(ticker, start="1998-01-01", progress=False)
        if d is not None and not d.empty:
            p = d["Close"]
            if hasattr(p, "columns"): p = p.iloc[:, 0]
            p.index = pd.to_datetime(p.index).tz_localize(None)
            frames[our] = p.pct_change()

    key = os.environ.get("FRED_API_KEY")
    if key:
        from fredapi import Fred
        tb = Fred(api_key=key).get_series("DTB3", observation_start="1998-01-01")
        if tb is not None:
            tb.index = pd.to_datetime(tb.index)
            frames["cash"] = tb / 36000

    return pd.DataFrame(frames).sort_index()


def main(force=False):
    config = RegimeConfig()

    print("=" * 80)
    print("  UNCONSTRAINED HARVEY ENGINE — 7-ETF WITH IWM")
    print("=" * 80)

    raw_macro = fetch_monthly_history(config)
    transformed = transform_variables(raw_macro, config)
    z_data = get_valid_zscored(transformed, config)
    z_data_lagged = z_data.shift(1).dropna()

    asset_ret = fetch_monthly_returns(force=force)
    asset_ret_fwd = asset_ret.shift(-1)

    daily_ret = fetch_daily_returns()

    print(f"\nMonthly returns: {asset_ret.shape}")
    for c in sorted(asset_ret.columns):
        n = asset_ret[c].notna().sum()
        fv = asset_ret[c].first_valid_index()
        print(f"  {c:>6}: {n:>4} months (from {fv.date() if fv else 'N/A'})")

    print(f"Daily returns: {daily_ret.shape}")

    # Backtest range
    common_start = max(daily_ret.dropna(how="all").index.min(), pd.Timestamp("2002-01-01"))
    trading_days = daily_ret.loc[common_start:].index
    print(f"Backtest: {len(trading_days)} days ({common_start.date()} to {trading_days.max().date()})")

    # Run
    strats = {"regime_iwm": {}, "bench_6040": {}, "bench_ivv": {}}
    current_w = {s: None for s in strats}
    total_tc = {s: 0.0 for s in strats}
    weight_rows = []
    harvey_w = {a: 0.0 for a in UNIVERSE}

    for day in trading_days:
        if day not in daily_ret.index:
            continue
        dr = daily_ret.loc[day]
        avail = [a for a in UNIVERSE if a in dr.index and pd.notna(dr[a])]
        if len(avail) < 3:
            continue
        actual = {a: float(dr[a]) for a in avail}

        is_ms = (day == trading_days[0] or day.month != trading_days[trading_days.get_loc(day)-1].month)

        if is_ms:
            z_cands = z_data_lagged.index[z_data_lagged.index < day]
            if len(z_cands) > 0:
                z_dt = z_cands[-1]
                try:
                    sim = compute_distances(z_data_lagged, z_dt, config)
                    sim_er = {}
                    for a in avail:
                        if a not in asset_ret_fwd.columns:
                            sim_er[a] = 0.0; continue
                        rets = [asset_ret_fwd.loc[d, a] for d in sim.similar_dates
                                if d in asset_ret_fwd.index and pd.notna(asset_ret_fwd.loc[d, a])]
                        sim_er[a] = np.mean(rets) if rets else 0.0
                    harvey_w = allocate_similarity_weighted(sim_er, max_single=0.50)

                    row = {"date": day}
                    for a in UNIVERSE: row[a] = harvey_w.get(a, 0)
                    row["sim_er"] = dict(sim_er)
                    weight_rows.append(row)
                except ValueError:
                    pass

        w_6040 = {a: 0.0 for a in avail}
        if "IVV" in avail: w_6040["IVV"] = 0.60
        if "VGLT" in avail: w_6040["VGLT"] = 0.40
        w_ivv = {a: (1.0 if a == "IVV" else 0.0) for a in avail}

        all_w = {"regime_iwm": (harvey_w, is_ms), "bench_6040": (w_6040, is_ms), "bench_ivv": (w_ivv, False)}

        for cn, (new_w, trade) in all_w.items():
            if current_w[cn] is not None and not trade:
                w_used = current_w[cn]
            else:
                w_used = new_w
                to = sum(abs(new_w.get(a,0) - (current_w[cn] or {}).get(a,0)) for a in avail) / 2
                total_tc[cn] += to * TC
                current_w[cn] = new_w

            strats[cn][day] = sum(w_used.get(a,0) * actual.get(a,0) for a in avail)

    results = {s: pd.Series(d).sort_index() for s, d in strats.items() if d}
    wdf = pd.DataFrame([{k:v for k,v in r.items() if k != "sim_er"} for r in weight_rows]).set_index("date")

    n_years = len(trading_days) / 252
    ivv = results.get("bench_ivv")
    labels = {"regime_iwm": "Regime+IWM", "bench_6040": "60/40", "bench_ivv": "IVV B&H"}

    # Report
    print(f"\n{'='*80}")
    print(f"  PERFORMANCE")
    print(f"{'='*80}")
    print(f"\n  {'Config':>14} {'AnnRet':>7} {'AnnVol':>7} {'Sharpe':>7} {'MaxDD':>8} {'CorrIVV':>8}")
    print(f"  {'-'*52}")

    for cn in ["regime_iwm","bench_6040","bench_ivv"]:
        s = results[cn]
        ar = s.mean()*252; av = s.std()*np.sqrt(252)
        sh = ar/av if av > 0 else 0
        cum = (1+s).cumprod()
        dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
        corr = s.corr(ivv)
        print(f"  {labels[cn]:>14} {ar:>6.1%} {av:>6.1%} {sh:>7.2f} {dd:>7.1%} {corr:>7.2f}")

    # Previous 7-ETF for comparison
    print(f"\n  Previous 7-ETF (no IWM): Sharpe ~0.78, MaxDD -24.3%")

    # Crisis
    print(f"\n{'='*80}")
    print(f"  CRISIS DRAWDOWNS")
    print(f"{'='*80}")
    for cn2, cs, ce in [("GFC","2008-09-01","2009-03-31"),("COVID","2020-02-19","2020-03-23"),("2022","2022-01-03","2022-10-31")]:
        print(f"\n  {cn2}:")
        for cn in ["regime_iwm","bench_6040","bench_ivv"]:
            s = results[cn]
            c = s[(s.index >= pd.Timestamp(cs)) & (s.index <= pd.Timestamp(ce))]
            if len(c) > 0:
                print(f"    {labels[cn]:>14}: {((1+c).prod()-1):>+7.1%}")

    # Calendar years
    print(f"\n{'='*80}")
    print(f"  CALENDAR YEAR RETURNS")
    print(f"{'='*80}")
    print(f"  {'Year':>6} {'Regime':>8} {'60/40':>8} {'IVV':>8}")
    for yr in range(2002, 2025):
        row = f"  {yr:>6}"
        for cn in ["regime_iwm","bench_6040","bench_ivv"]:
            s = results[cn]; y = s[s.index.year == yr]
            row += f" {(1+y).prod()-1:>+7.1%}" if len(y) > 20 else f" {'--':>8}"
        print(row)

    # Final values
    print(f"\n{'='*80}")
    print(f"  FINAL VALUES ($1)")
    print(f"{'='*80}")
    for cn in ["regime_iwm","bench_6040","bench_ivv"]:
        s = results[cn]
        print(f"  {labels[cn]:>14}: ${(1+s).cumprod().iloc[-1]:>8.2f}")

    # Average allocation
    print(f"\n{'='*80}")
    print(f"  AVERAGE ALLOCATION")
    print(f"{'='*80}")
    for a in sorted(wdf.columns):
        if a in UNIVERSE:
            print(f"  {a:>6}: {wdf[a].mean():>5.1%} (med={wdf[a].median():.0%}, max={wdf[a].max():.0%}, zero={((wdf[a]<0.005).mean()):.0%})")

    # GFC weights
    print(f"\n{'='*80}")
    print(f"  GFC ALLOCATION (2007-07 to 2009-06)")
    print(f"{'='*80}")
    gfc = wdf[(wdf.index >= "2007-07") & (wdf.index <= "2009-06")]
    if len(gfc) > 0:
        hdr = f"  {'Date':>7}"
        for a in UNIVERSE: hdr += f" {a[:4]:>5}"
        print(hdr)
        for dt, row in gfc.iterrows():
            line = f"  {dt.strftime('%Y-%m'):>7}"
            for a in UNIVERSE:
                v = row.get(a, 0)
                line += f" {v:>4.0%}" if v > 0.005 else f" {'--':>4}"
            print(line)

    # 2013
    print(f"\n{'='*80}")
    print(f"  2013 ALLOCATION")
    print(f"{'='*80}")
    w13 = wdf[wdf.index.year == 2013]
    if len(w13) > 0:
        eq_avg = w13.get("IVV",pd.Series(0)).mean() + w13.get("QQQ",pd.Series(0)).mean() + w13.get("IWM",pd.Series(0)).mean()
        print(f"  Avg equity (IVV+QQQ+IWM): {eq_avg:.0%}")
        for a in UNIVERSE:
            if a in w13.columns:
                print(f"    {a:>6}: {w13[a].mean():.0%}")

    # IWM analysis
    print(f"\n{'='*80}")
    print(f"  IWM ALLOCATION ANALYSIS")
    print(f"{'='*80}")
    iwm_col = wdf.get("IWM", pd.Series(0, index=wdf.index))

    for period, label in [(("2003","2003"), "2003 (early recovery)"),
                           (("2009","2010"), "2009-10 (post-GFC recovery)"),
                           (("2018","2019"), "2018-19 (late cycle)"),
                           (("2023","2023"), "2023 (tech momentum)")]:
        s, e = period
        mask = (wdf.index.year >= int(s)) & (wdf.index.year <= int(e))
        sub = iwm_col[mask]
        qqq_sub = wdf.get("QQQ", pd.Series(0))[mask]
        print(f"  {label}: IWM avg={sub.mean():.0%}, QQQ avg={qqq_sub.mean():.0%}")

    print()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    main(force=p.parse_args().force)
