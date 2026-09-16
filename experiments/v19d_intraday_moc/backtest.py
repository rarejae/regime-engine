"""Can a 15:50-ET market-on-close (MOC) order reproduce the close-based V19d CB?

The V19d circuit breaker sells the leveraged sleeve when the underlying CLOSES below
all 3 SMAs (126/200/252). The close isn't known until 16:00, so you cannot fill at
that close intraday -- the execution-lag study (experiments/v19d_execution_lag) shows
the honest fallback is T+1, which costs CAGR and deepens drawdown.

The intraday-MOC idea: at the 15:50-ET MOC entry cutoff, check price vs the (fixed,
prior-close) SMAs. If below all 3, submit a market-on-close order -> you fill at that
day's official closing auction, i.e. you HOLD through the whole breach day and exit at
its close (== the execution-lag 't_close' mode). That is only valid if the 15:50 read
reliably predicts the 16:00 close-based signal. This measures that reliability.

Data: Tiingo IEX intraday (2017+, the only intraday depth on the free tier), IEX venue
only. We work in ADJUSTED space (adjClose SMAs; the 15:50 raw print scaled by that
day's adjClose/close factor) so the IAU 2021-05-24 2:1 split and all dividends cancel.

RESULT (2017-01 .. 2026-09, 2,437 days/ticker, 1,139 close-breach days across QQQ/IVV/IAU):
  agreement 99.92%   caught 99.65% of real breaches
  2 false-positives + 4 misses, EVERY one a <0.18% razor-edge crossing
  15:50->close move: mean +0.001%, std 0.058%, 95%ile |move| 0.086%
Conclusion: intraday-MOC is live-achievable. It upgrades execution-lag 't_close' from
"reference" to the true achievable base case (~14.98% CAGR / -32.7% MaxDD at 100% sub).
It does NOT recover the look-ahead 'prior' (17.27% / -25.1%): that dodges the whole
breach-day decline, which no execution can avoid.

Run:  .venv/bin/python experiments/v19d_intraday_moc/backtest.py
Free-tier note: ~50 req/hour. 15min bars in yearly chunks = 30 calls; per-year parquet
cache (experiments/v19d_intraday_moc/_cache/) makes reruns free and resumes after a 429.
"""
import os, time
from pathlib import Path
from zoneinfo import ZoneInfo
import requests, pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent
CACHE = Path(__file__).resolve().parent / "_cache"
CACHE.mkdir(parents=True, exist_ok=True)
load_dotenv(str(ROOT / ".env"))
TOK = os.environ["TIINGO_API_KEY"]
BASE = "https://api.tiingo.com"
ET = ZoneInfo("America/New_York")
UNDERLYERS = ["QQQ", "IVV", "IAU"]
SMAS = [126, 200, 252]
CUT = 15 * 60 + 50  # 15:50 ET, the MOC entry cutoff


def _get_json(url, params, tries=6):
    """GET with a 429-aware wait (free-tier hourly cap)."""
    for i in range(tries):
        r = requests.get(url, params=params, timeout=120)
        if r.status_code == 429:
            wait = 300 * (i + 1)
            print(f"    429 rate-limited; sleeping {wait}s (attempt {i+1}/{tries})...")
            time.sleep(wait)
            continue
        return r.json()
    raise RuntimeError("rate-limited past retries")


def eod(ticker, start="2015-06-01"):
    """date -> (adjClose, factor=adjClose/close). Adjusted space is split/dividend-safe."""
    d = pd.DataFrame(_get_json(f"{BASE}/tiingo/daily/{ticker}/prices",
                     {"token": TOK, "startDate": start, "endDate": "2026-09-14",
                      "columns": "close,adjClose"}))
    d["date"] = pd.to_datetime(d["date"]).dt.tz_localize(None).dt.normalize()
    d = d.set_index("date")
    d["factor"] = d["adjClose"] / d["close"]
    return d[["adjClose", "factor"]]


def iex_1550(ticker):
    """date -> raw IEX price at the last 15min bar <= 15:50 ET (the ~15:45 bar), 2017-2026."""
    frames = []
    for yr in range(2017, 2027):
        cf = CACHE / f"{ticker}_{yr}_1545.parquet"
        if cf.exists():
            frames.append(pd.read_parquet(cf)["px"])
            continue
        rows = _get_json(f"{BASE}/iex/{ticker}/prices",
                         {"token": TOK, "startDate": f"{yr}-01-01", "endDate": f"{yr}-12-31",
                          "resampleFreq": "15min", "columns": "close"})
        if not isinstance(rows, list) or not rows:
            continue
        df = pd.DataFrame(rows)
        ts = pd.to_datetime(df["date"], utc=True).dt.tz_convert(ET)
        df["day"] = ts.dt.normalize().dt.tz_localize(None)
        df["mins"] = ts.dt.hour * 60 + ts.dt.minute
        df = df[df["mins"] <= CUT]
        if df.empty:
            continue
        near = df.sort_values("mins").groupby("day").tail(1).set_index("day")["close"].rename("px")
        near.to_frame().to_parquet(cf)
        frames.append(near)
        time.sleep(0.3)
    return pd.concat(frames).sort_index()


def below_all(price, row):
    return all(price < row[w] for w in SMAS)


def main():
    print("Loading EOD (adjusted) + SMAs...")
    ed = {t: eod(t) for t in UNDERLYERS}
    smas = {t: pd.DataFrame({w: ed[t]["adjClose"].rolling(w).mean() for w in SMAS}) for t in UNDERLYERS}
    print("Loading IEX 15:50 prices (cached after first run)...")
    px = {t: iex_1550(t) for t in UNDERLYERS}

    rows = []
    for t in UNDERLYERS:
        sm = smas[t].dropna()
        common = sm.index.intersection(px[t].index).intersection(ed[t].index)
        for d in common:
            ac, fac = ed[t]["adjClose"].get(d), ed[t]["factor"].get(d)
            if pd.isna(ac) or pd.isna(fac):
                continue
            adj1550 = px[t].loc[d] * fac
            rows.append(dict(ticker=t, date=d, adjclose=ac, adj1550=adj1550,
                             sig_close=below_all(ac, sm.loc[d]), sig_1550=below_all(adj1550, sm.loc[d])))
    df = pd.DataFrame(rows)

    print("\n" + "=" * 92)
    print("  15:50 MOC-cutoff signal vs official close-based CB signal (2017-2026)")
    print("=" * 92)
    for t in UNDERLYERS + ["ALL"]:
        d = df if t == "ALL" else df[df.ticker == t]
        tp = (d.sig_1550 & d.sig_close).sum()
        fp = (d.sig_1550 & ~d.sig_close).sum()
        fn = (~d.sig_1550 & d.sig_close).sum()
        tn = (~d.sig_1550 & ~d.sig_close).sum()
        bd = d.sig_close.sum()
        print(f"\n  {t:4}  n={len(d)}  close-breach days={bd}")
        print(f"        agreement {(tp+tn)/len(d):6.2%}   caught {tp/bd if bd else float('nan'):6.2%}"
              f"   false-positive {fp} ({fp/len(d):.3%} of days)   missed {fn}")

    fp = df[df.sig_1550 & ~df.sig_close]
    fn = df[~df.sig_1550 & df.sig_close]
    print(f"\n  FALSE POSITIVES (15:50 fired, close recovered above SMA):")
    for _, r in fp.iterrows():
        print(f"    {r.date.date()} {r.ticker}  15:50->close {(r.adjclose/r.adj1550-1):+.3%}")
    print(f"  MISSES (15:50 clean, close dipped below at the bell -> caught at T+1):")
    for _, r in fn.iterrows():
        print(f"    {r.date.date()} {r.ticker}  15:50->close {(r.adjclose/r.adj1550-1):+.3%}")
    m = (df.adjclose / df.adj1550 - 1)
    print(f"\n  15:50->close move: mean {m.mean():+.4%}  std {m.std():.4%}  "
          f"95%ile |move| {m.abs().quantile(.95):.3%}  max |move| {m.abs().max():.3%}")
    df.to_parquet(CACHE / "moc_study.parquet")
    print("\n  Done.  (detail saved to _cache/moc_study.parquet)")


if __name__ == "__main__":
    main()
