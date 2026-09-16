# V19d HSA Tracker

A **local, read-only** operational dashboard for the live V19d sleeves. An account
switcher toggles between the **Fidelity HSA** and the **Robinhood taxable** account.
For each it shows current value, allocation, and signal, and — using the *same dated
cash flows* — how V19d is doing versus SPY / QQQ / 60-40 buy-and-hold. It also
previews emergency withdrawals.

For the **taxable** account it additionally shows a **tax estimator** (realized ST/LT
gains this year, with wash-sale-disallowed losses *not* credited) and **wash-sale
banners** — a 31-day re-entry block after a loss, and a cross-account warning when a
taxable loss coincides with the HSA holding the same ticker.

It never trades and never writes portfolio state. It is a pure presentation layer:
it reads **one file the Python side produces** (`live/runtime/fidelity_tracker.json`)
and renders it. Data sources:

- **Your account** (holdings, cash) → the Fidelity **CSV** you import (`positions`).
- **Market / benchmark prices** → **yfinance**, inside `refresh` — the same source the
  signals use. No third-party feed.

The Python CLI stays the single source of truth.

## Run

```bash
# from the repo root — pick a password
TRACKER_PASSWORD=your-password node tracker/server.js
# → http://127.0.0.1:4319   (sign in with that password)
```

- **Local only:** binds `127.0.0.1`; not reachable from other machines.
- **Auth:** a password login (HttpOnly session cookie). Set `TRACKER_PASSWORD`, or the
  app generates one on first run and saves it to `tracker/data/password.txt`.
- **Node:** requires Node ≥ 18. No `npm install` — zero dependencies.

## Daily workflow

```bash
# 1. update holdings from a Fidelity positions export (as needed)
.venv/bin/python -m live.fidelity positions ~/Downloads/Portfolio_Positions.csv

# 2. rebuild the dashboard data (pulls benchmark prices from yfinance)
.venv/bin/python -m live.fidelity refresh

# 3. open / reload the tracker
```

The benchmark lines compare *the same contributions* had they gone into SPY / QQQ /
60-40 instead — an honest "am I beating just buying the index" read. The growth chart
adds one point per `refresh` day.

## Seed it

```bash
.venv/bin/python -m live.fidelity add-cash 8000     # your current HSA balance
.venv/bin/python -m live.fidelity plan --force      # deploy to the V19d target
.venv/bin/python -m live.fidelity confirm           # after placing at Fidelity
.venv/bin/python -m live.fidelity refresh           # build the dashboard data
```

## Emergency withdrawal

The **Emergency withdrawal** box previews what to sell to raise a given amount (cash
first, then holdings pro-rata). To actually do it:

```bash
.venv/bin/python -m live.fidelity withdraw 2000   # prints the sell plan
# place the sells + withdraw the cash at Fidelity, then:
.venv/bin/python -m live.fidelity confirm
.venv/bin/python -m live.fidelity refresh
```

Non-medical HSA withdrawals before 65 owe income tax + a 20% penalty; qualified
medical withdrawals are tax-free.

## Files it reads

| File | For |
|------|-----|
| `live/runtime/fidelity_tracker.json` | the whole payload (written by `refresh`) |
| `live/runtime/fidelity_state.json` | holdings + cash (via `positions` / `confirm`) |
| `live/runtime/fidelity_contributions.jsonl` | dated cash flows (drives benchmarks) |
| `live/runtime/fidelity_snapshots.jsonl` | daily value snapshots (the chart) |

`tracker/data/` (password, local state) is git-ignored.
