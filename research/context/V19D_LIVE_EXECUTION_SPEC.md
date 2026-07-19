# V19d Live Execution & Readiness Spec

**Status:** Active — build in progress  
**Account assumption:** Taxable (Robinhood Agentic sleeve)  
**Related:** [[V19D_PRODUCTION_SPEC]] | [[2026-07-18_marketstack_verification]]

---

## Purpose

Turn paper V19d into a live loop we can trust: identical signals, measured fills,
tax-aware re-entry, and staged capital. This document is the contract the code
in `live/` implements.

---

## Trust stages (do not skip)

| Stage | Capital | Duration | Pass criteria |
|-------|---------|----------|---------------|
| 0 Paper twin | $0 | ongoing | Watcher logs same scores/CB as research code |
| 1 Read-only MCP | $0 | ≥10 trading days | Positions/quotes readable; no orders |
| 2 Shadow tickets | $0 | ≥20 trading days | Alerts match; you fill manually; slip logged |
| 3 Tiny live | **$1–2.5K** | until ≥1 CB drill or 1 real CB + 2 monthly cycles | Slip ≤ budget; no double orders; kill switch works |
| 4 Scale | up to target | after audit | Live NAV within slip budget of paper twin YTD |

**Kill switch:** Disconnect Robinhood Agentic MCP + flatten Agentic account in the app. Must be tested in Stage 3 before scaling.

---

## Slippage budgets (pre-committed)

| Event | Benchmark | Soft limit | Hard limit (escalate / review) |
|-------|-----------|------------|--------------------------------|
| Routine rebalance | mid / prior close | 10 bps | 25 bps |
| CB exit | next regular-hours open | 40 bps | 100 bps |
| Annual drag vs paper | paper twin NAV | 20 bps/yr | 50 bps/yr |

If hard limit breached: pause new entries, keep CB sells armed, run post-mortem before scaling.

---

## Fill policy

### Circuit breaker (risk-off) — pre-authorized
1. Detect 3/3 SMA breach on QQQ / IVV / IAU (sleeve held).
2. **Prefer sell into the CB close** (MOC / limit 15:30–15:50 ET) when breach is clear before the bell — stress test shows next-open P90 delay cost >> 100 bps hard budget.
3. Else sell next session: limit at open print − 15 bps (sell), good for 15 minutes.
4. Unfilled remainder → **market**. Being flat > saving bps.
5. Target: **cash**, not unlevered ETF (V19d rule).

### Re-entry / monthly rebalance — human approve
1. Agent proposes tickets only (`review` / shadow).
2. You approve after checking wash-sale clock (taxable).
3. Limit near mid; no chase; may span 1–2 sessions.

### Pre-authorization matrix

| Action | Auto (Agentic) | Manual approve |
|--------|----------------|----------------|
| CB sell → cash | YES | — |
| Monthly buy / re-lever | NO | YES |
| Drift rebalance | NO | YES |
| Any options | NEVER | — |

---

## Taxable account rules

`tax_mode = TAXABLE_STANDARD` (default for this deployment).

1. **CB exits are never delayed for tax.** Risk first.
2. **Wash-sale clock:** after a **loss-taking** CB sell of ticker T, no buy of T (or substantially identical) for **31 calendar days**. Log divergence from paper V19d as `tax_deferral_days`.
3. **Specific lot ID** on sells when broker supports it.
4. **Drift band:** keep 5% in paper; optionally widen to 7% live to cut tax churn (flag in ledger).
5. **QLD/SSO stay in the Agentic taxable sleeve** — do not split the brain across accounts.

### Tax-drag model (dashboard)
Used for confidence / compounding sensitivity, not tax filing:

- Input: ordinary rate, LTCG rate, state rate, assumed STCG fraction of realized gains.
- Approximate annual realized gain from turnover proxy + CB months.
- Haircut equity curve / DCA terminal → **after-tax** series.
- Toggle on Streamlit: pre-tax vs after-tax.

---

## Signal integrity

- Source of truth: same SMA periods `[126, 200, 252]` and scoring as `experiments/v19d_final`.
- Live prices: primary Robinhood quote when MCP connected; fallback yfinance.
- **Parity job:** `live/parity.py` compares live scores to recomputation on packaged/cached history; fail if score differs.
- Dual-source check (optional): if RH vs Yahoo last &gt; 1%, amber alert — do not auto-trade on disagreement.

---

## State machine

```
HOLD → CB_PENDING → FLAT → REENTRY_ELIGIBLE → HOLD
         ↑                        │
         └──── (new breach) ──────┘
```

- `CB_PENDING`: breach detected; sell in flight or due next open.
- `FLAT`: sleeve in cash; wash-sale clock may block `REENTRY_ELIGIBLE`.
- `REENTRY_ELIGIBLE`: monthly score allows risk-on AND wash clock clear.
- Persisted in `live/runtime/state.json`. Idempotent: repeated runs must not duplicate sells.

---

## Robinhood Agentic constraints

- MCP: `https://agent.robinhood.com/mcp/trading` (OAuth; not in-repo credentials).
- Agent **trades only** the Agentic account — fund that sleeve with Stage 3/4 capital only.
- Equities only (enough for QLD/SSO/QQQ/IVV/IAU).
- Default mode in code: `DRY_RUN=true` until `LIVE_TRADING=1` **and** stage ≥ 3.

---

## Observability

| Artifact | Path |
|----------|------|
| Daily signal log | `live/runtime/signals.jsonl` |
| Order / fill ledger | `live/runtime/ledger.jsonl` |
| State | `live/runtime/state.json` |
| Paper twin NAV | `live/runtime/paper_nav.csv` |
| Slip stress report | `live/reports/slippage_stress.md` |

Alerts: stdout + optional Telegram (env `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID`).

---

## Stage 3 tiny-live checklist

- [ ] Agentic account funded with test capital only
- [ ] MCP connected in Cursor; read-only tools verified
- [ ] `DRY_RUN=false` only after kill-switch drill
- [ ] CB sells pre-authorized; buys require approval
- [ ] Watcher cron / manual run before 15:40 ET
- [ ] Tax rates set in viz; after-tax hurdle still clears vs 60/40
- [ ] First week: reconcile ledger vs Robinhood activity feed daily

---

## Explicit non-goals

- No discretionary “wait for a better open”
- No options overlay in v1
- No unofficial Robinhood scraping APIs
- No full-account automation outside Agentic sleeve
