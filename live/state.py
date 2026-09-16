"""Persisted V19d live state machine + wash-sale clocks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

RUNTIME = Path(__file__).resolve().parent / "runtime"
STATE_PATH = RUNTIME / "state.json"

SLEEVES = ("pod1", "pod2", "gold")


@dataclass
class SleeveRuntime:
    status: str = "HOLD"  # HOLD | CB_PENDING | FLAT | REENTRY_ELIGIBLE
    mode: str = "cash"
    levered: bool = False
    ticker: str | None = None
    wash_blocked_until: str | None = None  # ISO date
    last_cb: str | None = None
    last_fill_px: float | None = None
    rejected_mode: str | None = None


@dataclass
class SystemState:
    updated: str = ""
    stage: int = 0  # 0 paper, 1 agentic
    dry_run: bool = True
    tax_mode: str = "TAXABLE_STANDARD"
    capital: float = 1000.0
    sleeves: dict = field(default_factory=dict)

    def ensure_sleeves(self):
        for s in SLEEVES:
            if s not in self.sleeves:
                self.sleeves[s] = asdict(SleeveRuntime())
            elif isinstance(self.sleeves[s], SleeveRuntime):
                self.sleeves[s] = asdict(self.sleeves[s])


def load_state() -> SystemState:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    if not STATE_PATH.exists():
        st = SystemState(
            updated=datetime.now(timezone.utc).isoformat(),
            sleeves={s: asdict(SleeveRuntime()) for s in SLEEVES},
        )
        save_state(st)
        return st
    raw = json.loads(STATE_PATH.read_text())
    keys = ("updated", "stage", "dry_run", "tax_mode", "capital", "sleeves")
    st = SystemState(**{k: raw[k] for k in keys if k in raw})
    st.ensure_sleeves()
    return st


def save_state(st: SystemState) -> None:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    st.updated = datetime.now(timezone.utc).isoformat()
    st.ensure_sleeves()
    payload = {
        "updated": st.updated,
        "stage": st.stage,
        "dry_run": st.dry_run,
        "tax_mode": st.tax_mode,
        "capital": st.capital,
        "sleeves": st.sleeves,
    }
    STATE_PATH.write_text(json.dumps(payload, indent=2, default=str))


def wash_clear(sleeve: dict, as_of: datetime | None = None) -> bool:
    until = sleeve.get("wash_blocked_until")
    if not until:
        return True
    as_of = as_of or datetime.now(timezone.utc)
    return as_of.date() >= datetime.fromisoformat(until).date()


def arm_wash_sale(sleeve: dict, loss: bool, days: int = 31) -> None:
    if not loss:
        sleeve["wash_blocked_until"] = None
        return
    until = datetime.now(timezone.utc).date() + timedelta(days=days)
    sleeve["wash_blocked_until"] = until.isoformat()


def transition_on_signal(st: SystemState, eval_result: dict) -> list[dict]:
    """Update sleeve statuses from today's evaluation. Returns proposed actions."""
    actions = []
    mapping = {
        "pod1": ("p1_mode", "p1_lev", "QQQ", "qld"),
        "pod2": ("p2_mode", "p2_lev", "IVV", "sso"),
        "gold": ("gold_mode", None, "IAU", "iau"),
    }
    from live.signals import ticker_for_mode

    for sleeve, (mode_key, lev_key, signal_asset, _) in mapping.items():
        sl = st.sleeves[sleeve]
        mode = eval_result[mode_key]
        lev = eval_result[lev_key] if lev_key else False
        breach = eval_result["breaches"][signal_asset]
        ticker = ticker_for_mode(mode)

        # CB while holding risk
        holding_risk = mode != "cash" or sl.get("status") == "HOLD" and sl.get("mode") not in (None, "cash")
        if breach and sl.get("mode") not in (None, "cash") and sl.get("status") in ("HOLD", "CB_PENDING", "REENTRY_ELIGIBLE"):
            if sl.get("status") != "CB_PENDING":
                actions.append({
                    "sleeve": sleeve,
                    "action": "CB_SELL",
                    "ticker": sl.get("ticker") or ticker_for_mode(sl.get("mode", "cash")),
                    "reason": f"{signal_asset} below all 3 SMAs",
                    "preauthorized": True,
                })
            sl["status"] = "CB_PENDING"
            sl["last_cb"] = eval_result["day"]
        elif sl.get("status") == "CB_PENDING":
            # sell assumed in flight / due — watcher marks FLAT after fill
            pass
        elif mode == "cash":
            sl["status"] = "FLAT" if sl.get("status") != "HOLD" or sl.get("mode") != "cash" else "FLAT"
            sl["mode"] = "cash"
            sl["levered"] = False
            sl["ticker"] = None
        else:
            # risk-on signal
            if not wash_clear(sl):
                sl["status"] = "FLAT"
                actions.append({
                    "sleeve": sleeve,
                    "action": "BLOCKED_WASH",
                    "ticker": ticker,
                    "reason": f"wash-sale until {sl.get('wash_blocked_until')}",
                    "preauthorized": False,
                })
            else:
                holding = sl.get("status") == "HOLD" and sl.get("mode") not in (None, "cash")
                if holding and (sl.get("mode") != mode or sl.get("levered") != lev):
                    actions.append({
                        "sleeve": sleeve,
                        "action": "REBALANCE",
                        "ticker": ticker,
                        "mode": mode,
                        "levered": lev,
                        "reason": "mode change",
                        "preauthorized": False,
                    })
                elif holding:
                    pass  # already in the target; approval already happened
                elif sl.get("rejected_mode") == mode:
                    sl["status"] = "FLAT"
                else:
                    actions.append({
                        "sleeve": sleeve,
                        "action": "BUY",
                        "ticker": ticker,
                        "mode": mode,
                        "levered": lev,
                        "reason": "risk-on — needs your approval",
                        "preauthorized": False,
                    })
                    sl["status"] = "REENTRY_ELIGIBLE"
                sl["mode"] = mode
                sl["levered"] = bool(lev)
                sl["ticker"] = ticker

    return actions
