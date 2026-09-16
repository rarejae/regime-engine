"""Approval routing: CB is auto, buys wait, reject suppresses same-mode nag."""

from live.approvals import AUTO, INFORM, NEEDS_YOU, ticket_id
from live.sizing import shares_for, sleeve_notional
from live.state import SystemState, SleeveRuntime, transition_on_signal
from dataclasses import asdict


def _st(**sleeves):
    st = SystemState(capital=1000.0, dry_run=True)
    st.sleeves = {}
    for name, kwargs in sleeves.items():
            extra = {k: kwargs.pop(k) for k in list(kwargs) if k not in SleeveRuntime.__dataclass_fields__}
            row = asdict(SleeveRuntime(**kwargs))
            row.update(extra)
            st.sleeves[name] = row
    for name in ("pod1", "pod2", "gold"):
        st.sleeves.setdefault(name, asdict(SleeveRuntime()))
    return st


def _eval(p1="qld", p2="sso", gold="cash", bq=False, bi=False, bg=False):
    return {
        "day": "2026-08-24",
        "p1_mode": p1,
        "p1_lev": p1 == "qld",
        "p2_mode": p2,
        "p2_lev": p2 == "sso",
        "gold_mode": gold,
        "breaches": {"QQQ": bq, "IVV": bi, "IAU": bg},
    }


def test_flat_risk_on_is_buy_not_auto():
    st = _st(pod1={"status": "FLAT", "mode": "cash"})
    acts = transition_on_signal(st, _eval())
    kinds = {a["sleeve"]: a for a in acts}
    assert kinds["pod1"]["action"] == "BUY"
    assert kinds["pod1"]["action"] in NEEDS_YOU
    assert kinds["pod1"]["ticker"] == "QLD"
    assert kinds["pod1"]["preauthorized"] is False
    assert st.sleeves["pod1"]["status"] == "REENTRY_ELIGIBLE"


def test_hold_cb_is_auto():
    st = _st(pod1={"status": "HOLD", "mode": "qld", "ticker": "QLD"})
    acts = transition_on_signal(st, _eval(bq=True))
    cb = [a for a in acts if a["action"] == "CB_SELL"]
    assert len(cb) == 1
    assert cb[0]["preauthorized"] is True
    assert cb[0]["action"] in AUTO
    assert st.sleeves["pod1"]["status"] == "CB_PENDING"


def test_hold_matching_mode_is_silent():
    st = _st(
        pod1={"status": "HOLD", "mode": "qld", "levered": True, "ticker": "QLD"},
        pod2={"status": "HOLD", "mode": "sso", "levered": True, "ticker": "SSO"},
        gold={"status": "FLAT", "mode": "cash"},
    )
    acts = transition_on_signal(st, _eval())
    assert [a for a in acts if a["sleeve"] == "pod1"] == []
    assert [a for a in acts if a["sleeve"] == "pod2"] == []


def test_reject_same_mode_does_not_reask():
    st = _st(pod1={"status": "FLAT", "mode": "cash", "rejected_mode": "qld"})
    acts = transition_on_signal(st, _eval())
    assert [a for a in acts if a["sleeve"] == "pod1"] == []
    assert st.sleeves["pod1"]["status"] == "FLAT"


def test_sizing_partial():
    assert sleeve_notional(1000, "pod1", "qld") == 450.0
    assert sleeve_notional(1000, "pod1", "qqq_partial") == 315.0
    assert sleeve_notional(1000, "gold", "cash") == 0.0
    assert shares_for(450, 100) == 4.5


def test_ticket_id_stable():
    assert ticket_id("pod1", "BUY") == "pod1-buy"
    assert "BLOCKED_WASH" in INFORM
