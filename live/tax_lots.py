"""FIFO cost-basis lots + realized-gain tax estimation for taxable accounts.

Estimator, not a 1099. Long-term = held > 365 days. Wash-sale-disallowed losses
are tracked but **excluded from the loss offset** — a disallowed loss does not
reduce taxable gains (its basis would normally roll into the replacement lot; the
estimator simply does not credit it).
"""

from __future__ import annotations

from datetime import date, datetime


def as_date(x) -> date:
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    return datetime.fromisoformat(str(x)[:10]).date()


def add_lot(lots: list, shares: float, cost: float, dt) -> None:
    lots.append({"shares": round(float(shares), 6), "cost": float(cost),
                 "date": as_date(dt).isoformat()})


def sell_fifo(lots: list, qty: float, price: float, sell_date) -> list[dict]:
    """Consume `qty` shares FIFO from `lots` (mutated). Return realized events."""
    remaining = round(float(qty), 6)
    sd = as_date(sell_date)
    events: list[dict] = []
    while remaining > 1e-9 and lots:
        lot = lots[0]
        take = min(lot["shares"], remaining)
        gain = (price - lot["cost"]) * take
        held = (sd - as_date(lot["date"])).days
        events.append({
            "sell_date": sd.isoformat(),
            "ticker": None,  # filled by caller
            "qty": round(take, 6),
            "proceeds": round(price * take, 2),
            "cost": round(lot["cost"] * take, 2),
            "gain": round(gain, 2),
            "long_term": held > 365,
            "held_days": held,
            "wash_disallowed": False,
        })
        lot["shares"] = round(lot["shares"] - take, 6)
        remaining = round(remaining - take, 6)
        if lot["shares"] <= 1e-9:
            lots.pop(0)
    return events


def estimate_tax(realized: list[dict], year: int,
                 ordinary: float, ltcg: float, state: float) -> dict:
    """Aggregate a year's realized events into an estimated tax bill.

    Wash-disallowed losses are counted separately and do NOT offset gains.
    Simplified: ignores the $3k ordinary-income loss allowance and carryforwards.
    """
    st_gain = lt_gain = st_loss = lt_loss = wash = 0.0
    for e in realized:
        if as_date(e["sell_date"]).year != year:
            continue
        g = float(e["gain"])
        if g >= 0:
            if e["long_term"]:
                lt_gain += g
            else:
                st_gain += g
        elif e.get("wash_disallowed"):
            wash += -g
        elif e["long_term"]:
            lt_loss += -g
        else:
            st_loss += -g
    net_st = st_gain - st_loss
    net_lt = lt_gain - lt_loss
    # Losses offset gains within a category, then across categories (IRS netting).
    tax = 0.0
    if net_st >= 0 and net_lt >= 0:
        tax = net_st * (ordinary + state) + net_lt * (ltcg + state)
    else:
        total = net_st + net_lt
        if total > 0:
            # a net gain survives; it keeps the character of the side that was positive
            rate = (ordinary + state) if net_st > 0 else (ltcg + state)
            tax = total * rate
        # else: net loss overall — no tax (≤ $3k offsets ordinary income; carryforward — not modeled)
    return {
        "year": year,
        "st_gain": round(st_gain, 2), "lt_gain": round(lt_gain, 2),
        "st_loss_allowed": round(st_loss, 2), "lt_loss_allowed": round(lt_loss, 2),
        "wash_disallowed_loss": round(wash, 2),
        "net_short": round(net_st, 2), "net_long": round(net_lt, 2),
        "est_tax": round(tax, 2),
        "rates": f"{ordinary:.0%} ST / {ltcg:.0%} LT + {state:.0%} state",
    }
