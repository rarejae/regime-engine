"""Ensemble voting system: Harvey, Faber, HMM, Kritzman → aggregate score → weights."""

import numpy as np

BASELINE = {"IVV": 0.45, "QQQ": 0.25, "VGLT": 0.05, "IAU": 0.10, "DBC": 0.05, "cash": 0.10}
EQUITY = ["IVV", "QQQ"]
DEFENSIVE = ["VGLT", "IAU"]
RISK = ["IVV", "QQQ", "DBC"]
ASSETS = list(BASELINE.keys())

SCORE_MULTIPLIER = {4: 1.30, 3: 1.20, 2: 1.10, 1: 1.00, 0: 1.00,
                    -1: 0.90, -2: 0.70, -3: 0.50, -4: 0.30}


def harvey_votes(expected_returns: dict, unconditional_means: dict,
                 signal_stds: dict) -> dict:
    """Harvey engine: +1/0/-1 per asset based on excess expected return."""
    votes = {}
    for a in ASSETS:
        if a == "cash":
            votes[a] = 0; continue
        er = expected_returns.get(a, 0)
        mu = unconditional_means.get(a, 0)
        sigma = signal_stds.get(a, 0.01)
        excess = er - mu
        if excess > 0.5 * sigma:
            votes[a] = 1
        elif excess < -0.5 * sigma:
            votes[a] = -1
        else:
            votes[a] = 0
    votes["cash"] = 0
    return votes


def faber_votes(trend_signals: dict) -> dict:
    """Faber: +1 if above SMA, -1 if below. Binary."""
    votes = {}
    for a in ASSETS:
        if a == "cash":
            votes[a] = 0; continue
        votes[a] = 1 if trend_signals.get(a, True) else -1
    votes["cash"] = 0
    return votes


def hmm_votes(bull_prob: float) -> dict:
    """HMM trend: equity votes from bull probability, inverse for defensives."""
    if bull_prob > 0.7:
        eq_vote = 1
    elif bull_prob < 0.3:
        eq_vote = -1
    else:
        eq_vote = 0

    votes = {}
    for a in EQUITY:
        votes[a] = eq_vote
    for a in DEFENSIVE:
        votes[a] = -eq_vote  # inverse for defensives
    votes["DBC"] = eq_vote  # DBC follows equity direction
    votes["cash"] = 0
    return votes


def kritzman_votes(turb_pctl: float, ar_z: float) -> dict:
    """Kritzman: only votes underweight risk during stress."""
    stressed = turb_pctl > 0.80 or ar_z > 1.5
    votes = {}
    if stressed:
        for a in RISK:
            votes[a] = -1
        for a in DEFENSIVE:
            votes[a] = 1
        votes["cash"] = 1
    else:
        for a in ASSETS:
            votes[a] = 0
    return votes


def aggregate_scores(voter_votes: list[dict]) -> dict:
    """Sum votes across all voters per asset."""
    scores = {a: 0 for a in ASSETS}
    for votes in voter_votes:
        for a in ASSETS:
            scores[a] += votes.get(a, 0)
    return scores


def scores_to_weights(scores: dict, voters_active: int = 4) -> dict:
    """Convert aggregate scores to portfolio weights via score multipliers."""
    w = {}
    for a in ASSETS:
        s = max(-voters_active, min(voters_active, scores.get(a, 0)))
        mult = SCORE_MULTIPLIER.get(s, 1.0)
        w[a] = BASELINE[a] * mult

    # Normalize
    total = sum(max(v, 0) for v in w.values())
    if total > 0:
        w = {a: max(v, 0) / total for a, v in w.items()}

    # Enforce limits
    for a in w:
        w[a] = min(w[a], 0.60)
    if w.get("cash", 0) < 0.02:
        deficit = 0.02 - w["cash"]
        others = [a for a in w if a != "cash" and w[a] > 0]
        tot = sum(w[a] for a in others)
        if tot > deficit:
            for a in others: w[a] -= deficit * (w[a] / tot)
        w["cash"] = 0.02

    total = sum(w.values())
    if total > 0:
        w = {a: v / total for a, v in w.items()}
    return w
