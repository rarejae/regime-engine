"""Portfolio strategy layer: Harvey tilts + baseline bands + Kritzman adjustment."""

import numpy as np
import pandas as pd

PARAMS = {
    "IVV":  {"floor": 0.05, "baseline": 0.30, "ceiling": 0.50,
             "elev_floor": 0.05, "elev_ceil": 0.35,
             "crisis_floor": 0.00, "crisis_ceil": 0.20},
    "QQQ":  {"floor": 0.00, "baseline": 0.15, "ceiling": 0.45,
             "elev_floor": 0.00, "elev_ceil": 0.25,
             "crisis_floor": 0.00, "crisis_ceil": 0.10},
    "VGLT": {"floor": 0.00, "baseline": 0.10, "ceiling": 0.50,
             "elev_floor": 0.05, "elev_ceil": 0.50,
             "crisis_floor": 0.10, "crisis_ceil": 0.55},
    "IAU":  {"floor": 0.00, "baseline": 0.10, "ceiling": 0.35,
             "elev_floor": 0.05, "elev_ceil": 0.35,
             "crisis_floor": 0.05, "crisis_ceil": 0.40},
    "DBC":  {"floor": 0.00, "baseline": 0.10, "ceiling": 0.25,
             "elev_floor": 0.00, "elev_ceil": 0.15,
             "crisis_floor": 0.00, "crisis_ceil": 0.10},
    "VNQ":  {"floor": 0.00, "baseline": 0.10, "ceiling": 0.25,
             "elev_floor": 0.00, "elev_ceil": 0.10,
             "crisis_floor": 0.00, "crisis_ceil": 0.05},
    "cash": {"floor": 0.00, "baseline": 0.15, "ceiling": 0.50,
             "elev_floor": 0.05, "elev_ceil": 0.40,
             "crisis_floor": 0.10, "crisis_ceil": 0.50},
}

ASSETS = list(PARAMS.keys())


def get_kritzman_state(turb_pctl: float, ar_z: float) -> str:
    """Determine Kritzman state from turbulence percentile and AR z-score."""
    if turb_pctl > 0.95 or ar_z > 2.0:
        return "crisis"
    if turb_pctl > 0.80 or ar_z > 1.0:
        return "elevated"
    return "normal"


def get_active_bands(state: str, recovery_blend: float = 0.0) -> dict:
    """Get floor/ceiling per asset for the current Kritzman state.

    During recovery (state transitioning from elevated/crisis to normal),
    recovery_blend interpolates: 0.0 = full constrained, 1.0 = full normal.
    """
    bands = {}
    for asset, p in PARAMS.items():
        if state == "crisis":
            fl, cl = p["crisis_floor"], p["crisis_ceil"]
        elif state == "elevated":
            fl, cl = p["elev_floor"], p["elev_ceil"]
        else:
            fl, cl = p["floor"], p["ceiling"]

        # Blend toward normal during recovery
        if recovery_blend > 0 and state != "normal":
            norm_fl, norm_cl = p["floor"], p["ceiling"]
            fl = fl + (norm_fl - fl) * recovery_blend
            cl = cl + (norm_cl - cl) * recovery_blend

        bands[asset] = {"floor": fl, "ceiling": cl, "baseline": p["baseline"]}
    return bands


def compute_tilts(
    expected_returns: dict,
    unconditional_means: dict,
    signal_stds: dict,
    min_distance: float,
    dist_25: float,
    dist_75: float,
) -> dict:
    """Compute normalized tilts from Harvey expected returns.

    Returns dict of adj_tilt per asset, in [-1, +1].
    """
    # Confidence from minimum distance
    if dist_75 > dist_25:
        conf = 1.0 - np.clip((min_distance - dist_25) / (dist_75 - dist_25), 0, 1)
    else:
        conf = 0.5

    tilts = {}
    for asset in expected_returns:
        er = expected_returns.get(asset, 0)
        mu = unconditional_means.get(asset, 0)
        sigma = signal_stds.get(asset, 0.01)

        signal = er - mu
        normalized = np.clip(signal / max(sigma, 1e-6), -1.0, 1.0)
        tilts[asset] = normalized * conf

    return tilts


def construct_weights(tilts: dict, bands: dict) -> dict:
    """Convert tilts to weights within active bands, normalize to sum=1.

    Iterative clip-and-renormalize (max 10 rounds).
    """
    raw = {}
    for asset in ASSETS:
        b = bands.get(asset, {"floor": 0, "ceiling": 1, "baseline": 0.1})
        t = tilts.get(asset, 0)
        baseline = b["baseline"]
        fl = b["floor"]
        cl = b["ceiling"]

        if t >= 0:
            w = baseline + t * (cl - baseline)
        else:
            w = baseline + t * (baseline - fl)

        raw[asset] = np.clip(w, fl, cl)

    # Normalize iteratively
    for _ in range(10):
        total = sum(raw.values())
        if abs(total - 1.0) < 1e-6:
            break
        if total == 0:
            raw = {a: 1.0 / len(ASSETS) for a in ASSETS}
            break

        raw = {a: v / total for a, v in raw.items()}

        # Re-clip after normalization
        changed = False
        for asset in ASSETS:
            b = bands.get(asset, {"floor": 0, "ceiling": 1})
            old = raw[asset]
            raw[asset] = np.clip(old, b["floor"], b["ceiling"])
            if abs(old - raw[asset]) > 1e-6:
                changed = True

        if not changed:
            break

    # Final normalization
    total = sum(raw.values())
    if total > 0:
        raw = {a: v / total for a, v in raw.items()}

    return raw


def get_baseline_weights() -> dict:
    """Return the static baseline portfolio."""
    w = {a: PARAMS[a]["baseline"] for a in ASSETS}
    total = sum(w.values())
    return {a: v / total for a, v in w.items()}
