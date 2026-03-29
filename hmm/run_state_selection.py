"""Compare N=3 vs N=4 states and pick optimal number."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from hmm.config import HMMConfig
from hmm.features import build_hmm_features, get_spy_prices
from hmm.model import fit_and_predict_rolling
from hmm.labeler import label_states
from hmm.validation import walk_forward_validate

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    base = HMMConfig()
    features = build_hmm_features(base)
    spy = get_spy_prices(base)
    if spy is None:
        logger.error("No SPY prices"); return

    results = {}
    for n in base.states_to_compare:
        logger.info(f"\n{'='*60}\nTESTING N_STATES = {n}\n{'='*60}")
        cfg = HMMConfig(n_states=n)
        preds, last_model = fit_and_predict_rolling(features, cfg)
        lr = label_states(preds, features, spy, cfg, last_model)
        profs = lr.profiles
        val = walk_forward_validate(features, spy, cfg)

        pi = preds.set_index("trade_date")
        fwd = spy.pct_change(21).shift(-21)
        pi["r"] = fwd.reindex(pi.index)

        sr = {}
        for s in range(n):
            r = pi.loc[pi["most_likely_state"] == s, "r"].dropna()
            sr[s] = {"mean": r.mean() if len(r) else 0, "std": r.std() if len(r) > 1 else 1, "n": len(r)}

        means = [v["mean"] for v in sr.values()]
        pstd = np.mean([v["std"] for v in sr.values()])
        sep = (max(means) - min(means)) / pstd if pstd > 0 else 0

        results[n] = {"profiles": profs, "val": val, "sep": sep, "sr": sr}

        for s, v in sr.items():
            lab = next((p.label for p in profs if p.state_id == s), "?")
            logger.info(f"  State {s} ({lab}): mean={v['mean']:+.4f}, std={v['std']:.4f}, n={v['n']}")

    logger.info(f"\n{'='*60}\nSTATE COUNT COMPARISON\n{'='*60}")
    logger.info(f"{'N':>3} {'BIC':>14} {'AIC':>14} {'Duration':>10} {'Sep':>8} {'Used':>6}")
    for n, r in sorted(results.items()):
        v = r["val"]
        logger.info(f"{n:3d} {v.avg_bic:>14,.0f} {v.avg_aic:>14,.0f} {v.avg_state_duration:>10.1f} {r['sep']:>8.3f} {v.avg_states_used:>6.1f}")

    bb = min(results, key=lambda n: results[n]["val"].avg_bic)
    bs = max(results, key=lambda n: results[n]["sep"])
    logger.info(f"\nBest BIC: N={bb} | Best separation: N={bs}")
    if bb == bs:
        logger.info(f"RECOMMENDATION: N={bb} (both agree)")
    else:
        logger.info(f"RECOMMENDATION: N={bb} (BIC preferred)")


if __name__ == "__main__":
    main()
