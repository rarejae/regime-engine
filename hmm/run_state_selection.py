"""Compare N=3, 4, 5 states and pick optimal number."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from hmm.config import HMMConfig
from hmm.features import build_hmm_features, get_spy_prices
from hmm.model import fit_and_predict_rolling
from hmm.labeler import label_states
from hmm.validation import walk_forward_validate

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    base_config = HMMConfig()

    features = build_hmm_features(base_config)

    raw = pd.read_parquet(base_config.raw_indicators_path)
    raw.index = pd.to_datetime(raw.index)
    raw = raw.sort_index()
    spy_prices = get_spy_prices(raw, base_config)

    results = {}

    for n_states in base_config.states_to_compare:
        logger.info(f"\n{'='*60}")
        logger.info(f"TESTING N_STATES = {n_states}")
        logger.info(f"{'='*60}")

        config = HMMConfig(n_states=n_states)

        predictions = fit_and_predict_rolling(features, config)
        if predictions.empty:
            logger.warning(f"N={n_states}: no predictions generated")
            continue

        profiles = label_states(predictions, features, spy_prices, config)
        val_report = walk_forward_validate(features, spy_prices, config)

        # Return separation
        pred_indexed = predictions.set_index("trade_date")
        fwd_1m = spy_prices.pct_change(21).shift(-21)
        pred_indexed["spy_fwd_1m"] = fwd_1m.reindex(pred_indexed.index)

        state_returns = {}
        for state in range(n_states):
            mask = pred_indexed["most_likely_state"] == state
            rets = pred_indexed.loc[mask, "spy_fwd_1m"].dropna()
            state_returns[state] = {
                "mean": float(rets.mean()) if len(rets) else 0.0,
                "std": float(rets.std()) if len(rets) > 1 else 0.01,
                "count": len(rets),
            }

        means = [sr["mean"] for sr in state_returns.values() if sr["count"] > 0]
        stds = [sr["std"] for sr in state_returns.values() if sr["count"] > 0]
        pooled_std = np.mean(stds) if stds else 1.0
        separation = (max(means) - min(means)) / pooled_std if pooled_std > 0 and means else 0

        results[n_states] = {
            "profiles": profiles,
            "val_report": val_report,
            "separation": separation,
            "state_returns": state_returns,
        }

    # Print comparison
    print(f"\n{'='*70}")
    print(f"  STATE COUNT COMPARISON")
    print(f"{'='*70}")
    print(f"\n{'N':>4} {'BIC':>14} {'AIC':>14} {'Avg Duration':>13} {'Separation':>12} {'States Used':>12}")
    print("-" * 72)
    for n, r in sorted(results.items()):
        vr = r["val_report"]
        print(f"{n:4d} {vr.avg_bic:>14,.0f} {vr.avg_aic:>14,.0f} "
              f"{vr.avg_state_duration:>13.1f} {r['separation']:>12.3f} {vr.avg_states_used:>12.1f}")

    print(f"\nState return profiles:")
    for n, r in sorted(results.items()):
        print(f"\n  N={n}:")
        for state, sr in sorted(r["state_returns"].items()):
            label = "?"
            for p in r["profiles"]:
                if p.state_id == state:
                    label = p.label
                    break
            print(f"    State {state} ({label:>16}): mean={sr['mean']:+.4f}, "
                  f"std={sr['std']:.4f}, n={sr['count']}")

    # Recommend
    if results:
        best_bic = min(results.items(), key=lambda x: x[1]["val_report"].avg_bic)
        best_sep = max(results.items(), key=lambda x: x[1]["separation"])

        print(f"\n  Best by BIC: N={best_bic[0]} ({best_bic[1]['val_report'].avg_bic:,.0f})")
        print(f"  Best by return separation: N={best_sep[0]} ({best_sep[1]['separation']:.3f})")

        if best_bic[0] == best_sep[0]:
            print(f"\n  RECOMMENDATION: N={best_bic[0]} (both metrics agree)")
        else:
            print(f"\n  RECOMMENDATION: N={best_bic[0]} (BIC preferred for parsimony)")
            print(f"  Consider N={best_sep[0]} if return separation matters more")
    print()


if __name__ == "__main__":
    main()
