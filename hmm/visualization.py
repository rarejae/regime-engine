"""Analysis plots for HMM regime model. Requires matplotlib."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def plot_regime_timeline(predictions: pd.DataFrame, output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping plots")
        return

    pred = predictions.set_index("trade_date")
    prob_cols = [c for c in pred.columns if c.endswith("_prob")]

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.stackplot(pred.index, *[pred[c] for c in prob_cols], labels=prob_cols, alpha=0.7)
    ax.set_ylabel("State Probability")
    ax.set_title("HMM Regime State Probabilities Over Time")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "regime_timeline.png", dpi=150)
    plt.close(fig)
    logger.info("Saved regime_timeline.png")


def plot_state_returns(predictions: pd.DataFrame, spy_prices: pd.Series, output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    pred = predictions.set_index("trade_date")
    fwd = spy_prices.pct_change(21).shift(-21)
    pred["r"] = fwd.reindex(pred.index)

    states = sorted(pred["most_likely_state"].unique())
    fig, axes = plt.subplots(1, len(states), figsize=(4 * len(states), 4), sharey=True)
    if len(states) == 1:
        axes = [axes]

    for ax, s in zip(axes, states):
        r = pred.loc[pred["most_likely_state"] == s, "r"].dropna()
        ax.hist(r, bins=30, alpha=0.7, color=f"C{s}")
        ax.axvline(r.mean(), color="red", linestyle="--")
        ax.set_title(f"State {s} (n={len(r)}, μ={r.mean():.3f})")
        ax.set_xlabel("1m fwd return")

    fig.suptitle("Forward Return Distributions by HMM State")
    fig.tight_layout()
    fig.savefig(output_dir / "state_returns.png", dpi=150)
    plt.close(fig)
    logger.info("Saved state_returns.png")
