"""Side-by-side comparison with charts: Kritzman relevance vs Harvey-Mulliner."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT = Path("experiments/kritzman_relevance/output")


def generate_comparison_charts():
    """Generate comparison visualizations from saved backtest results."""
    OUTPUT.mkdir(parents=True, exist_ok=True)

    ret_df = pd.read_parquet(OUTPUT / "monthly_returns.parquet")
    strategies = ret_df.columns.tolist()

    # 1. Cumulative equity curves
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {"Harvey": "#2c3e50", "Kritzman-InvVol": "#e74c3c",
              "Kritzman-MV": "#3498db", "Kritzman-RP": "#2ecc71"}
    for s in strategies:
        cum = (1 + ret_df[s]).cumprod()
        ax.plot(cum.index, cum.values, label=s, color=colors.get(s, "gray"), linewidth=1.2)
    ax.set_title("Cumulative Return: Kritzman Relevance vs Harvey-Mulliner (1x, Monthly)")
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(OUTPUT / "equity_curves.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {OUTPUT / 'equity_curves.png'}")

    # 2. Drawdown comparison
    fig, axes = plt.subplots(len(strategies), 1, figsize=(12, 2.5 * len(strategies)), sharex=True)
    for i, s in enumerate(strategies):
        cum = (1 + ret_df[s]).cumprod()
        dd = (cum - cum.expanding().max()) / cum.expanding().max()
        axes[i].fill_between(dd.index, 0, dd.values, alpha=0.4, color=colors.get(s, "gray"))
        axes[i].set_ylabel(s, fontsize=9)
        axes[i].set_ylim(-0.45, 0.02)
        axes[i].grid(True, alpha=0.3)
    fig.suptitle("Drawdowns by Strategy", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT / "drawdowns.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {OUTPUT / 'drawdowns.png'}")

    # 3. Monthly return scatter: Harvey vs each Kritzman variant
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for i, s in enumerate(["Kritzman-InvVol", "Kritzman-MV", "Kritzman-RP"]):
        if s not in ret_df.columns or "Harvey" not in ret_df.columns:
            continue
        ax = axes[i]
        h = ret_df["Harvey"].values * 100
        k = ret_df[s].values * 100
        ax.scatter(h, k, alpha=0.3, s=10, edgecolors="none")
        lim = max(abs(h).max(), abs(k).max()) * 1.1
        ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.3, linewidth=0.8)
        ax.set_xlabel("Harvey (%)")
        ax.set_ylabel(f"{s} (%)")
        corr = np.corrcoef(h[~np.isnan(h) & ~np.isnan(k)], k[~np.isnan(h) & ~np.isnan(k)])[0, 1]
        ax.set_title(f"{s}\nr={corr:.3f}")
        ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle("Monthly Return Scatter: Harvey vs Kritzman Variants", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT / "return_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUTPUT / 'return_scatter.png'}")

    # 4. Rolling 12-month Sharpe comparison
    fig, ax = plt.subplots(figsize=(12, 5))
    for s in strategies:
        rolling_ret = ret_df[s].rolling(12).mean() * 12
        rolling_vol = ret_df[s].rolling(12).std() * np.sqrt(12)
        rolling_sharpe = rolling_ret / rolling_vol.replace(0, np.nan)
        ax.plot(rolling_sharpe.index, rolling_sharpe.values, label=s,
                color=colors.get(s, "gray"), linewidth=0.9, alpha=0.8)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_title("Rolling 12-Month Sharpe Ratio")
    ax.set_ylabel("Sharpe")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT / "rolling_sharpe.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {OUTPUT / 'rolling_sharpe.png'}")

    print("\n  All comparison charts generated.")


if __name__ == "__main__":
    generate_comparison_charts()
