"""Configuration for Kritzman relevance-weighted allocation experiment."""

CONFIG = {
    # Relevance engine
    "relevance_top_pct": 0.20,
    "min_history_months": 120,
    "z_score_window": 120,
    "covariance_regularization": 1e-6,

    # Indicator exclusion (anti-momentum mask, same as Harvey)
    "exclude_recent_months": 36,

    # Allocation
    "max_single_asset": 0.40,
    "max_equity": 0.85,
    "min_cash": 0.03,
    "baseline_weights": {
        "IVV": 0.45, "QQQ": 0.25, "VGLT": 0.05,
        "IAU": 0.10, "DBC": 0.05, "cash": 0.10,
    },

    # Faber filter (same as existing system)
    "sma_periods": [6, 10, 12],
    "trend_threshold_full": 3,
    "trend_threshold_partial": 2,
    "partial_multiplier": 0.70,

    # Backtest
    "backtest_start": "2002-01-01",
    "backtest_end": "2026-04-01",
    "transaction_cost": 0.0010,

    # Assets
    "assets": ["IVV", "QQQ", "VGLT", "IAU", "DBC", "cash"],
    "equity_assets": ["IVV", "QQQ"],
}
