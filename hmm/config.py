"""HMM v2 configuration — 3 states, 5 orthogonal features, diagonal covariance."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class HMMConfig:
    # Model structure
    n_states: int = 4
    covariance_type: str = "diag"
    random_state: int = 42
    n_iter: int = 100
    tol: float = 1e-4

    # Rolling window
    window_days: int = 756
    min_window_days: int = 504
    refit_frequency: int = 21

    # 8 predictive-optimized features (max |r| = 0.439, mean |r| = 0.163)
    # Selected by greedy forward selection optimizing |corr with fwd SPY returns|
    hmm_features: list[str] = field(default_factory=lambda: [
        "nfci_z",         # Financial conditions (pred=0.127)
        "wti_z",          # Oil / inflation signal (pred=0.110)
        "ust_10y3m_z",    # Yield curve slope (pred=0.090)
        "mort_30y_z",     # Mortgage rates / consumer credit (pred=0.062)
        "fed_bs_z",       # Fed balance sheet / liquidity (pred=0.041)
        "spy_rtn_21d",    # 21-day SPY momentum (pred=0.035)
        "vvix_z",         # Vol-of-vol / tail risk (pred=0.030)
        "skew_z",         # Options skew / positioning (pred=0.026)
    ])

    # Raw indicator column mapping
    raw_indicator_map: dict = field(default_factory=lambda: {
        "nfci_z": "NFCI",
        "wti_z": "WTI",
        "ust_10y3m_z": "UST_10Y_3M",
        "mort_30y_z": "MORT_30Y",
        "fed_bs_z": "FED_BS_DELTA_3M",
        "spy_rtn_21d": "SPY_CLOSE",
        "vvix_z": "VVIX",
        "skew_z": "SKEW",
    })

    # Z-score parameters
    zscore_halflife: int = 252
    zscore_min_periods: int = 63
    zscore_clip: float = 3.0

    # PIT safety
    exclude_current_bar: bool = True

    # State comparison
    states_to_compare: list[int] = field(default_factory=lambda: [3, 4])

    # Data paths
    raw_indicators_path: Path = Path("data/macro/raw_indicators.parquet")
    output_dir: Path = Path("hmm/output")

    # Validation
    walk_forward_test_days: int = 63
    walk_forward_step_days: int = 63
    min_train_days: int = 756
