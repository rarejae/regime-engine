"""HMM module configuration — all hyperparameters in one place."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class HMMConfig:
    # Model structure
    n_states: int = 4
    covariance_type: str = "full"
    random_state: int = 42
    n_iter: int = 100
    tol: float = 1e-4

    # Rolling window
    window_days: int = 756         # ~3 years of trading days
    min_window_days: int = 504     # ~2 years minimum for early periods
    refit_frequency: int = 21      # Refit every ~1 month

    # Feature selection — columns from raw_indicators.parquet (after engineering)
    hmm_features: list[str] = field(default_factory=lambda: [
        "spy_return_21d",
        "spy_vol_21d",
        "vix_z",
        "yield_curve_z",
        "credit_spread_z",
        "inflation_z",
    ])

    # Z-score parameters
    zscore_halflife: int = 252
    zscore_clip: float = 3.0

    # PIT safety
    exclude_current_bar: bool = True

    # State comparison
    states_to_compare: list[int] = field(default_factory=lambda: [3, 4, 5])

    # Data paths
    raw_indicators_path: Path = Path("data/macro/raw_indicators.parquet")
    options_data_dir: Path = Path("data/processed")
    output_dir: Path = Path("hmm/output")

    # Validation
    walk_forward_test_days: int = 63
    walk_forward_step_days: int = 63
    min_train_days: int = 756
