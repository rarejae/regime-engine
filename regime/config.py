"""Harvey-Mulliner regime similarity configuration."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RegimeConfig:
    # Variable definitions: name → (FRED series, description)
    # Variables are equal-weighted per Harvey
    variables: dict = field(default_factory=lambda: {
        "sp500":       {"fred": "SP500",      "desc": "S&P 500 level",         "yf_fallback": "^GSPC"},
        "yield_curve": {"fred_long": "GS10",  "fred_short": "TB3MS", "desc": "10Y minus 3M yield"},
        "oil":         {"fred": "MCOILWTICO", "desc": "WTI crude oil",         "start": "1986-01"},
        "copper":      {"fred": "PCOPPUSDM",  "desc": "Copper price USD/lb"},
        "tbill":       {"fred": "TB3MS",      "desc": "3-month T-bill yield"},
        "vol":         {"yf": "^VIX",         "desc": "VIX (realized vol pre-1990)"},
        "stock_bond_corr": {"derived": True,  "desc": "3Y rolling SPY-bond correlation"},
    })

    # Transformation parameters (exact Harvey methodology)
    change_lookback: int = 12          # 12-month change
    zscore_window: int = 120           # 10-year rolling std
    winsorize_clip: float = 3.0        # ±3 winsorization

    # Similarity parameters
    exclude_trailing_months: int = 36  # Anti-momentum mask
    similar_quantile: float = 0.15     # Bottom 15% = "similar"
    dissimilar_quantile: float = 0.85  # Top 15% = "anti-regime"

    # Backtest
    backtest_start: str = "1985-01"
    backtest_end: str = "2024-12"

    # Paths
    monthly_history_path: Path = Path("data/macro/monthly_history.parquet")
    output_dir: Path = Path("regime/output")

    # Fetch
    fred_start: str = "1955-01-01"  # Fetch extra for rolling windows
