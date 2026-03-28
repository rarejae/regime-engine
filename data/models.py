"""Pydantic models for the Macro Regime Engine data layer."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator


class IndicatorValue(BaseModel):
    """A single resolved value for one indicator at a point in time."""

    id: str = Field(description="Indicator ID from indicators.yaml, e.g. 'VIX'")
    value: float = Field(description="Resolved numeric value")
    unit: str = Field(default="", description="Display unit, e.g. '%', 'bps', 'idx'")
    axis: str = Field(description="Composite axis: GROWTH | INFLATION | LIQUIDITY | RISK")
    polarity: str = Field(description="risk_on | risk_off | ambiguous")
    weight: float = Field(description="Relative weight within its axis")
    source: str = Field(description="Data source module that produced this value")
    is_proxy: bool = Field(default=False, description="True when a fallback/proxy was used")
    proxy_note: str = Field(default="", description="Explanation when is_proxy=True")
    as_of: datetime = Field(description="Timestamp of the underlying observation")
    fetched_at: datetime = Field(default_factory=datetime.utcnow)


class IndicatorSnapshot(BaseModel):
    """Full collection of resolved indicator values for a single evaluation run."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    indicators: dict[str, IndicatorValue] = Field(default_factory=dict)
    missing: list[str] = Field(default_factory=list, description="Indicator IDs that could not be fetched")
    warnings: list[str] = Field(default_factory=list, description="Non-fatal issues during fetch")

    def get(self, indicator_id: str) -> IndicatorValue | None:
        """Return an indicator value by ID, or None if missing."""
        return self.indicators.get(indicator_id)

    def values_for_axis(self, axis: str) -> list[IndicatorValue]:
        """Return all indicator values belonging to the given axis."""
        return [v for v in self.indicators.values() if v.axis == axis]


class NormalizedIndicator(BaseModel):
    """An indicator value after z-score normalization."""

    id: str
    raw_value: float
    z_score: float = Field(description="Z-score clipped to ±z_clip")
    z_score_unclipped: float
    mean: float
    std: float
    polarity_sign: int = Field(description="+1 for risk_on, -1 for risk_off, 0 for ambiguous")
    weight: float
    axis: str
    is_proxy: bool = False


class AxisScores(BaseModel):
    """Composite axis scores after weighting and tanh bounding."""

    GROWTH: float = Field(description="Growth axis score [-100, +100]")
    INFLATION: float = Field(description="Inflation axis score [-100, +100]")
    LIQUIDITY: float = Field(description="Liquidity axis score [-100, +100]")
    RISK: float = Field(description="Risk/Sentiment axis score [-100, +100]")
    coverage: dict[str, float] = Field(
        default_factory=dict,
        description="Fraction of total axis weight covered by available indicators",
    )


class RegimeClassification(BaseModel):
    """Output of the regime classifier."""

    primary: str = Field(description="GOLDILOCKS | REFLATION | STAGFLATION | DEFLATION | MIXED")
    modifiers: list[str] = Field(default_factory=list)
    conviction: float = Field(description="0–100 conviction score")
    transition: str = Field(description="accelerating | decelerating | stable")
    axis_scores: AxisScores
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class TradeCard(BaseModel):
    """Complete trade recommendation output."""

    regime: str
    modifiers: list[str]
    conviction: float
    transition: str
    strategy: str
    strategy_type: str = Field(description="credit | debit")
    underlying: str = Field(default="SPY")
    short_strike: float
    short_delta: float
    long_strike: float
    long_delta: float
    width: float
    expiry: str = Field(description="YYYY-MM-DD")
    dte: int
    estimated_credit: float = Field(description="Per share (multiply by 100 for $ per contract)")
    max_risk: float
    credit_to_width: float
    breakeven: float
    win_probability: float
    axis_scores: dict[str, float]
    management: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list, description="Proxy usage and caveats")
    timestamp: str

    @model_validator(mode="after")
    def compute_derived(self) -> "TradeCard":
        if self.width > 0:
            object.__setattr__(self, "credit_to_width", round(self.estimated_credit / self.width, 4))
        return self
