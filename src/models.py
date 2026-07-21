"""Typed result models used by the analysis and presentation layers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Factor:
    key: str
    values: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    symbol: str
    company_name: str
    mode: str
    decision: str
    score: int
    confidence: str
    current_price: float
    currency: str
    price_change_pct: float | None
    entry_low: float | None
    entry_high: float | None
    stop_loss: float | None
    target_1: float | None
    target_2: float | None
    support: float | None
    resistance: float | None
    risk_reward: float | None
    positives: list[Factor]
    risks: list[Factor]
    blockers: list[Factor]
    breakdown: dict[str, float]
    indicators: dict[str, Any]
    fundamentals: dict[str, Any]
    data_completeness: int
    as_of: str
    exchange: str = ""
    sector: str = ""
    target_pct: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
