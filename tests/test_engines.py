from __future__ import annotations

import numpy as np
import pandas as pd

from src.engines import analyze_investment, analyze_swing
from src.indicators import compute_indicators


def make_history(days: int = 300, start: float = 100, end: float = 150) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-01", periods=days)
    close = np.linspace(start, end, days) + np.sin(np.linspace(0, 12, days)) * 1.8
    open_ = close * (1 + np.sin(np.linspace(0, 8, days)) * 0.002)
    high = np.maximum(open_, close) * 1.008
    low = np.minimum(open_, close) * 0.992
    volume = np.full(days, 4_000_000.0)
    volume[-1] = 5_000_000
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=index,
    )


def make_bundle() -> dict:
    history = make_history()
    benchmark = make_history(start=400, end=510)
    return {
        "symbol": "TEST",
        "history": history,
        "benchmark": benchmark,
        "sector_history": benchmark,
        "sector_etf": "XLK",
        "fast_info": {"currency": "USD"},
        "calendar": {},
        "info": {
            "longName": "Test Company",
            "currency": "USD",
            "fullExchangeName": "NASDAQ",
            "sector": "Technology",
            "revenueGrowth": 0.14,
            "earningsGrowth": 0.18,
            "profitMargins": 0.22,
            "returnOnEquity": 0.24,
            "freeCashflow": 5_000_000_000,
            "debtToEquity": 45,
            "currentRatio": 1.8,
            "trailingPE": 24,
            "forwardPE": 20,
            "pegRatio": 1.4,
            "marketCap": 150_000_000_000,
            "targetMeanPrice": 175,
            "fiftyTwoWeekHigh": 168,
        },
    }


def test_indicators_are_added_without_mutating_input() -> None:
    history = make_history()
    original_columns = list(history.columns)
    calculated = compute_indicators(history)
    assert list(history.columns) == original_columns
    assert {"EMA20", "SMA50", "SMA200", "RSI14", "ATR14", "MACD"}.issubset(calculated.columns)
    assert calculated["RSI14"].iloc[-1] > 0


def test_swing_engine_returns_bounded_explainable_result() -> None:
    result = analyze_swing(make_bundle(), target_pct=5)
    assert result.symbol == "TEST"
    assert result.mode == "swing"
    assert 0 <= result.score <= 100
    assert result.decision in {"ENTER", "WAIT", "AVOID"}
    assert result.entry_low < result.entry_high
    assert result.stop_loss < result.current_price < result.target_1
    assert sum(result.breakdown.values()) <= 100
    assert result.positives


def test_investment_engine_uses_fundamentals() -> None:
    result = analyze_investment(make_bundle())
    assert result.mode == "investment"
    assert result.data_completeness == 100
    assert result.decision in {"ENTER", "GRADUAL", "WAIT", "AVOID"}
    assert round(result.fundamentals["revenue_growth"], 2) == 14
    assert result.target_1 == 175
    assert "growth" in result.breakdown


def test_missing_fundamentals_force_wait() -> None:
    bundle = make_bundle()
    bundle["info"] = {"longName": "Sparse Company", "currency": "USD"}
    result = analyze_investment(bundle)
    assert result.decision == "WAIT"
    assert any(item.key == "fundamentals_limited" for item in result.blockers)
