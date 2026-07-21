from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import src.backtest as backtest


def price_history(days: int = 260) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-01", periods=days)
    close = np.linspace(90, 110, days)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": np.full(days, 3_000_000.0),
        },
        index=index,
    )


def test_same_bar_target_and_stop_uses_conservative_stop() -> None:
    day = pd.Timestamp("2025-01-02")
    future = pd.DataFrame(
        {"Open": [100.0], "High": [106.0], "Low": [94.0], "Close": [102.0], "Volume": [1_000_000]},
        index=[day],
    )
    outcome = backtest._simulate_exit(future, day, 100, 95, 105, 0)
    assert outcome["outcome"] == "STOP"
    assert outcome["exit_price"] == 95
    assert outcome["return_pct"] == pytest.approx(-5)


def test_backtest_uses_only_data_available_at_each_signal(monkeypatch) -> None:
    history = price_history()
    first_future_day = history.index[210]
    history.loc[first_future_day, ["Open", "High", "Low", "Close"]] = [100, 106, 98, 104]
    calls: list[pd.Timestamp] = []

    def fake_engine(bundle: dict, target_pct: float):
        latest = pd.Timestamp(bundle["history"].index[-1])
        calls.append(latest)
        if len(calls) == 1:
            return SimpleNamespace(
                decision="ENTER",
                entry_low=99.0,
                entry_high=101.0,
                stop_loss=95.0,
                target_1=105.0,
                score=82,
                risk_reward=2.0,
            )
        return SimpleNamespace(decision="AVOID")

    monkeypatch.setattr(backtest, "analyze_swing", fake_engine)
    bundle = {
        "symbol": "TEST",
        "history": history,
        "benchmark": history,
        "sector_history": history,
        "info": {"longName": "Test", "currency": "USD"},
    }
    result = backtest.run_swing_backtest(bundle, target_pct=5, holding_days=10, entry_window=3, slippage_pct=0)

    assert calls[0] == history.index[209]
    assert all(date <= history.index[-14] for date in calls)
    assert result.metrics["signals"] == 1
    assert result.metrics["executed"] == 1
    assert result.metrics["targets"] == 1
    assert result.trades.iloc[0]["entry_date"] == first_future_day
    assert result.trades.iloc[0]["outcome"] == "TARGET"
    assert result.trades.iloc[0]["return_pct"] == pytest.approx(5)


def test_entry_requires_a_touch_inside_published_zone() -> None:
    future = pd.DataFrame(
        {
            "Open": [105.0, 104.0, 103.0],
            "High": [106.0, 105.0, 104.0],
            "Low": [103.0, 102.0, 101.5],
            "Close": [104.0, 103.0, 102.0],
            "Volume": [1_000_000] * 3,
        },
        index=pd.bdate_range("2025-01-02", periods=3),
    )
    assert backtest._find_entry(future, 99, 101) is None
