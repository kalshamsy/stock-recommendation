from __future__ import annotations

import pandas as pd
import pytest

from src.backtest import BacktestResult
from src.portfolio_backtest import build_universe_portfolio


def result_for(symbol: str, trades: list[dict]) -> BacktestResult:
    frame = pd.DataFrame(trades)
    return BacktestResult(
        symbol=symbol,
        target_pct=5,
        holding_days=10,
        entry_window=3,
        slippage_pct=0.1,
        start_date="2024-01-01",
        end_date="2025-01-01",
        trades=frame,
        equity_curve=pd.DataFrame(),
        score_buckets=pd.DataFrame(),
        metrics={
            "signals": len(frame),
            "executed": len(frame),
            "target_rate": float((frame["outcome"] == "TARGET").mean() * 100),
            "average_return": float(frame["return_pct"].mean()),
            "profit_factor": 2.0,
            "stock_buy_hold_return": 10.0,
            "benchmark_return": 8.0,
        },
    )


def trade(entry: str, exit_: str, outcome: str = "TARGET", return_pct: float = 5.0, score: int = 85) -> dict:
    return {
        "symbol": "",
        "signal_date": pd.Timestamp(entry) - pd.Timedelta(days=1),
        "score": score,
        "entry_low": 99.0,
        "entry_high": 101.0,
        "entry_date": pd.Timestamp(entry),
        "entry_price": 100.0,
        "stop_loss": 95.0,
        "target_price": 105.0,
        "exit_date": pd.Timestamp(exit_),
        "exit_price": 105.0 if return_pct > 0 else 95.0,
        "outcome": outcome,
        "return_pct": return_pct,
        "r_multiple": return_pct / 5,
        "holding_days": 5,
        "planned_risk_reward": 1.0,
    }


def test_portfolio_enforces_concurrent_position_limit() -> None:
    first = trade("2025-01-02", "2025-01-10", score=90)
    second = trade("2025-01-02", "2025-01-10", score=80)
    first["symbol"], second["symbol"] = "AAA", "BBB"
    report = build_universe_portfolio(
        [result_for("AAA", [first]), result_for("BBB", [second])], max_positions=1
    )
    assert report.metrics["accepted_trades"] == 1
    assert report.metrics["skipped_capacity"] == 1
    assert report.accepted_trades.iloc[0]["symbol"] == "AAA"


def test_risk_sizing_caps_loss_to_one_percent_of_equity() -> None:
    losing = trade("2025-01-02", "2025-01-10", outcome="STOP", return_pct=-5.0)
    losing["symbol"] = "AAA"
    report = build_universe_portfolio(
        [result_for("AAA", [losing])], risk_per_trade_pct=1.0, max_position_pct=20.0
    )
    accepted = report.accepted_trades.iloc[0]
    assert accepted["allocation"] == pytest.approx(20_000)
    assert accepted["risk_dollars"] == pytest.approx(1_000)
    assert report.metrics["total_return"] == pytest.approx(-1.0)


def test_same_day_exit_releases_slot_before_new_entry() -> None:
    first = trade("2025-01-02", "2025-01-10")
    second = trade("2025-01-10", "2025-01-17")
    first["symbol"], second["symbol"] = "AAA", "BBB"
    report = build_universe_portfolio(
        [result_for("AAA", [first]), result_for("BBB", [second])], max_positions=1
    )
    assert report.metrics["accepted_trades"] == 2
    assert report.metrics["skipped_capacity"] == 0
    assert report.metrics["total_return"] > 0.9
