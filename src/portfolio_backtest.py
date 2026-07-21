"""Portfolio simulation and report serialization for multi-security backtests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import json

import numpy as np
import pandas as pd

from .backtest import BacktestResult


@dataclass(frozen=True)
class UniverseBacktestResult:
    """A portfolio-level result assembled from point-in-time security tests."""

    metadata: dict[str, Any]
    metrics: dict[str, Any]
    accepted_trades: pd.DataFrame
    skipped_trades: pd.DataFrame
    equity_curve: pd.DataFrame
    per_symbol: pd.DataFrame
    score_buckets: pd.DataFrame
    failures: pd.DataFrame


def _json_value(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if pd.isna(value):
        return None
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {key: _json_value(value) for key, value in row.items()}
        for row in frame.to_dict("records")
    ]


def _calibration(trades: pd.DataFrame) -> pd.DataFrame:
    columns = ["score_range", "trades", "target_rate", "profitable_rate", "average_return"]
    if trades.empty:
        return pd.DataFrame(columns=columns)
    work = trades.copy()
    work["score_range"] = pd.cut(
        work["score"], bins=[77, 84, 90, 100], labels=["78–84", "85–90", "91–100"], include_lowest=True
    )
    rows: list[dict[str, Any]] = []
    for label, group in work.groupby("score_range", observed=False):
        if group.empty:
            continue
        rows.append(
            {
                "score_range": str(label),
                "trades": int(len(group)),
                "target_rate": float((group["outcome"] == "TARGET").mean() * 100),
                "profitable_rate": float((group["return_pct"] > 0).mean() * 100),
                "average_return": float(group["return_pct"].mean()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_universe_portfolio(
    results: Iterable[BacktestResult],
    failures: Iterable[dict[str, str]] | None = None,
    *,
    initial_capital: float = 100_000,
    max_positions: int = 10,
    risk_per_trade_pct: float = 1.0,
    max_position_pct: float = 10.0,
    metadata: dict[str, Any] | None = None,
) -> UniverseBacktestResult:
    """Combine independent signals with cash, capacity, and risk-sizing constraints.

    Capital is sized so the published stop risks at most ``risk_per_trade_pct`` of
    current book equity, while any position is capped at ``max_position_pct``.
    Exits are processed before new entries on the same session.
    """
    result_list = list(results)
    failure_frame = pd.DataFrame(list(failures or []), columns=["symbol", "reason"])
    executed_frames: list[pd.DataFrame] = []
    per_symbol_rows: list[dict[str, Any]] = []

    for result in result_list:
        trades = result.trades.copy()
        executed = trades[trades["outcome"] != "EXPIRED"].copy() if not trades.empty else trades
        if not executed.empty:
            executed_frames.append(executed)
        per_symbol_rows.append(
            {
                "symbol": result.symbol,
                "signals": int(result.metrics.get("signals", 0)),
                "executed_candidates": int(result.metrics.get("executed", 0)),
                "target_rate": float(result.metrics.get("target_rate", 0)),
                "average_return": float(result.metrics.get("average_return", 0)),
                "profit_factor": result.metrics.get("profit_factor"),
                "stock_buy_hold_return": result.metrics.get("stock_buy_hold_return"),
            }
        )

    candidates = pd.concat(executed_frames, ignore_index=True) if executed_frames else pd.DataFrame()
    if not candidates.empty:
        candidates["entry_date"] = pd.to_datetime(candidates["entry_date"])
        candidates["exit_date"] = pd.to_datetime(candidates["exit_date"])
        candidates = candidates.sort_values(
            ["entry_date", "score", "planned_risk_reward", "symbol"],
            ascending=[True, False, False, True],
        ).reset_index(drop=True)

    cash = float(initial_capital)
    active: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    peak = float(initial_capital)

    entry_dates = sorted(candidates["entry_date"].dropna().unique()) if not candidates.empty else []
    for raw_date in entry_dates:
        date = pd.Timestamp(raw_date)
        still_active: list[dict[str, Any]] = []
        for position in active:
            if pd.Timestamp(position["exit_date"]) <= date:
                proceeds = position["allocation"] * (1 + position["return_pct"] / 100)
                cash += proceeds
                position["pnl"] = proceeds - position["allocation"]
                accepted.append(position)
            else:
                still_active.append(position)
        active = still_active

        day_candidates = candidates[candidates["entry_date"] == date]
        for _, trade in day_candidates.iterrows():
            base = trade.to_dict()
            if len(active) >= max_positions:
                skipped.append({**base, "skip_reason": "MAX_POSITIONS"})
                continue
            entry_price = float(trade["entry_price"])
            stop_loss = float(trade["stop_loss"])
            risk_fraction = (entry_price - stop_loss) / entry_price if entry_price > 0 else 0
            if risk_fraction <= 0:
                skipped.append({**base, "skip_reason": "INVALID_STOP"})
                continue

            book_equity = cash + sum(item["allocation"] for item in active)
            risk_budget = book_equity * risk_per_trade_pct / 100
            risk_sized = risk_budget / risk_fraction
            weight_cap = book_equity * max_position_pct / 100
            allocation = min(cash, risk_sized, weight_cap)
            if allocation < 1:
                skipped.append({**base, "skip_reason": "INSUFFICIENT_CASH"})
                continue
            cash -= allocation
            active.append(
                {
                    **base,
                    "allocation": float(allocation),
                    "position_weight_pct": float(allocation / book_equity * 100),
                    "risk_dollars": float(allocation * risk_fraction),
                }
            )

        book_equity = cash + sum(item["allocation"] for item in active)
        peak = max(peak, book_equity)
        curve.append(
            {
                "date": date,
                "equity": book_equity,
                "return_pct": (book_equity / initial_capital - 1) * 100,
                "drawdown_pct": (book_equity / peak - 1) * 100,
                "open_positions": len(active),
            }
        )

    for position in sorted(active, key=lambda item: pd.Timestamp(item["exit_date"])):
        proceeds = position["allocation"] * (1 + position["return_pct"] / 100)
        cash += proceeds
        position["pnl"] = proceeds - position["allocation"]
        accepted.append(position)
        book_equity = cash + sum(
            item["allocation"] for item in active if pd.Timestamp(item["exit_date"]) > pd.Timestamp(position["exit_date"])
        )
        peak = max(peak, book_equity)
        curve.append(
            {
                "date": pd.Timestamp(position["exit_date"]),
                "equity": book_equity,
                "return_pct": (book_equity / initial_capital - 1) * 100,
                "drawdown_pct": (book_equity / peak - 1) * 100,
                "open_positions": 0,
            }
        )

    accepted_frame = pd.DataFrame(accepted)
    skipped_frame = pd.DataFrame(skipped)
    # Rebuild the chart at every realized exit. Open positions remain at cost,
    # which is transparent and avoids inventing unobserved intraday marks.
    if accepted_frame.empty:
        curve_frame = pd.DataFrame(columns=["date", "equity", "return_pct", "drawdown_pct"])
    else:
        realized = accepted_frame.copy()
        realized["exit_date"] = pd.to_datetime(realized["exit_date"])
        realized = realized.groupby("exit_date", as_index=False)["pnl"].sum().sort_values("exit_date")
        realized["equity"] = initial_capital + realized["pnl"].cumsum()
        realized["return_pct"] = (realized["equity"] / initial_capital - 1) * 100
        realized["drawdown_pct"] = (realized["equity"] / realized["equity"].cummax() - 1) * 100
        curve_frame = realized.rename(columns={"exit_date": "date"}).drop(columns="pnl")
    per_symbol = pd.DataFrame(per_symbol_rows)
    if not accepted_frame.empty:
        accepted_counts = accepted_frame.groupby("symbol").agg(
            portfolio_trades=("symbol", "size"),
            portfolio_average_return=("return_pct", "mean"),
            portfolio_pnl=("pnl", "sum"),
        )
        per_symbol = per_symbol.merge(accepted_counts, how="left", left_on="symbol", right_index=True)
    for column in ("portfolio_trades", "portfolio_average_return", "portfolio_pnl"):
        if column not in per_symbol:
            per_symbol[column] = 0
        per_symbol[column] = per_symbol[column].fillna(0)

    count = int(len(accepted_frame))
    returns = accepted_frame["return_pct"] if count else pd.Series(dtype=float)
    gains = float(accepted_frame.loc[accepted_frame["pnl"] > 0, "pnl"].sum()) if count else 0.0
    losses = abs(float(accepted_frame.loc[accepted_frame["pnl"] < 0, "pnl"].sum())) if count else 0.0
    benchmark_values = [
        result.metrics.get("benchmark_return") for result in result_list if result.metrics.get("benchmark_return") is not None
    ]
    final_equity = float(initial_capital + accepted_frame["pnl"].sum()) if count else float(initial_capital)
    metrics = {
        "symbols_requested": len(result_list) + len(failure_frame),
        "symbols_succeeded": len(result_list),
        "symbols_failed": len(failure_frame),
        "candidate_trades": int(len(candidates)),
        "accepted_trades": count,
        "skipped_trades": int(len(skipped_frame)),
        "skipped_capacity": int((skipped_frame.get("skip_reason") == "MAX_POSITIONS").sum()) if not skipped_frame.empty else 0,
        "skipped_cash": int((skipped_frame.get("skip_reason") == "INSUFFICIENT_CASH").sum()) if not skipped_frame.empty else 0,
        "targets": int((accepted_frame.get("outcome") == "TARGET").sum()) if count else 0,
        "stops": int((accepted_frame.get("outcome") == "STOP").sum()) if count else 0,
        "time_exits": int((accepted_frame.get("outcome") == "TIME_EXIT").sum()) if count else 0,
        "target_rate": float((accepted_frame["outcome"] == "TARGET").mean() * 100) if count else 0.0,
        "profitable_rate": float((returns > 0).mean() * 100) if count else 0.0,
        "average_return": float(returns.mean()) if count else 0.0,
        "profit_factor": gains / losses if losses > 0 else (float("inf") if gains > 0 else None),
        "initial_capital": float(initial_capital),
        "final_equity": final_equity,
        "total_return": (final_equity / initial_capital - 1) * 100,
        "max_drawdown": float(curve_frame["drawdown_pct"].min()) if not curve_frame.empty else 0.0,
        "benchmark_return": float(np.median(benchmark_values)) if benchmark_values else None,
    }
    return UniverseBacktestResult(
        metadata={
            "max_positions": max_positions,
            "risk_per_trade_pct": risk_per_trade_pct,
            "max_position_pct": max_position_pct,
            **(metadata or {}),
        },
        metrics=metrics,
        accepted_trades=accepted_frame,
        skipped_trades=skipped_frame,
        equity_curve=curve_frame,
        per_symbol=per_symbol,
        score_buckets=_calibration(accepted_frame),
        failures=failure_frame,
    )


def save_universe_report(result: UniverseBacktestResult, path: str | Path) -> Path:
    """Save the complete UI report as one portable JSON file."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ledger_columns = [
        "symbol", "signal_date", "score", "entry_date", "entry_price", "stop_loss",
        "target_price", "exit_date", "exit_price", "outcome", "return_pct",
        "holding_days", "allocation", "position_weight_pct", "risk_dollars", "pnl",
    ]
    ledger = result.accepted_trades[
        [column for column in ledger_columns if column in result.accepted_trades]
    ].copy()
    payload = {
        "metadata": {key: _json_value(value) for key, value in result.metadata.items()},
        "metrics": {key: _json_value(value) for key, value in result.metrics.items()},
        "equity_curve": _records(result.equity_curve),
        "accepted_trades": _records(ledger),
        "per_symbol": _records(result.per_symbol),
        "score_buckets": _records(result.score_buckets),
        "failures": _records(result.failures),
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def load_universe_report(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
