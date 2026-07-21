"""Conservative, point-in-time swing backtesting for Stock Decision Pro."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import MIN_HISTORY_ROWS
from .engines import analyze_swing
from .indicators import compute_indicators, safe_float


@dataclass(frozen=True)
class BacktestResult:
    symbol: str
    target_pct: float
    holding_days: int
    entry_window: int
    slippage_pct: float
    start_date: str
    end_date: str
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    score_buckets: pd.DataFrame
    metrics: dict[str, float | int | str | None]


def _slice_to_date(frame: pd.DataFrame, date: pd.Timestamp, rows: int = 280) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    return frame.loc[frame.index <= date].tail(rows).copy()


def _find_entry(
    future: pd.DataFrame,
    entry_low: float,
    entry_high: float,
) -> tuple[pd.Timestamp, float] | None:
    """Return the first realistic fill inside the published entry zone."""
    for date, row in future.iterrows():
        open_price = safe_float(row.get("Open"))
        high = safe_float(row.get("High"))
        low = safe_float(row.get("Low"))
        if open_price is None or high is None or low is None:
            continue
        if entry_low <= open_price <= entry_high:
            return pd.Timestamp(date), open_price
        if open_price > entry_high and low <= entry_high:
            return pd.Timestamp(date), entry_high
        if open_price < entry_low and high >= entry_low:
            return pd.Timestamp(date), entry_low
    return None


def _simulate_exit(
    future: pd.DataFrame,
    entry_date: pd.Timestamp,
    entry_price: float,
    stop_loss: float,
    target_price: float,
    slippage_pct: float,
) -> dict[str, Any]:
    """Conservatively assume the stop is hit first when both levels trade in one daily bar."""
    slip = slippage_pct / 100
    entry_with_cost = entry_price * (1 + slip)
    holding = future.loc[future.index >= entry_date]
    outcome = "TIME_EXIT"
    exit_date = pd.Timestamp(holding.index[-1])
    exit_raw = safe_float(holding.iloc[-1].get("Close")) or entry_price

    for date, row in holding.iterrows():
        open_price = safe_float(row.get("Open")) or entry_price
        high = safe_float(row.get("High")) or open_price
        low = safe_float(row.get("Low")) or open_price
        stop_hit = low <= stop_loss
        target_hit = high >= target_price
        if stop_hit:
            outcome = "STOP"
            exit_date = pd.Timestamp(date)
            exit_raw = open_price if open_price < stop_loss else stop_loss
            break
        if target_hit:
            outcome = "TARGET"
            exit_date = pd.Timestamp(date)
            exit_raw = target_price
            break

    exit_with_cost = exit_raw * (1 - slip)
    return_pct = ((exit_with_cost / entry_with_cost) - 1) * 100
    risk = entry_with_cost - stop_loss
    r_multiple = (exit_with_cost - entry_with_cost) / risk if risk > 0 else None
    return {
        "entry_price": entry_with_cost,
        "exit_date": exit_date,
        "exit_price": exit_with_cost,
        "outcome": outcome,
        "return_pct": return_pct,
        "r_multiple": r_multiple,
        "holding_days": int(holding.index.get_loc(exit_date)) + 1,
    }


def _score_buckets(executed: pd.DataFrame) -> pd.DataFrame:
    columns = ["score_range", "trades", "target_rate", "profitable_rate", "average_return"]
    if executed.empty:
        return pd.DataFrame(columns=columns)
    work = executed.copy()
    work["score_range"] = pd.cut(
        work["score"],
        bins=[77, 84, 90, 100],
        labels=["78–84", "85–90", "91–100"],
        include_lowest=True,
    )
    rows: list[dict[str, Any]] = []
    for label, group in work.groupby("score_range", observed=False):
        valid = group.dropna(subset=["return_pct"])
        if valid.empty:
            continue
        rows.append(
            {
                "score_range": str(label),
                "trades": int(len(valid)),
                "target_rate": float((valid["outcome"] == "TARGET").mean() * 100),
                "profitable_rate": float((valid["return_pct"] > 0).mean() * 100),
                "average_return": float(valid["return_pct"].mean()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _metrics(
    trades: pd.DataFrame,
    equity_curve: pd.DataFrame,
    stock_buy_hold_return: float | None,
    benchmark_return: float | None,
) -> dict[str, float | int | str | None]:
    executed = trades[trades["outcome"] != "EXPIRED"].copy() if not trades.empty else trades.copy()
    signals = int(len(trades))
    count = int(len(executed))
    targets = int((executed["outcome"] == "TARGET").sum()) if count else 0
    stops = int((executed["outcome"] == "STOP").sum()) if count else 0
    timed = int((executed["outcome"] == "TIME_EXIT").sum()) if count else 0
    expired = signals - count
    returns = executed["return_pct"].dropna() if count else pd.Series(dtype=float)
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    profit_factor = float(gains / losses) if losses > 0 else (float("inf") if gains > 0 else None)
    total_return = float((np.prod(1 + returns / 100) - 1) * 100) if not returns.empty else 0.0
    max_drawdown = 0.0
    if not equity_curve.empty:
        max_drawdown = float(equity_curve["drawdown_pct"].min())
    sample_status = "INSUFFICIENT" if count < 30 else "EARLY" if count < 100 else "ESTABLISHED"
    return {
        "signals": signals,
        "executed": count,
        "expired": expired,
        "targets": targets,
        "stops": stops,
        "time_exits": timed,
        "target_rate": float(targets / count * 100) if count else 0.0,
        "profitable_rate": float((returns > 0).mean() * 100) if count else 0.0,
        "average_return": float(returns.mean()) if count else 0.0,
        "median_return": float(returns.median()) if count else 0.0,
        "total_return": total_return,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "average_holding_days": float(executed["holding_days"].mean()) if count else 0.0,
        "stock_buy_hold_return": stock_buy_hold_return,
        "benchmark_return": benchmark_return,
        "sample_status": sample_status,
    }


def run_swing_backtest(
    bundle: dict[str, Any],
    target_pct: float = 5,
    holding_days: int = 10,
    entry_window: int = 3,
    slippage_pct: float = 0.10,
) -> BacktestResult:
    """Run the current swing engine on rolling historical snapshots.

    A signal is calculated after each daily close. The simulated order may fill only
    inside the published entry zone during the following ``entry_window`` sessions.
    Only one trade can be open at a time for the tested symbol.
    """
    history = compute_indicators(bundle.get("history", pd.DataFrame()).copy().sort_index())
    if len(history) < MIN_HISTORY_ROWS + holding_days + entry_window + 2:
        raise ValueError("Not enough history for a point-in-time backtest")

    symbol = str(bundle.get("symbol") or "").upper()
    info = bundle.get("info", {})
    identity_info = {
        "longName": info.get("longName") or info.get("shortName") or symbol,
        "shortName": info.get("shortName") or symbol,
        "currency": info.get("currency") or "USD",
        "fullExchangeName": info.get("fullExchangeName") or info.get("exchange") or "",
        "sector": info.get("sector") or "",
    }
    benchmark = compute_indicators(bundle.get("benchmark", pd.DataFrame()))
    sector_history = compute_indicators(bundle.get("sector_history", pd.DataFrame()))
    rows: list[dict[str, Any]] = []
    first_signal_index = MIN_HISTORY_ROWS - 1
    last_signal_index = len(history) - entry_window - holding_days - 1
    cursor = first_signal_index

    while cursor <= last_signal_index:
        signal_date = pd.Timestamp(history.index[cursor])
        point_in_time_bundle = {
            "symbol": symbol,
            "history": history.iloc[max(0, cursor - 279) : cursor + 1].copy(),
            "benchmark": _slice_to_date(benchmark, signal_date),
            "sector_history": _slice_to_date(sector_history, signal_date),
            "sector_etf": bundle.get("sector_etf"),
            "info": identity_info,
            "fast_info": {"currency": identity_info["currency"]},
            "calendar": {},
        }
        signal = analyze_swing(point_in_time_bundle, target_pct=target_pct)
        if signal.decision != "ENTER":
            cursor += 1
            continue

        entry_slice = history.iloc[cursor + 1 : cursor + 1 + entry_window]
        entry = _find_entry(entry_slice, float(signal.entry_low), float(signal.entry_high))
        base = {
            "symbol": symbol,
            "signal_date": signal_date,
            "score": signal.score,
            "entry_low": signal.entry_low,
            "entry_high": signal.entry_high,
            "stop_loss": signal.stop_loss,
            "target_price": signal.target_1,
            "planned_risk_reward": signal.risk_reward,
        }
        if entry is None:
            rows.append(
                {
                    **base,
                    "entry_date": pd.NaT,
                    "entry_price": np.nan,
                    "exit_date": pd.NaT,
                    "exit_price": np.nan,
                    "outcome": "EXPIRED",
                    "return_pct": np.nan,
                    "r_multiple": np.nan,
                    "holding_days": 0,
                }
            )
            cursor += entry_window
            continue

        entry_date, entry_raw = entry
        entry_position = int(history.index.get_loc(entry_date))
        evaluation = history.iloc[entry_position : entry_position + holding_days]
        simulated = _simulate_exit(
            evaluation,
            entry_date,
            entry_raw,
            float(signal.stop_loss),
            float(signal.target_1),
            slippage_pct,
        )
        rows.append({**base, "entry_date": entry_date, **simulated})
        exit_position = int(history.index.get_loc(simulated["exit_date"]))
        cursor = max(cursor + 1, exit_position + 1)

    columns = [
        "symbol",
        "signal_date",
        "score",
        "entry_low",
        "entry_high",
        "entry_date",
        "entry_price",
        "stop_loss",
        "target_price",
        "exit_date",
        "exit_price",
        "outcome",
        "return_pct",
        "r_multiple",
        "holding_days",
        "planned_risk_reward",
    ]
    trades = pd.DataFrame(rows, columns=columns)
    executed = trades[trades["outcome"] != "EXPIRED"].copy() if not trades.empty else trades.copy()
    equity_curve = pd.DataFrame(columns=["date", "equity", "drawdown_pct"])
    if not executed.empty:
        executed = executed.sort_values("exit_date")
        equity = (1 + executed["return_pct"] / 100).cumprod()
        peaks = equity.cummax()
        equity_curve = pd.DataFrame(
            {
                "date": executed["exit_date"].to_list(),
                "equity": equity.to_list(),
                "drawdown_pct": ((equity / peaks) - 1).mul(100).to_list(),
            }
        )

    stock_buy_hold_return = None
    start_close = safe_float(history["Close"].iloc[first_signal_index])
    end_close = safe_float(history["Close"].iloc[-1])
    if start_close and end_close:
        stock_buy_hold_return = ((end_close / start_close) - 1) * 100

    benchmark_return = None
    benchmark_window = benchmark.loc[
        (benchmark.index >= history.index[first_signal_index]) & (benchmark.index <= history.index[-1])
    ] if benchmark is not None and not benchmark.empty else pd.DataFrame()
    if not benchmark_window.empty:
        benchmark_start = safe_float(benchmark_window["Close"].iloc[0])
        benchmark_end = safe_float(benchmark_window["Close"].iloc[-1])
        if benchmark_start and benchmark_end:
            benchmark_return = ((benchmark_end / benchmark_start) - 1) * 100

    metrics = _metrics(trades, equity_curve, stock_buy_hold_return, benchmark_return)
    return BacktestResult(
        symbol=symbol,
        target_pct=target_pct,
        holding_days=holding_days,
        entry_window=entry_window,
        slippage_pct=slippage_pct,
        start_date=history.index[first_signal_index].strftime("%Y-%m-%d"),
        end_date=history.index[-1].strftime("%Y-%m-%d"),
        trades=trades,
        equity_curve=equity_curve,
        score_buckets=_score_buckets(executed),
        metrics=metrics,
    )
