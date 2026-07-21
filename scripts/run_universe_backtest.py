#!/usr/bin/env python3
"""Generate the fixed-universe report used by the Streamlit evaluation center."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import run_swing_backtest  # noqa: E402
from src.portfolio_backtest import build_universe_portfolio, save_universe_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", default="5y", choices=["2y", "5y", "10y"])
    parser.add_argument("--target", type=float, default=5)
    parser.add_argument("--holding-days", type=int, default=10)
    parser.add_argument("--entry-window", type=int, default=3)
    parser.add_argument("--slippage", type=float, default=0.10)
    parser.add_argument("--max-positions", type=int, default=10)
    parser.add_argument("--risk-per-trade", type=float, default=1.0)
    parser.add_argument("--max-position", type=float, default=10.0)
    parser.add_argument("--initial-capital", type=float, default=100_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-symbols", type=int, default=None)
    parser.add_argument("--output", default=str(ROOT / "data" / "universe_report.json"))
    return parser.parse_args()


def _extract(download: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if download.empty:
        return pd.DataFrame()
    if isinstance(download.columns, pd.MultiIndex):
        if symbol not in download.columns.get_level_values(0):
            return pd.DataFrame()
        frame = download[symbol].copy()
    else:
        frame = download.copy()
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if not all(column in frame for column in needed):
        return pd.DataFrame()
    frame = frame[needed].dropna(subset=["Open", "High", "Low", "Close"])
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    return frame


def _worker(payload: tuple[str, pd.DataFrame, pd.DataFrame, argparse.Namespace]):
    symbol, history, benchmark, args = payload
    result = run_swing_backtest(
        {"symbol": symbol, "history": history, "benchmark": benchmark, "info": {"shortName": symbol}},
        target_pct=args.target,
        holding_days=args.holding_days,
        entry_window=args.entry_window,
        slippage_pct=args.slippage,
    )
    return result


def main() -> int:
    args = parse_args()
    universe = pd.read_csv(ROOT / "data" / "stock_universe.csv")
    symbols = universe["symbol"].dropna().astype(str).str.upper().drop_duplicates().tolist()
    if args.max_symbols:
        symbols = symbols[: args.max_symbols]
    requested = symbols.copy()
    download_symbols = list(dict.fromkeys(symbols + ["SPY"]))
    print(f"Downloading {len(download_symbols)} symbols for {args.period}…", flush=True)
    raw = yf.download(
        download_symbols,
        period=args.period,
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    benchmark = _extract(raw, "SPY")
    payloads = []
    failures: list[dict[str, str]] = []
    for symbol in symbols:
        history = _extract(raw, symbol)
        if history.empty:
            failures.append({"symbol": symbol, "reason": "NO_PRICE_HISTORY"})
        else:
            payloads.append((symbol, history, benchmark, args))

    results = []
    # Threads avoid platform semaphore restrictions on managed runners. The
    # indicator calculations rely heavily on vectorized pandas/numpy operations.
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_map = {executor.submit(_worker, payload): payload[0] for payload in payloads}
        total = len(future_map)
        for completed, future in enumerate(as_completed(future_map), start=1):
            symbol = future_map[future]
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append({"symbol": symbol, "reason": f"{type(exc).__name__}: {exc}"})
            if completed % 10 == 0 or completed == total:
                print(f"Evaluated {completed}/{total} symbols…", flush=True)

    dates = [result.start_date for result in results] + [result.end_date for result in results]
    report = build_universe_portfolio(
        results,
        failures,
        initial_capital=args.initial_capital,
        max_positions=args.max_positions,
        risk_per_trade_pct=args.risk_per_trade,
        max_position_pct=args.max_position,
        metadata={
            "engine_version": "1.2.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "universe_name": "Bundled fixed universe",
            "universe_symbols": len(requested),
            "period": args.period,
            "start_date": min(dates) if dates else None,
            "end_date": max(dates) if dates else None,
            "target_pct": args.target,
            "holding_days": args.holding_days,
            "entry_window": args.entry_window,
            "slippage_pct": args.slippage,
            "survivorship_bias": True,
        },
    )
    output = save_universe_report(report, args.output)
    print(f"Saved {output}", flush=True)
    print(report.metrics, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
