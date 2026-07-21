"""Market-data access and resilient ticker/company autocomplete."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import yfinance as yf

from .config import DATA_CACHE_TTL_SECONDS, SECTOR_ETFS


PROJECT_ROOT = Path(__file__).resolve().parent.parent
UNIVERSE_PATH = PROJECT_ROOT / "data" / "stock_universe.csv"


@st.cache_data(show_spinner=False)
def load_local_universe() -> pd.DataFrame:
    frame = pd.read_csv(UNIVERSE_PATH, dtype=str).fillna("")
    for column in ("symbol", "name", "exchange", "type"):
        frame[column] = frame[column].str.strip()
    return frame.drop_duplicates(subset=["symbol"])


def _candidate(symbol: str, name: str, exchange: str, quote_type: str, source: str) -> dict[str, str]:
    return {
        "symbol": symbol.strip().upper(),
        "name": name.strip() or symbol.strip().upper(),
        "exchange": exchange.strip(),
        "type": quote_type.strip() or "EQUITY",
        "source": source,
    }


@st.cache_data(ttl=3600, show_spinner=False)
def _remote_search(query: str) -> list[dict[str, str]]:
    try:
        quotes = yf.Search(query, max_results=14, news_count=0).quotes
    except Exception:
        return []

    candidates: list[dict[str, str]] = []
    allowed = {"EQUITY", "ETF"}
    for item in quotes or []:
        quote_type = str(item.get("quoteType") or item.get("typeDisp") or "").upper()
        if quote_type not in allowed:
            continue
        symbol = str(item.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        candidates.append(
            _candidate(
                symbol,
                str(item.get("longname") or item.get("shortname") or symbol),
                str(item.get("exchDisp") or item.get("exchange") or ""),
                quote_type,
                "remote",
            )
        )
    return candidates


def search_securities(query: str, limit: int = 12) -> list[dict[str, str]]:
    """Search the bundled universe first, then enrich from yfinance Search."""
    clean = " ".join((query or "").strip().split())
    if not clean:
        return []
    upper = clean.upper()
    lower = clean.lower()
    universe = load_local_universe()

    def match_score(row: pd.Series) -> tuple[int, int, str]:
        symbol = str(row["symbol"]).upper()
        name = str(row["name"]).lower()
        if symbol.startswith(upper):
            rank = 0
        elif name.startswith(lower):
            rank = 1
        elif upper in symbol:
            rank = 2
        elif lower in name:
            rank = 3
        else:
            rank = 9
        return rank, len(symbol), symbol

    mask = universe["symbol"].str.upper().str.contains(upper, regex=False) | universe[
        "name"
    ].str.lower().str.contains(lower, regex=False)
    local_rows = [row for _, row in universe[mask].iterrows()]
    local_rows.sort(key=match_score)
    merged = [
        _candidate(row["symbol"], row["name"], row["exchange"], row["type"], "local")
        for row in local_rows[:limit]
    ]

    if len(clean) >= 2:
        for item in _remote_search(clean):
            if not any(existing["symbol"] == item["symbol"] for existing in merged):
                merged.append(item)

    merged.sort(
        key=lambda item: (
            0 if item["symbol"].startswith(upper) else 1 if item["name"].lower().startswith(lower) else 2,
            len(item["symbol"]),
            item["symbol"],
        )
    )
    return merged[:limit]


def _clean_history(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    result = frame.copy()
    if isinstance(result.columns, pd.MultiIndex):
        result.columns = result.columns.get_level_values(0)
    required = ["Open", "High", "Low", "Close", "Volume"]
    if not all(column in result.columns for column in required):
        return pd.DataFrame()
    result = result[required].apply(pd.to_numeric, errors="coerce")
    result = result.dropna(subset=["Close", "High", "Low"])
    if getattr(result.index, "tz", None) is not None:
        result.index = result.index.tz_localize(None)
    return result[~result.index.duplicated(keep="last")].sort_index()


def _safe_info(ticker: yf.Ticker) -> dict[str, Any]:
    try:
        info = ticker.get_info()
        return info if isinstance(info, dict) else {}
    except Exception:
        return {}


def _safe_fast_info(ticker: yf.Ticker) -> dict[str, Any]:
    try:
        return dict(ticker.fast_info)
    except Exception:
        return {}


def _safe_calendar(ticker: yf.Ticker) -> dict[str, Any]:
    try:
        calendar = ticker.calendar
        return calendar if isinstance(calendar, dict) else {}
    except Exception:
        return {}


@st.cache_data(ttl=DATA_CACHE_TTL_SECONDS, show_spinner=False)
def load_market_bundle(symbol: str) -> dict[str, Any]:
    """Fetch price, benchmark, company, and optional sector context."""
    clean_symbol = (symbol or "").strip().upper()
    if not clean_symbol or len(clean_symbol) > 20:
        raise ValueError("Invalid symbol")

    ticker = yf.Ticker(clean_symbol)
    try:
        history = _clean_history(
            ticker.history(period="2y", interval="1d", auto_adjust=False, timeout=15)
        )
    except Exception as exc:
        raise RuntimeError(f"Unable to load history for {clean_symbol}") from exc
    if history.empty:
        raise RuntimeError(f"No history returned for {clean_symbol}")

    info = _safe_info(ticker)
    fast_info = _safe_fast_info(ticker)
    calendar = _safe_calendar(ticker)

    benchmark = pd.DataFrame()
    try:
        benchmark = _clean_history(
            yf.Ticker("SPY").history(period="1y", interval="1d", auto_adjust=False, timeout=12)
        )
    except Exception:
        pass

    sector_history = pd.DataFrame()
    sector_etf = SECTOR_ETFS.get(str(info.get("sector") or ""))
    if sector_etf:
        try:
            sector_history = _clean_history(
                yf.Ticker(sector_etf).history(period="1y", interval="1d", auto_adjust=False, timeout=12)
            )
        except Exception:
            pass

    return {
        "symbol": clean_symbol,
        "history": history,
        "benchmark": benchmark,
        "sector_history": sector_history,
        "sector_etf": sector_etf,
        "info": info,
        "fast_info": fast_info,
        "calendar": calendar,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
    }


@st.cache_data(ttl=3600, show_spinner=False)
def load_backtest_bundle(symbol: str, period: str = "5y") -> dict[str, Any]:
    """Fetch adjusted daily history for conservative point-in-time testing."""
    clean_symbol = (symbol or "").strip().upper()
    allowed_periods = {"2y", "5y", "10y"}
    clean_period = period if period in allowed_periods else "5y"
    if not clean_symbol or len(clean_symbol) > 20:
        raise ValueError("Invalid symbol")

    ticker = yf.Ticker(clean_symbol)
    try:
        history = _clean_history(
            ticker.history(period=clean_period, interval="1d", auto_adjust=True, timeout=20)
        )
    except Exception as exc:
        raise RuntimeError(f"Unable to load backtest history for {clean_symbol}") from exc
    if history.empty:
        raise RuntimeError(f"No backtest history returned for {clean_symbol}")

    info = _safe_info(ticker)
    benchmark = pd.DataFrame()
    try:
        benchmark = _clean_history(
            yf.Ticker("SPY").history(period=clean_period, interval="1d", auto_adjust=True, timeout=15)
        )
    except Exception:
        pass

    sector_history = pd.DataFrame()
    sector_etf = SECTOR_ETFS.get(str(info.get("sector") or ""))
    if sector_etf:
        try:
            sector_history = _clean_history(
                yf.Ticker(sector_etf).history(
                    period=clean_period,
                    interval="1d",
                    auto_adjust=True,
                    timeout=15,
                )
            )
        except Exception:
            pass

    return {
        "symbol": clean_symbol,
        "history": history,
        "benchmark": benchmark,
        "sector_history": sector_history,
        "sector_etf": sector_etf,
        "info": info,
        "fast_info": {"currency": info.get("currency") or "USD"},
        "calendar": {},
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
    }
