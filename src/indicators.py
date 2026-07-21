"""Technical-indicator calculations with no network dependencies."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _series(frame: pd.DataFrame, name: str) -> pd.Series:
    series = frame[name]
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    return pd.to_numeric(series, errors="coerce")


def compute_indicators(history: pd.DataFrame) -> pd.DataFrame:
    """Return a defensive copy with the indicators used by both engines."""
    if history is None or history.empty:
        return pd.DataFrame()

    df = history.copy()
    close = _series(df, "Close")
    high = _series(df, "High")
    low = _series(df, "Low")
    volume = _series(df, "Volume").fillna(0)

    df["EMA20"] = close.ewm(span=20, adjust=False, min_periods=10).mean()
    df["SMA50"] = close.rolling(50, min_periods=30).mean()
    df["SMA200"] = close.rolling(200, min_periods=150).mean()

    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = losses.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI14"] = (100 - (100 / (1 + rs))).fillna(50)

    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False, min_periods=9).mean()

    previous_close = close.shift(1)
    true_range = pd.concat(
        [(high - low).abs(), (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    df["ATR14"] = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    df["VOLUME_AVG20"] = volume.rolling(20, min_periods=10).mean()
    df["DOLLAR_VOLUME_AVG20"] = (close * volume).rolling(20, min_periods=10).mean()
    df["RETURN_1D"] = close.pct_change(fill_method=None)
    return df


def technical_levels(df: pd.DataFrame) -> tuple[float | None, float | None]:
    """Estimate nearby support and overhead resistance from recent price action."""
    if df is None or len(df) < 30:
        return None, None

    current = safe_float(_series(df, "Close").iloc[-1])
    if current is None:
        return None, None

    recent = df.iloc[-90:-2] if len(df) >= 92 else df.iloc[:-2]
    lows = _series(recent, "Low").dropna()
    highs = _series(recent, "High").dropna()
    if lows.empty or highs.empty:
        return None, None

    below = lows[lows < current]
    above = highs[highs > current]

    support_candidates = [
        safe_float(below.quantile(0.82)) if not below.empty else None,
        safe_float(df["EMA20"].iloc[-1]) if "EMA20" in df else None,
        safe_float(df["SMA50"].iloc[-1]) if "SMA50" in df else None,
    ]
    support_candidates = [x for x in support_candidates if x is not None and x < current]
    support = max(support_candidates) if support_candidates else safe_float(lows.min())

    resistance = safe_float(above.quantile(0.18)) if not above.empty else None
    if resistance is not None and resistance <= current * 1.002:
        higher = above[above > current * 1.002]
        resistance = safe_float(higher.min()) if not higher.empty else None
    return support, resistance


def trend_is_positive(history: pd.DataFrame) -> bool | None:
    df = compute_indicators(history)
    if df.empty or len(df) < 50:
        return None
    row = df.iloc[-1]
    close = safe_float(row.get("Close"))
    sma50 = safe_float(row.get("SMA50"))
    return None if close is None or sma50 is None else close > sma50


def safe_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None
