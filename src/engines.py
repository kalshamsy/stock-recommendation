"""Deterministic Investment and Swing Trading decision engines."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .config import MIN_AVERAGE_DOLLAR_VOLUME, MIN_HISTORY_ROWS
from .indicators import compute_indicators, safe_float, technical_levels, trend_is_positive
from .models import AnalysisResult, Factor


def _factor(key: str, **values: Any) -> Factor:
    return Factor(key=key, values=values)


def _clamp(value: float, low: float = 0, high: float = 100) -> float:
    return max(low, min(high, value))


def _pct(value: Any) -> float | None:
    number = safe_float(value)
    return number * 100 if number is not None else None


def _human_number(value: Any, currency: str = "USD") -> str:
    number = safe_float(value)
    if number is None:
        return "—"
    absolute = abs(number)
    if absolute >= 1_000_000_000_000:
        compact = f"{number / 1_000_000_000_000:.1f}T"
    elif absolute >= 1_000_000_000:
        compact = f"{number / 1_000_000_000:.1f}B"
    elif absolute >= 1_000_000:
        compact = f"{number / 1_000_000:.1f}M"
    else:
        compact = f"{number:,.0f}"
    return f"{currency} {compact}"


def _days_to_earnings(calendar: dict[str, Any], info: dict[str, Any]) -> int | None:
    candidates: list[Any] = []
    for key in ("Earnings Date", "EarningsDate", "earningsDate"):
        value = calendar.get(key)
        if isinstance(value, (list, tuple)):
            candidates.extend(value)
        elif value is not None:
            candidates.append(value)
    for key in ("earningsTimestamp", "earningsTimestampStart"):
        if info.get(key) is not None:
            candidates.append(info[key])

    now = datetime.now(timezone.utc)
    future_days: list[int] = []
    for value in candidates:
        try:
            if isinstance(value, (int, float)):
                moment = datetime.fromtimestamp(value, tz=timezone.utc)
            else:
                stamp = pd.Timestamp(value)
                moment = stamp.to_pydatetime()
                if moment.tzinfo is None:
                    moment = moment.replace(tzinfo=timezone.utc)
                else:
                    moment = moment.astimezone(timezone.utc)
            days = (moment - now).days
            if days >= 0:
                future_days.append(days)
        except (TypeError, ValueError, OverflowError):
            continue
    return min(future_days) if future_days else None


def _identity(bundle: dict[str, Any]) -> tuple[str, str, str, str, str]:
    info = bundle.get("info", {})
    fast = bundle.get("fast_info", {})
    symbol = str(bundle.get("symbol") or "").upper()
    company_name = str(info.get("longName") or info.get("shortName") or symbol)
    currency = str(info.get("currency") or fast.get("currency") or "USD")
    exchange = str(info.get("fullExchangeName") or info.get("exchange") or "")
    sector = str(info.get("sector") or "")
    return symbol, company_name, currency, exchange, sector


def _market_context(bundle: dict[str, Any]) -> tuple[bool | None, bool | None]:
    return trend_is_positive(bundle.get("benchmark", pd.DataFrame())), trend_is_positive(
        bundle.get("sector_history", pd.DataFrame())
    )


def _confidence(score: float, completeness: int, blockers: int) -> str:
    if completeness < 60 or blockers > 1:
        return "LOW"
    if completeness >= 85 and blockers == 0 and (score >= 78 or score < 48):
        return "HIGH"
    return "MEDIUM"


def analyze_swing(bundle: dict[str, Any], target_pct: float = 5) -> AnalysisResult:
    history = compute_indicators(bundle["history"])
    if history.empty:
        raise ValueError("Price history is empty")
    row = history.iloc[-1]
    symbol, company, currency, exchange, sector = _identity(bundle)
    info = bundle.get("info", {})
    current = safe_float(row.get("Close"))
    if current is None or current <= 0:
        raise ValueError("Current price is unavailable")

    previous = safe_float(history["Close"].iloc[-2]) if len(history) > 1 else None
    change_pct = ((current / previous) - 1) * 100 if previous else None
    ema20 = safe_float(row.get("EMA20"))
    sma50 = safe_float(row.get("SMA50"))
    sma200 = safe_float(row.get("SMA200"))
    rsi = safe_float(row.get("RSI14"))
    macd = safe_float(row.get("MACD"))
    macd_signal = safe_float(row.get("MACD_SIGNAL"))
    atr = safe_float(row.get("ATR14")) or current * 0.02
    volume = safe_float(row.get("Volume")) or 0
    volume_avg = safe_float(row.get("VOLUME_AVG20")) or 0
    dollar_volume = safe_float(row.get("DOLLAR_VOLUME_AVG20")) or 0
    volume_ratio = volume / volume_avg if volume_avg > 0 else None
    support, resistance = technical_levels(history)
    market_up, sector_up = _market_context(bundle)

    positives: list[Factor] = []
    risks: list[Factor] = []
    blockers: list[Factor] = []
    breakdown: dict[str, float] = {}

    trend_score = 0.0
    if ema20 is not None and current > ema20:
        trend_score += 7
        positives.append(_factor("above_ema20"))
    else:
        risks.append(_factor("below_ema20"))
    if sma50 is not None and current > sma50:
        trend_score += 7
        positives.append(_factor("above_sma50"))
    else:
        risks.append(_factor("below_sma50"))
    if sma200 is not None and current > sma200:
        trend_score += 5
        positives.append(_factor("above_sma200"))
    else:
        risks.append(_factor("below_sma200"))
    sma50_lookback = safe_float(history["SMA50"].iloc[-11]) if len(history) > 11 else None
    if sma50 is not None and sma50_lookback is not None and sma50 > sma50_lookback:
        trend_score += 4
        positives.append(_factor("sma50_rising"))
    else:
        risks.append(_factor("sma50_falling"))
    if market_up is True:
        trend_score += 2
        positives.append(_factor("market_positive"))
    elif market_up is False:
        risks.append(_factor("market_weak"))
    if sector_up is True:
        trend_score += 1
    breakdown["trend"] = round(_clamp(trend_score, 0, 25), 1)

    momentum_score = 0.0
    if rsi is not None:
        if 50 <= rsi <= 68:
            momentum_score += 10
            positives.append(_factor("rsi_healthy", value=rsi))
        elif 42 <= rsi < 50 or 68 < rsi <= 73:
            momentum_score += 6
        elif rsi > 73:
            momentum_score += 2
            risks.append(_factor("rsi_hot", value=rsi))
        else:
            risks.append(_factor("rsi_weak", value=rsi))
    if macd is not None and macd_signal is not None and macd > macd_signal:
        momentum_score += 5
        positives.append(_factor("momentum_positive"))
    else:
        risks.append(_factor("momentum_negative"))
    breakdown["momentum"] = round(_clamp(momentum_score, 0, 15), 1)

    volume_score = 0.0
    if dollar_volume >= 100_000_000:
        volume_score += 10
    elif dollar_volume >= 25_000_000:
        volume_score += 8
    elif dollar_volume >= MIN_AVERAGE_DOLLAR_VOLUME:
        volume_score += 5
    else:
        blocker = _factor("liquidity_low", value=_human_number(dollar_volume, currency))
        risks.append(blocker)
        blockers.append(blocker)
    if dollar_volume >= MIN_AVERAGE_DOLLAR_VOLUME:
        positives.append(_factor("liquidity_good", value=_human_number(dollar_volume, currency)))
    if volume_ratio is not None and volume_ratio >= 1.1:
        volume_score += 5
        positives.append(_factor("volume_support", value=volume_ratio))
    elif volume_ratio is not None and volume_ratio >= 0.75:
        volume_score += 3
    elif volume_ratio is not None:
        risks.append(_factor("volume_weak", value=volume_ratio))
    breakdown["volume_liquidity"] = round(_clamp(volume_score, 0, 15), 1)

    distance_ema_pct = ((current / ema20) - 1) * 100 if ema20 else None
    entry_score = 0.0
    if distance_ema_pct is not None and -1.5 <= distance_ema_pct <= 3.0:
        entry_score = 15
        positives.append(_factor("near_ema"))
    elif distance_ema_pct is not None and -3 <= distance_ema_pct <= 6:
        entry_score = 9
    elif distance_ema_pct is not None and distance_ema_pct > 6:
        entry_score = 3
        risks.append(_factor("extended_from_ema", value=distance_ema_pct))
    breakdown["entry_quality"] = round(entry_score, 1)

    resistance_distance = ((resistance / current) - 1) * 100 if resistance and resistance > current else None
    if resistance_distance is None:
        room_score = 10.0
    elif resistance_distance >= target_pct + 1:
        room_score = 15.0
        positives.append(_factor("room_available", value=resistance_distance))
    elif resistance_distance >= target_pct:
        room_score = 12.0
        positives.append(_factor("room_available", value=resistance_distance))
    elif resistance_distance >= target_pct * 0.8:
        room_score = 6.0
        risks.append(_factor("resistance_close", value=resistance_distance))
    else:
        room_score = 1.0
        blocker = _factor("resistance_close", value=resistance_distance)
        risks.append(blocker)
        blockers.append(blocker)
    breakdown["room_to_target"] = room_score

    stop_candidate = current - max(atr * 1.35, current * 0.018)
    if support is not None and current * 0.94 < support < current:
        stop_candidate = min(current - atr * 0.8, support - atr * 0.20)
    stop_loss = max(current * 0.94, stop_candidate)
    risk_per_share = current - stop_loss
    target_1 = current * (1 + target_pct / 100)
    target_2 = current * (1 + min(target_pct + 2, target_pct * 1.5) / 100)
    rr = (target_1 - current) / risk_per_share if risk_per_share > 0 else None
    if rr is not None and rr >= 2:
        rr_score = 15.0
        positives.append(_factor("rr_good", value=rr))
    elif rr is not None and rr >= 1.5:
        rr_score = 11.0
        positives.append(_factor("rr_good", value=rr))
    elif rr is not None and rr >= 1.35:
        rr_score = 7.0
        risks.append(_factor("rr_weak", value=rr))
    else:
        rr_score = 2.0
        blocker = _factor("rr_weak", value=rr or 0)
        risks.append(blocker)
        blockers.append(blocker)
    breakdown["risk_reward"] = rr_score

    if len(history) < MIN_HISTORY_ROWS:
        blocker = _factor("insufficient_history")
        risks.append(blocker)
        blockers.append(blocker)
    earnings_days = _days_to_earnings(bundle.get("calendar", {}), info)
    if earnings_days is not None and earnings_days <= 5:
        blocker = _factor("earnings_soon", value=earnings_days)
        risks.append(blocker)
        blockers.append(blocker)

    score = int(round(_clamp(sum(breakdown.values()))))
    if blockers:
        decision = "AVOID"
    elif score >= 78:
        decision = "ENTER"
    elif score >= 60:
        decision = "WAIT"
    else:
        decision = "AVOID"

    coverage_items = [ema20, sma50, sma200, rsi, macd, volume_ratio, dollar_volume, market_up, resistance, rr]
    completeness = int(round(100 * sum(item is not None for item in coverage_items) / len(coverage_items)))
    entry_low = max(0.01, current - atr * 0.35)
    entry_high = current + atr * 0.10

    return AnalysisResult(
        symbol=symbol,
        company_name=company,
        mode="swing",
        decision=decision,
        score=score,
        confidence=_confidence(score, completeness, len(blockers)),
        current_price=current,
        currency=currency,
        price_change_pct=change_pct,
        entry_low=entry_low,
        entry_high=entry_high,
        stop_loss=stop_loss,
        target_1=target_1,
        target_2=target_2,
        support=support,
        resistance=resistance,
        risk_reward=rr,
        positives=_dedupe(positives),
        risks=_dedupe(risks),
        blockers=_dedupe(blockers),
        breakdown=breakdown,
        indicators={
            "rsi": rsi,
            "volume_ratio": volume_ratio,
            "ema20": ema20,
            "sma50": sma50,
            "sma200": sma200,
            "market_trend": "UP" if market_up else "DOWN" if market_up is False else None,
            "sector_trend": "UP" if sector_up else "DOWN" if sector_up is False else None,
        },
        fundamentals={},
        data_completeness=completeness,
        as_of=history.index[-1].strftime("%Y-%m-%d"),
        exchange=exchange,
        sector=sector,
        target_pct=target_pct,
    )


def analyze_investment(bundle: dict[str, Any]) -> AnalysisResult:
    history = compute_indicators(bundle["history"])
    if history.empty:
        raise ValueError("Price history is empty")
    row = history.iloc[-1]
    symbol, company, currency, exchange, sector = _identity(bundle)
    info = bundle.get("info", {})
    current = safe_float(row.get("Close"))
    if current is None or current <= 0:
        raise ValueError("Current price is unavailable")
    previous = safe_float(history["Close"].iloc[-2]) if len(history) > 1 else None
    change_pct = ((current / previous) - 1) * 100 if previous else None
    support, resistance = technical_levels(history)

    revenue_growth = _pct(info.get("revenueGrowth"))
    earnings_growth = _pct(info.get("earningsGrowth"))
    profit_margin = _pct(info.get("profitMargins"))
    roe = _pct(info.get("returnOnEquity"))
    free_cash_flow = safe_float(info.get("freeCashflow"))
    debt_equity = safe_float(info.get("debtToEquity"))
    current_ratio = safe_float(info.get("currentRatio"))
    trailing_pe = safe_float(info.get("trailingPE"))
    forward_pe = safe_float(info.get("forwardPE"))
    peg = safe_float(info.get("pegRatio"))
    market_cap = safe_float(info.get("marketCap"))
    target_mean = safe_float(info.get("targetMeanPrice"))
    sma50 = safe_float(row.get("SMA50"))
    sma200 = safe_float(row.get("SMA200"))

    positives: list[Factor] = []
    risks: list[Factor] = []
    blockers: list[Factor] = []
    breakdown: dict[str, float] = {}

    growth_score = 0.0
    if revenue_growth is not None:
        if revenue_growth >= 15:
            growth_score += 10
            positives.append(_factor("revenue_growth_good", value=revenue_growth))
        elif revenue_growth >= 5:
            growth_score += 7
            positives.append(_factor("revenue_growth_good", value=revenue_growth))
        elif revenue_growth >= 0:
            growth_score += 4
        else:
            risks.append(_factor("revenue_growth_weak", value=revenue_growth))
    if earnings_growth is not None:
        if earnings_growth >= 15:
            growth_score += 10
            positives.append(_factor("earnings_growth_good", value=earnings_growth))
        elif earnings_growth >= 5:
            growth_score += 7
            positives.append(_factor("earnings_growth_good", value=earnings_growth))
        elif earnings_growth >= 0:
            growth_score += 4
        else:
            risks.append(_factor("earnings_growth_weak", value=earnings_growth))
    breakdown["growth"] = round(_clamp(growth_score, 0, 20), 1)

    profitability_score = 0.0
    if profit_margin is not None:
        if profit_margin >= 15:
            profitability_score += 8
            positives.append(_factor("margins_good", value=profit_margin))
        elif profit_margin >= 5:
            profitability_score += 5
        else:
            risks.append(_factor("margins_weak", value=profit_margin))
    if free_cash_flow is not None:
        if free_cash_flow > 0:
            profitability_score += 6
            positives.append(_factor("fcf_positive", value=_human_number(free_cash_flow, currency)))
        else:
            risks.append(_factor("fcf_negative", value=_human_number(free_cash_flow, currency)))
    if roe is not None:
        if roe >= 15:
            profitability_score += 6
            positives.append(_factor("roe_good", value=roe))
        elif roe >= 8:
            profitability_score += 3
    breakdown["profitability"] = round(_clamp(profitability_score, 0, 20), 1)

    health_score = 0.0
    if debt_equity is not None:
        if debt_equity <= 80:
            health_score += 9
            positives.append(_factor("debt_manageable", value=debt_equity))
        elif debt_equity <= 150:
            health_score += 5
        else:
            risks.append(_factor("debt_high", value=debt_equity))
    if current_ratio is not None:
        if current_ratio >= 1.5:
            health_score += 6
            positives.append(_factor("current_ratio_good", value=current_ratio))
        elif current_ratio >= 1:
            health_score += 4
    breakdown["financial_health"] = round(_clamp(health_score, 0, 15), 1)

    valuation_score = 0.0
    if trailing_pe is not None and trailing_pe > 0:
        if trailing_pe <= 22:
            valuation_score += 10
            positives.append(_factor("valuation_reasonable", value=trailing_pe))
        elif trailing_pe <= 35:
            valuation_score += 6
        else:
            valuation_score += 2
            risks.append(_factor("valuation_high", value=trailing_pe))
    if forward_pe is not None and forward_pe > 0:
        if trailing_pe and forward_pe < trailing_pe:
            valuation_score += 6
            positives.append(_factor("forward_valuation_better", value=forward_pe))
        elif forward_pe <= 30:
            valuation_score += 4
    if peg is not None and 0 < peg <= 2:
        valuation_score += 4
    breakdown["valuation"] = round(_clamp(valuation_score, 0, 20), 1)

    quality_score = 0.0
    if market_cap is not None:
        if market_cap >= 50_000_000_000:
            quality_score += 8
            positives.append(_factor("large_established", value=_human_number(market_cap, currency)))
        elif market_cap >= 10_000_000_000:
            quality_score += 6
        elif market_cap >= 2_000_000_000:
            quality_score += 3
    if sector:
        quality_score += 3
    if profit_margin is not None and profit_margin > 0:
        quality_score += 2
    if revenue_growth is not None and revenue_growth > 0:
        quality_score += 2
    breakdown["quality"] = round(_clamp(quality_score, 0, 15), 1)

    technical_score = 0.0
    if sma200 is not None and current > sma200:
        technical_score += 6
        positives.append(_factor("above_sma200"))
    else:
        risks.append(_factor("below_sma200"))
    if sma50 is not None and current > sma50:
        technical_score += 4
        positives.append(_factor("above_sma50"))
    else:
        risks.append(_factor("below_sma50"))
    breakdown["technical_timing"] = technical_score

    fundamental_values = [
        revenue_growth,
        earnings_growth,
        profit_margin,
        roe,
        free_cash_flow,
        debt_equity,
        current_ratio,
        trailing_pe,
        forward_pe,
        market_cap,
    ]
    available = sum(value is not None for value in fundamental_values)
    completeness = int(round(available / len(fundamental_values) * 100))
    if available < 5:
        blocker = _factor("fundamentals_limited")
        risks.append(blocker)
        blockers.append(blocker)
    if len(history) < MIN_HISTORY_ROWS:
        blocker = _factor("insufficient_history")
        risks.append(blocker)
        blockers.append(blocker)
    if target_mean is None:
        risks.append(_factor("no_reference_target"))

    score = int(round(_clamp(sum(breakdown.values()))))
    if blockers:
        decision = "WAIT"
    elif score >= 80:
        decision = "ENTER"
    elif score >= 65:
        decision = "GRADUAL"
    elif score >= 50:
        decision = "WAIT"
    else:
        decision = "AVOID"

    entry_low = min(current, sma50) if sma50 and current * 0.92 <= sma50 <= current * 1.03 else current * 0.97
    entry_high = current * 1.01
    stop_loss = max(current * 0.90, support * 0.985) if support else current * 0.90
    target_1 = target_mean if target_mean and target_mean > current else None
    target_2 = safe_float(info.get("fiftyTwoWeekHigh"))
    if target_2 is not None and target_2 <= current:
        target_2 = None
    rr = ((target_1 - current) / (current - stop_loss)) if target_1 and current > stop_loss else None

    fundamentals = {
        "market_cap": market_cap,
        "trailing_pe": trailing_pe,
        "forward_pe": forward_pe,
        "revenue_growth": revenue_growth,
        "earnings_growth": earnings_growth,
        "profit_margin": profit_margin,
        "debt_equity": debt_equity,
        "roe": roe,
        "free_cash_flow": free_cash_flow,
        "current_ratio": current_ratio,
    }
    return AnalysisResult(
        symbol=symbol,
        company_name=company,
        mode="investment",
        decision=decision,
        score=score,
        confidence=_confidence(score, completeness, len(blockers)),
        current_price=current,
        currency=currency,
        price_change_pct=change_pct,
        entry_low=entry_low,
        entry_high=entry_high,
        stop_loss=stop_loss,
        target_1=target_1,
        target_2=target_2,
        support=support,
        resistance=resistance,
        risk_reward=rr,
        positives=_dedupe(positives),
        risks=_dedupe(risks),
        blockers=_dedupe(blockers),
        breakdown=breakdown,
        indicators={
            "rsi": safe_float(row.get("RSI14")),
            "volume_ratio": (
                safe_float(row.get("Volume")) / safe_float(row.get("VOLUME_AVG20"))
                if safe_float(row.get("VOLUME_AVG20"))
                else None
            ),
            "ema20": safe_float(row.get("EMA20")),
            "sma50": sma50,
            "sma200": sma200,
            "market_trend": None,
        },
        fundamentals=fundamentals,
        data_completeness=completeness,
        as_of=history.index[-1].strftime("%Y-%m-%d"),
        exchange=exchange,
        sector=sector,
    )


def _dedupe(factors: list[Factor]) -> list[Factor]:
    seen: set[str] = set()
    result: list[Factor] = []
    for item in factors:
        if item.key not in seen:
            seen.add(item.key)
            result.append(item)
    return result
