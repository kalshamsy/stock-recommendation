"""Shared configuration for the first professional release."""

APP_NAME = "Stock Decision Pro"
APP_VERSION = "1.0.0"

SWING_SCORE_WEIGHTS = {
    "trend": 25,
    "momentum": 15,
    "volume_liquidity": 15,
    "entry_quality": 15,
    "room_to_target": 15,
    "risk_reward": 15,
}

INVESTMENT_SCORE_WEIGHTS = {
    "growth": 20,
    "profitability": 20,
    "financial_health": 15,
    "valuation": 20,
    "quality": 15,
    "technical_timing": 10,
}

SECTOR_ETFS = {
    "Basic Materials": "XLB",
    "Communication Services": "XLC",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Industrials": "XLI",
    "Real Estate": "XLRE",
    "Technology": "XLK",
    "Utilities": "XLU",
}

DEFAULT_TARGET_PCT = 5
MIN_HISTORY_ROWS = 210
MIN_AVERAGE_DOLLAR_VOLUME = 5_000_000
DATA_CACHE_TTL_SECONDS = 900
