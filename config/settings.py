"""
config/settings.py
Centralised settings loaded from .env
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Exchange
API_KEY    = os.getenv("BINANCE_API_KEY", "")
SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "")

# Trading
SYMBOL               = os.getenv("SYMBOL", "BTC/USDT")
TIMEFRAME            = os.getenv("TIMEFRAME", "1h")
TRADE_AMOUNT_USDT    = float(os.getenv("TRADE_AMOUNT_USDT", 50))
MAX_OPEN_TRADES      = int(os.getenv("MAX_OPEN_TRADES", 3))
STOP_LOSS_PCT        = float(os.getenv("STOP_LOSS_PCT", 1.5))    # Tightened: 2.0 -> 1.5
TAKE_PROFIT_PCT      = float(os.getenv("TAKE_PROFIT_PCT", 5.0))  # Widened:   4.0 -> 5.0

# ML
LOOKBACK_CANDLES         = int(os.getenv("LOOKBACK_CANDLES", 200))
RETRAIN_EVERY_N          = int(os.getenv("RETRAIN_EVERY_N", 50))
MIN_SIGNAL_CONFIDENCE    = float(os.getenv("MIN_SIGNAL_CONFIDENCE", 0.62))  # Balanced: high enough to filter noise, low enough to trade

# Regime filter
ADX_THRESHOLD            = float(os.getenv("ADX_THRESHOLD", 20.0))  # Min ADX to allow trades
REQUIRE_TREND            = os.getenv("REQUIRE_TREND", "true").lower() == "true"

# Logging
LOG_LEVEL       = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE        = os.getenv("LOG_FILE", "logs/bot.log")
TRADE_LOG_FILE  = os.getenv("TRADE_LOG_FILE", "logs/trades.csv")
