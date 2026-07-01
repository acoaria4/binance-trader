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
TRAIL_ACTIVATE_PCT   = float(os.getenv("TRAIL_ACTIVATE_PCT", 1.0))
TRAIL_DISTANCE_PCT   = float(os.getenv("TRAIL_DISTANCE_PCT", STOP_LOSS_PCT))
FORWARD_CANDLES        = int(os.getenv("FORWARD_CANDLES", 10))

# ML
LOOKBACK_CANDLES         = int(os.getenv("LOOKBACK_CANDLES", 200))
RETRAIN_EVERY_N          = int(os.getenv("RETRAIN_EVERY_N", 50))
MIN_SIGNAL_CONFIDENCE    = float(os.getenv("MIN_SIGNAL_CONFIDENCE", 0.62))  # Balanced: high enough to filter noise, low enough to trade

# Regime filter
ADX_THRESHOLD            = float(os.getenv("ADX_THRESHOLD", 20.0))
MIN_ATR_RATIO            = float(os.getenv("MIN_ATR_RATIO", 0.8))
REQUIRE_TREND            = os.getenv("REQUIRE_TREND", "true").lower() == "true"

# ML retrain gate — walk-forward trading simulation on validation fold
RETRAIN_MIN_F1             = float(os.getenv("RETRAIN_MIN_F1", 0.25))
RETRAIN_MIN_VAL_RETURN_PCT = float(os.getenv("RETRAIN_MIN_VAL_RETURN_PCT", 0.0))
RETRAIN_MIN_VAL_SHARPE     = float(os.getenv("RETRAIN_MIN_VAL_SHARPE", 0.0))
RETRAIN_MIN_VAL_TRADES     = int(os.getenv("RETRAIN_MIN_VAL_TRADES", 3))

# Execution costs (backtest + EV gate)
FEE_PCT                  = float(os.getenv("FEE_PCT", 0.10))
SLIPPAGE_PCT             = float(os.getenv("SLIPPAGE_PCT", 0.05))

# High-conviction entry gates
EV_MIN                   = float(os.getenv("EV_MIN", 0.005))
P_WIN_MIN                  = float(os.getenv("P_WIN_MIN", 0.55))
CONVICTION_MIN           = float(os.getenv("CONVICTION_MIN", 0.65))

# Multi-timeframe confirmation (4h resampled from base TF)
REQUIRE_MTF              = os.getenv("REQUIRE_MTF", "true").lower() == "true"
MTF_TIMEFRAME            = os.getenv("MTF_TIMEFRAME", "4h")
MTF_MIN_EMA_ALIGN        = int(os.getenv("MTF_MIN_EMA_ALIGN", 1))
MTF_MIN_PRICE_VS_EMA200  = float(os.getenv("MTF_MIN_PRICE_VS_EMA200", -0.02))
MTF_MIN_ADX              = float(os.getenv("MTF_MIN_ADX", 15.0))

# Position persistence
POSITIONS_FILE           = os.getenv("POSITIONS_FILE", "logs/positions.json")
SIGNAL_LOG_FILE          = os.getenv("SIGNAL_LOG_FILE", "logs/signals.csv")

# Logging
LOG_LEVEL       = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE        = os.getenv("LOG_FILE", "logs/bot.log")
TRADE_LOG_FILE  = os.getenv("TRADE_LOG_FILE", "logs/trades.csv")
