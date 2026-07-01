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

# Phase 2 — risk-based sizing & portfolio controls
USE_RISK_SIZING           = os.getenv("USE_RISK_SIZING", "true").lower() == "true"
RISK_PCT_PER_TRADE        = float(os.getenv("RISK_PCT_PER_TRADE", 0.5))
MAX_POSITION_USDT         = float(os.getenv("MAX_POSITION_USDT", 200))
MAX_PORTFOLIO_HEAT_PCT    = float(os.getenv("MAX_PORTFOLIO_HEAT_PCT", 2.0))
DAILY_LOSS_LIMIT_PCT      = float(os.getenv("DAILY_LOSS_LIMIT_PCT", 3.0))
ENTRY_HALT_HOURS          = int(os.getenv("ENTRY_HALT_HOURS", 24))
CONSECUTIVE_LOSS_LIMIT    = int(os.getenv("CONSECUTIVE_LOSS_LIMIT", 3))
CONSECUTIVE_LOSS_SIZE_FACTOR = float(os.getenv("CONSECUTIVE_LOSS_SIZE_FACTOR", 0.5))
PORTFOLIO_STATE_FILE      = os.getenv("PORTFOLIO_STATE_FILE", "logs/portfolio_state.json")

# ATR-based dynamic barriers
USE_ATR_STOPS             = os.getenv("USE_ATR_STOPS", "true").lower() == "true"
ATR_SL_MULT               = float(os.getenv("ATR_SL_MULT", 1.5))
ATR_TP_MULT               = float(os.getenv("ATR_TP_MULT", 3.0))
ATR_TRAIL_MULT            = float(os.getenv("ATR_TRAIL_MULT", 1.5))
MIN_STOP_LOSS_PCT         = float(os.getenv("MIN_STOP_LOSS_PCT", 0.8))
MAX_STOP_LOSS_PCT         = float(os.getenv("MAX_STOP_LOSS_PCT", 4.0))
MIN_TAKE_PROFIT_PCT       = float(os.getenv("MIN_TAKE_PROFIT_PCT", 2.0))
MAX_TAKE_PROFIT_PCT       = float(os.getenv("MAX_TAKE_PROFIT_PCT", 12.0))
MIN_TRAIL_DISTANCE_PCT    = float(os.getenv("MIN_TRAIL_DISTANCE_PCT", 0.5))

# Phase 3 — execution fidelity
USE_LIMIT_ORDERS            = os.getenv("USE_LIMIT_ORDERS", "true").lower() == "true"
MAX_SPREAD_PCT              = float(os.getenv("MAX_SPREAD_PCT", 0.15))
LIMIT_ORDER_OFFSET_PCT      = float(os.getenv("LIMIT_ORDER_OFFSET_PCT", 0.02))
LIMIT_ORDER_TIMEOUT_SEC     = int(os.getenv("LIMIT_ORDER_TIMEOUT_SEC", 30))
LIMIT_FALLBACK_TO_MARKET    = os.getenv("LIMIT_FALLBACK_TO_MARKET", "true").lower() == "true"
USE_LIMIT_EXITS             = os.getenv("USE_LIMIT_EXITS", "false").lower() == "true"
RECONCILE_ON_START          = os.getenv("RECONCILE_ON_START", "true").lower() == "true"
RECONCILE_EVERY_N_LOOPS     = int(os.getenv("RECONCILE_EVERY_N_LOOPS", 10))
CANCEL_STALE_ORDERS_SEC     = int(os.getenv("CANCEL_STALE_ORDERS_SEC", 300))
PENDING_ORDERS_FILE         = os.getenv("PENDING_ORDERS_FILE", "logs/pending_orders.json")

# Position persistence
POSITIONS_FILE           = os.getenv("POSITIONS_FILE", "logs/positions.json")
SIGNAL_LOG_FILE          = os.getenv("SIGNAL_LOG_FILE", "logs/signals.csv")

# Logging
LOG_LEVEL       = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE        = os.getenv("LOG_FILE", "logs/bot.log")
TRADE_LOG_FILE  = os.getenv("TRADE_LOG_FILE", "logs/trades.csv")
