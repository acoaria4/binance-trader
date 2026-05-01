"""
utils/trade_logger.py
Logs every trade action to a CSV file for post-analysis.
"""
import csv
import os
from datetime import datetime, timezone

from config import settings
from utils.logger import get_logger

log = get_logger(__name__)

HEADERS = [
    "timestamp", "symbol", "action", "price",
    "quantity", "usdt_value", "signal_confidence",
    "reason", "pnl_pct",
]


class TradeLogger:
    def __init__(self, filepath: str = settings.TRADE_LOG_FILE):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if not os.path.exists(filepath):
            with open(filepath, "w", newline="") as f:
                csv.writer(f).writerow(HEADERS)

    def log_trade(self,
                  symbol: str,
                  action: str,          # BUY | SELL_SL | SELL_TP | SELL_SIGNAL
                  price: float,
                  quantity: float,
                  signal_confidence: float = 0.0,
                  reason: str = "",
                  pnl_pct: float = 0.0) -> None:
        row = [
            datetime.now(timezone.utc).isoformat(),
            symbol,
            action,
            round(price, 4),
            round(quantity, 8),
            round(price * quantity, 4),
            round(signal_confidence, 4),
            reason,
            round(pnl_pct, 4),
        ]
        with open(self.filepath, "a", newline="") as f:
            csv.writer(f).writerow(row)
        log.info(f"Trade logged → {action} {symbol} @ {price:.2f} | PnL: {pnl_pct:.2f}%")
