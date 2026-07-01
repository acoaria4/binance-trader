"""
utils/signal_logger.py
Audit log for every signal evaluation (trades and skips).
"""
import csv
import os
from datetime import datetime, timezone

from config import settings
from utils.logger import get_logger

log = get_logger(__name__)

HEADERS = [
    "timestamp", "symbol", "signal", "p_sell", "p_hold", "p_buy",
    "p_win", "ev", "conviction", "regime_ok", "mtf_ok",
    "action", "block_reason",
]


class SignalLogger:
    def __init__(self, filepath: str = None):
        self.filepath = filepath or settings.SIGNAL_LOG_FILE
        os.makedirs(os.path.dirname(self.filepath) or ".", exist_ok=True)
        if not os.path.exists(self.filepath):
            with open(self.filepath, "w", newline="") as f:
                csv.writer(f).writerow(HEADERS)

    def log(self, symbol: str, evaluation, action: str) -> None:
        row = [
            datetime.now(timezone.utc).isoformat(),
            symbol,
            evaluation.signal,
            round(evaluation.p_sell, 4),
            round(evaluation.p_hold, 4),
            round(evaluation.p_buy, 4),
            round(evaluation.p_win, 4),
            round(evaluation.ev, 6),
            round(evaluation.conviction, 4),
            evaluation.regime_ok,
            evaluation.mtf_ok,
            action,
            evaluation.block_reason,
        ]
        with open(self.filepath, "a", newline="") as f:
            csv.writer(f).writerow(row)
        log.debug(f"Signal logged → {action} | {evaluation.signal} | {evaluation.block_reason}")
