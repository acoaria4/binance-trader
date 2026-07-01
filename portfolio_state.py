"""
portfolio_state.py
Portfolio-level risk tracking: daily loss limits, consecutive losses, halts.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta

from config import settings
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class PortfolioState:
    utc_date: str = ""
    daily_pnl_pct: float = 0.0
    consecutive_losses: int = 0
    peak_equity: float = 0.0
    entries_halted_until: str = ""

    @classmethod
    def load(cls, path: str = None) -> "PortfolioState":
        path = path or settings.PORTFOLIO_STATE_FILE
        if not os.path.exists(path):
            return cls()
        try:
            with open(path) as f:
                data = json.load(f)
            return cls(**{k: data[k] for k in asdict(cls()) if k in data})
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            log.warning(f"Portfolio state load failed: {e}")
            return cls()

    def save(self, path: str = None) -> None:
        path = path or settings.PORTFOLIO_STATE_FILE
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    def _roll_day(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.utc_date != today:
            self.utc_date = today
            self.daily_pnl_pct = 0.0

    def update_peak(self, equity: float) -> None:
        if equity > self.peak_equity:
            self.peak_equity = equity

    def record_closed_trade(self, pnl_pct: float, equity: float) -> None:
        self._roll_day()
        self.daily_pnl_pct += pnl_pct
        self.update_peak(equity)

        if pnl_pct < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        if self.daily_pnl_pct <= -settings.DAILY_LOSS_LIMIT_PCT:
            until = datetime.now(timezone.utc) + timedelta(hours=settings.ENTRY_HALT_HOURS)
            self.entries_halted_until = until.isoformat()
            log.warning(
                f"Daily loss limit hit ({self.daily_pnl_pct:.2f}%). "
                f"Entries halted until {self.entries_halted_until}"
            )

    def size_multiplier(self) -> float:
        if self.consecutive_losses >= settings.CONSECUTIVE_LOSS_LIMIT:
            return settings.CONSECUTIVE_LOSS_SIZE_FACTOR
        return 1.0

    def can_enter_new(self) -> tuple[bool, str]:
        self._roll_day()

        if self.entries_halted_until:
            try:
                until = datetime.fromisoformat(self.entries_halted_until)
                if until.tzinfo is None:
                    until = until.replace(tzinfo=timezone.utc)
                if datetime.now(timezone.utc) < until:
                    return False, "daily_loss_halt"
                self.entries_halted_until = ""
            except ValueError:
                self.entries_halted_until = ""

        if self.daily_pnl_pct <= -settings.DAILY_LOSS_LIMIT_PCT:
            return False, "daily_loss_limit"

        return True, "OK"

    def drawdown_pct(self, equity: float) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - equity) / self.peak_equity * 100
