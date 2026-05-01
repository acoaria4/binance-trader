"""
risk.py
Risk management: position sizing, stop-loss / take-profit calculation,
and open position tracking.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from config import settings
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class Position:
    symbol: str
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    side: str = "long"
    order_id: Optional[str] = None
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def current_pnl_pct(self, current_price: float = 0.0) -> float:
        if current_price == 0:
            return 0.0
        return (current_price - self.entry_price) / self.entry_price * 100


class RiskManager:
    def __init__(self):
        self.open_positions: list[Position] = []

    # ── Position sizing ───────────────────────────────────────────────────────
    def calculate_position(self, entry_price: float,
                           available_usdt: float) -> dict:
        """
        Returns a dict with:
          qty         - base asset quantity to buy
          stop_loss   - price to exit on loss
          take_profit - price to exit on gain
          risk_usdt   - USDT amount at risk
        """
        trade_usdt  = min(settings.TRADE_AMOUNT_USDT, available_usdt * 0.95)
        qty         = trade_usdt / entry_price
        stop_loss   = entry_price * (1 - settings.STOP_LOSS_PCT   / 100)
        take_profit = entry_price * (1 + settings.TAKE_PROFIT_PCT / 100)
        risk_usdt   = (entry_price - stop_loss) * qty

        log.info(
            f"Position sizing: qty={qty:.6f} | "
            f"SL={stop_loss:.2f} | TP={take_profit:.2f} | "
            f"Risk={risk_usdt:.2f} USDT"
        )
        return dict(qty=qty, stop_loss=stop_loss,
                    take_profit=take_profit, risk_usdt=risk_usdt)

    # ── Guard checks ──────────────────────────────────────────────────────────
    def can_open_trade(self, symbol: str) -> tuple[bool, str]:
        """Returns (allowed, reason)."""
        if len(self.open_positions) >= settings.MAX_OPEN_TRADES:
            return False, f"Max open trades ({settings.MAX_OPEN_TRADES}) reached"
        if any(p.symbol == symbol for p in self.open_positions):
            return False, f"Already have open position in {symbol}"
        return True, "OK"

    # ── Position lifecycle ────────────────────────────────────────────────────
    def open_position(self, symbol: str, entry_price: float,
                      quantity: float, order_id: str = None) -> Position:
        sl = entry_price * (1 - settings.STOP_LOSS_PCT   / 100)
        tp = entry_price * (1 + settings.TAKE_PROFIT_PCT / 100)
        pos = Position(
            symbol=symbol,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=sl,
            take_profit=tp,
            order_id=order_id,
        )
        self.open_positions.append(pos)
        log.info(f"Position opened: {symbol} @ {entry_price:.2f} | "
                 f"SL={sl:.2f} | TP={tp:.2f}")
        return pos

    def close_position(self, symbol: str) -> Optional[Position]:
        for i, pos in enumerate(self.open_positions):
            if pos.symbol == symbol:
                self.open_positions.pop(i)
                log.info(f"Position closed: {symbol}")
                return pos
        return None

    # ── Exit checks ───────────────────────────────────────────────────────────
    def check_exits(self, current_price: float) -> list[tuple[Position, str]]:
        """
        Checks all open positions against current price.
        Returns list of (position, reason) that should be closed.
        reason: 'stop_loss' | 'take_profit'
        """
        exits = []
        for pos in self.open_positions:
            if current_price <= pos.stop_loss:
                exits.append((pos, "stop_loss"))
            elif current_price >= pos.take_profit:
                exits.append((pos, "take_profit"))
        return exits

    def summary(self) -> str:
        if not self.open_positions:
            return "No open positions."
        lines = []
        for p in self.open_positions:
            lines.append(
                f"  {p.symbol} | entry={p.entry_price:.2f} | "
                f"SL={p.stop_loss:.2f} | TP={p.take_profit:.2f} | "
                f"qty={p.quantity:.6f}"
            )
        return "\n".join(lines)
