"""
risk.py
Risk management: position sizing, trailing stop-loss, take-profit,
and open position tracking.

v2 changes:
  - Trailing stop-loss: stop moves up as price rises, locking in profit
  - SELL signal exit: positions can be closed on ML SELL signal
  - Improved position summary showing live PnL and trailing SL
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from config import settings
from utils.logger import get_logger

log = get_logger(__name__)

# Trailing stop activates once trade is profitable by this %
TRAIL_ACTIVATE_PCT  = float(1.0)
# Once activated, stop trails this % below the highest price seen
TRAIL_DISTANCE_PCT  = float(getattr(settings, 'STOP_LOSS_PCT', 1.5))


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

    # Trailing stop fields
    highest_price: float = field(init=False)
    trailing_active: bool = False

    def __post_init__(self):
        self.highest_price = self.entry_price

    def update_trailing_stop(self, current_price: float) -> None:
        """
        Update trailing stop based on current price.
        Activates once price is TRAIL_ACTIVATE_PCT% above entry.
        Then trails TRAIL_DISTANCE_PCT% below the highest seen price.
        """
        # Update highest price seen
        if current_price > self.highest_price:
            self.highest_price = current_price

        # Check if we should activate trailing stop
        gain_pct = (self.highest_price - self.entry_price) / self.entry_price * 100
        if gain_pct >= TRAIL_ACTIVATE_PCT:
            self.trailing_active = True

        # If trailing active, move stop up
        if self.trailing_active:
            new_sl = self.highest_price * (1 - TRAIL_DISTANCE_PCT / 100)
            # Only move stop UP, never down
            if new_sl > self.stop_loss:
                old_sl = self.stop_loss
                self.stop_loss = new_sl
                log.info(f"Trailing SL updated: {old_sl:.2f} → {new_sl:.2f} "
                         f"(highest={self.highest_price:.2f})")

    def pnl_pct(self, current_price: float) -> float:
        return (current_price - self.entry_price) / self.entry_price * 100


class RiskManager:
    def __init__(self):
        self.open_positions: list[Position] = []

    # ── Position sizing ───────────────────────────────────────────────────────
    def calculate_position(self, entry_price: float,
                           available_usdt: float) -> dict:
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
                 f"SL={sl:.2f} | TP={tp:.2f} | Trail activates at "
                 f"{entry_price * (1 + TRAIL_ACTIVATE_PCT/100):.2f}")
        return pos

    def close_position(self, symbol: str) -> Optional[Position]:
        for i, pos in enumerate(self.open_positions):
            if pos.symbol == symbol:
                self.open_positions.pop(i)
                log.info(f"Position closed: {symbol}")
                return pos
        return None

    # ── Exit checks with trailing stop ────────────────────────────────────────
    def check_exits(self, current_price: float) -> list[tuple[Position, str]]:
        """
        Updates trailing stops and checks all exit conditions.
        Returns list of (position, reason) to close.
        reason: 'stop_loss' | 'trailing_stop' | 'take_profit'
        """
        exits = []
        for pos in self.open_positions:
            # Update trailing stop first
            was_trailing = pos.trailing_active
            pos.update_trailing_stop(current_price)

            if current_price >= pos.take_profit:
                exits.append((pos, "take_profit"))
            elif current_price <= pos.stop_loss:
                reason = "trailing_stop" if pos.trailing_active else "stop_loss"
                exits.append((pos, reason))

        return exits

    def summary(self) -> str:
        if not self.open_positions:
            return "No open positions."
        lines = []
        for p in self.open_positions:
            trail_str = f"TRAIL active (high={p.highest_price:.2f})" \
                        if p.trailing_active else "trail pending"
            lines.append(
                f"  {p.symbol} | entry={p.entry_price:.2f} | "
                f"SL={p.stop_loss:.2f} | TP={p.take_profit:.2f} | "
                f"qty={p.quantity:.6f} | {trail_str}"
            )
        return "\n".join(lines)
