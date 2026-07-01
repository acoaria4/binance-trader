"""
risk.py
Risk management: position sizing, trailing stop-loss, take-profit,
open position tracking, and JSON persistence across restarts.
"""
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

import ccxt

from config import settings
from simulation import LongTradeState, check_bar_exit
from utils.logger import get_logger

log = get_logger(__name__)

BALANCE_TOLERANCE = 0.90


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
    highest_price: float = field(init=False)
    trailing_active: bool = False

    def __post_init__(self):
        self.highest_price = self.entry_price

    def to_state(self) -> LongTradeState:
        return LongTradeState(
            entry_price=self.entry_price,
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            highest_price=self.highest_price,
            trailing_active=self.trailing_active,
        )

    def apply_state(self, state: LongTradeState) -> None:
        self.stop_loss = state.stop_loss
        self.highest_price = state.highest_price
        self.trailing_active = state.trailing_active

    def to_dict(self) -> dict:
        data = asdict(self)
        data["opened_at"] = self.opened_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Position":
        d = dict(data)
        opened = d.pop("opened_at", None)
        highest = d.pop("highest_price", None)
        trailing = d.pop("trailing_active", False)
        if isinstance(opened, str):
            opened = datetime.fromisoformat(opened)
        pos = cls(
            symbol=d["symbol"],
            entry_price=float(d["entry_price"]),
            quantity=float(d["quantity"]),
            stop_loss=float(d["stop_loss"]),
            take_profit=float(d["take_profit"]),
            side=d.get("side", "long"),
            order_id=d.get("order_id"),
            opened_at=opened or datetime.now(timezone.utc),
        )
        pos.trailing_active = bool(trailing)
        if highest is not None:
            pos.highest_price = float(highest)
        return pos

    def pnl_pct(self, current_price: float) -> float:
        return (current_price - self.entry_price) / self.entry_price * 100


class RiskManager:
    def __init__(self, persist_path: str = None):
        self.persist_path = persist_path or settings.POSITIONS_FILE
        self.open_positions: list[Position] = []
        self.load()

    def _persist(self) -> None:
        os.makedirs(os.path.dirname(self.persist_path) or ".", exist_ok=True)
        with open(self.persist_path, "w") as f:
            json.dump([p.to_dict() for p in self.open_positions], f, indent=2)

    def load(self) -> None:
        if not os.path.exists(self.persist_path):
            return
        try:
            with open(self.persist_path) as f:
                raw = json.load(f)
            self.open_positions = [Position.from_dict(p) for p in raw]
            log.info(f"Restored {len(self.open_positions)} open position(s) from disk")
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            log.warning(f"Could not load positions file: {e}")

    def reconcile(self, exchange: ccxt.binance, symbol: str = None) -> None:
        symbol = symbol or settings.SYMBOL
        base_asset = symbol.split("/")[0]

        try:
            balance = exchange.fetch_balance()
            free_base = float(balance["free"].get(base_asset, 0.0))
        except Exception as e:
            log.warning(f"Balance reconcile skipped: {e}")
            return

        kept = []
        for pos in self.open_positions:
            if pos.symbol != symbol:
                kept.append(pos)
                continue
            if free_base >= pos.quantity * BALANCE_TOLERANCE:
                kept.append(pos)
            else:
                log.warning(
                    f"Dropping stale position {pos.symbol}: "
                    f"tracked qty={pos.quantity:.6f}, exchange free={free_base:.6f}"
                )

        self.open_positions = kept

        tracked_qty = sum(p.quantity for p in self.open_positions if p.symbol == symbol)
        if free_base > 0 and tracked_qty < free_base * BALANCE_TOLERANCE:
            log.warning(
                f"Untracked {base_asset} balance on exchange: "
                f"{free_base:.6f} (tracked={tracked_qty:.6f}) — "
                f"close manually or delete {self.persist_path}"
            )

        self._persist()

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

    def can_open_trade(self, symbol: str) -> tuple[bool, str]:
        if len(self.open_positions) >= settings.MAX_OPEN_TRADES:
            return False, f"Max open trades ({settings.MAX_OPEN_TRADES}) reached"
        if any(p.symbol == symbol for p in self.open_positions):
            return False, f"Already have open position in {symbol}"
        return True, "OK"

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
        self._persist()
        log.info(
            f"Position opened: {symbol} @ {entry_price:.2f} | "
            f"SL={sl:.2f} | TP={tp:.2f} | Trail activates at "
            f"{entry_price * (1 + settings.TRAIL_ACTIVATE_PCT/100):.2f}"
        )
        return pos

    def close_position(self, symbol: str) -> Optional[Position]:
        for i, pos in enumerate(self.open_positions):
            if pos.symbol == symbol:
                self.open_positions.pop(i)
                self._persist()
                log.info(f"Position closed: {symbol}")
                return pos
        return None

    def check_exits(self, bar_high: float, bar_low: float) -> list[tuple[Position, str]]:
        """Check all positions for intrabar TP/SL/trailing exits."""
        exits = []
        for pos in self.open_positions:
            state = pos.to_state()
            reason, _ = check_bar_exit(state, bar_high, bar_low)
            pos.apply_state(state)
            if reason:
                exits.append((pos, reason))
        return exits

    def summary(self) -> str:
        if not self.open_positions:
            return "No open positions."
        lines = []
        for p in self.open_positions:
            trail_str = (
                f"TRAIL active (high={p.highest_price:.2f})"
                if p.trailing_active else "trail pending"
            )
            lines.append(
                f"  {p.symbol} | entry={p.entry_price:.2f} | "
                f"SL={p.stop_loss:.2f} | TP={p.take_profit:.2f} | "
                f"qty={p.quantity:.6f} | {trail_str}"
            )
        return "\n".join(lines)
