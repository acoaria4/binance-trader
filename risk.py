"""
risk.py
Risk management: volatility sizing, portfolio heat, daily loss limits,
trailing stops, and position persistence.
"""
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

import ccxt

from config import settings
from simulation import LongTradeState, check_bar_exit
from risk_sizing import (
    TradeBarriers, barriers_for_entry,
    calculate_position_size, portfolio_heat_pct,
)
from portfolio_state import PortfolioState
from reconciliation import reconcile_positions
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
    trail_distance_pct: float = 0.0
    risk_usdt: float = 0.0

    def __post_init__(self):
        self.highest_price = self.entry_price

    def to_state(self) -> LongTradeState:
        return LongTradeState(
            entry_price=self.entry_price,
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            highest_price=self.highest_price,
            trailing_active=self.trailing_active,
            trail_distance_pct=self.trail_distance_pct or settings.TRAIL_DISTANCE_PCT,
        )

    def apply_state(self, state: LongTradeState) -> None:
        self.stop_loss = state.stop_loss
        self.highest_price = state.highest_price
        self.trailing_active = state.trailing_active
        if state.trail_distance_pct is not None:
            self.trail_distance_pct = state.trail_distance_pct

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
        trail_pct = d.pop("trail_distance_pct", settings.TRAIL_DISTANCE_PCT)
        risk_usdt = d.pop("risk_usdt", 0.0)
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
            risk_usdt=float(risk_usdt),
        )
        pos.trailing_active = bool(trailing)
        pos.trail_distance_pct = float(trail_pct)
        if highest is not None:
            pos.highest_price = float(highest)
        return pos

    def pnl_pct(self, current_price: float) -> float:
        return (current_price - self.entry_price) / self.entry_price * 100


class RiskManager:
    def __init__(self, persist_path: str = None):
        self.persist_path = persist_path or settings.POSITIONS_FILE
        self.open_positions: list[Position] = []
        self.portfolio = PortfolioState.load()
        self.load()

    def _persist(self) -> None:
        os.makedirs(os.path.dirname(self.persist_path) or ".", exist_ok=True)
        with open(self.persist_path, "w") as f:
            json.dump([p.to_dict() for p in self.open_positions], f, indent=2)
        self.portfolio.save()

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
        reconcile_positions(exchange, self, symbol=symbol, adopt_untracked=False)

    def record_closed_trade(self, pnl_pct: float, equity: float) -> None:
        self.portfolio.record_closed_trade(pnl_pct, equity)
        self.portfolio.save()

    def calculate_position(
        self,
        entry_price: float,
        available_usdt: float,
        atr: float = None,
    ) -> dict:
        barriers = barriers_for_entry(entry_price, atr)
        sizing = calculate_position_size(
            equity=available_usdt,
            entry_price=entry_price,
            stop_loss=barriers.stop_loss,
            size_multiplier=self.portfolio.size_multiplier(),
        )
        sizing["stop_loss"] = barriers.stop_loss
        sizing["take_profit"] = barriers.take_profit
        sizing["trail_distance_pct"] = barriers.trail_distance_pct
        sizing["stop_loss_pct"] = barriers.stop_loss_pct
        sizing["take_profit_pct"] = barriers.take_profit_pct

        log.info(
            f"Position sizing: qty={sizing['qty']:.6f} | "
            f"USDT={sizing['trade_usdt']:.2f} | "
            f"SL={barriers.stop_loss:.2f} ({barriers.stop_loss_pct:.2f}%) | "
            f"TP={barriers.take_profit:.2f} ({barriers.take_profit_pct:.2f}%) | "
            f"Risk={sizing['risk_usdt']:.2f} USDT | "
            f"mult={self.portfolio.size_multiplier():.2f}"
        )
        return sizing

    def can_open_trade(self, symbol: str, equity: float,
                       proposed_risk_usdt: float = 0.0) -> tuple[bool, str]:
        ok, reason = self.portfolio.can_enter_new()
        if not ok:
            return False, reason

        if len(self.open_positions) >= settings.MAX_OPEN_TRADES:
            return False, f"max_open_trades_{settings.MAX_OPEN_TRADES}"

        if any(p.symbol == symbol for p in self.open_positions):
            return False, f"already_open_{symbol}"

        heat = portfolio_heat_pct(self.open_positions, equity)
        if proposed_risk_usdt > 0:
            new_heat = heat + (proposed_risk_usdt / equity * 100 if equity > 0 else 0)
            if new_heat > settings.MAX_PORTFOLIO_HEAT_PCT:
                return False, f"portfolio_heat_{new_heat:.1f}pct"

        return True, "OK"

    def open_position(
        self,
        symbol: str,
        entry_price: float,
        quantity: float,
        stop_loss: float,
        take_profit: float,
        trail_distance_pct: float = None,
        risk_usdt: float = 0.0,
        order_id: str = None,
    ) -> Position:
        pos = Position(
            symbol=symbol,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            order_id=order_id,
            trail_distance_pct=trail_distance_pct or settings.TRAIL_DISTANCE_PCT,
            risk_usdt=risk_usdt,
        )
        self.open_positions.append(pos)
        self._persist()
        log.info(
            f"Position opened: {symbol} @ {entry_price:.2f} | "
            f"SL={stop_loss:.2f} | TP={take_profit:.2f} | "
            f"trail_dist={pos.trail_distance_pct:.2f}%"
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
        exits = []
        for pos in self.open_positions:
            state = pos.to_state()
            reason, _ = check_bar_exit(state, bar_high, bar_low)
            pos.apply_state(state)
            if reason:
                exits.append((pos, reason))
        return exits

    def portfolio_summary(self, equity: float) -> str:
        heat = portfolio_heat_pct(self.open_positions, equity)
        dd = self.portfolio.drawdown_pct(equity)
        return (
            f"Heat: {heat:.2f}% / {settings.MAX_PORTFOLIO_HEAT_PCT}% | "
            f"Daily PnL: {self.portfolio.daily_pnl_pct:+.2f}% | "
            f"Consec losses: {self.portfolio.consecutive_losses} | "
            f"DD from peak: {dd:.2f}%"
        )

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
                f"qty={p.quantity:.6f} | risk={p.risk_usdt:.2f} | {trail_str}"
            )
        return "\n".join(lines)
