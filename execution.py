"""
execution.py
Execution fidelity: spread checks, limit orders with fill polling, market fallback.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import ccxt

from config import settings
from exchange import (
    get_spread_pct,
    place_limit_buy,
    place_limit_sell,
    place_market_buy,
    place_market_sell,
    wait_for_order_fill,
    cancel_order_safe,
    normalize_fill,
)
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class ExecutionResult:
    success: bool
    symbol: str
    side: str
    order_id: Optional[str]
    requested_qty: float
    filled_qty: float
    avg_price: float
    notional_usdt: float
    order_type: str
    spread_pct: float = 0.0
    reason: str = ""

    @property
    def fully_filled(self) -> bool:
        if self.requested_qty <= 0:
            return self.filled_qty > 0
        return self.filled_qty >= self.requested_qty * 0.99


def check_spread(exchange: ccxt.binance, symbol: str) -> tuple[bool, float, str]:
    """
    Returns (ok, spread_pct, reason).
    Blocks entry when spread exceeds MAX_SPREAD_PCT.
    """
    spread_pct, bid, ask = get_spread_pct(exchange, symbol)
    if bid <= 0 or ask <= 0:
        return False, spread_pct, "invalid_order_book"

    if spread_pct > settings.MAX_SPREAD_PCT:
        log.warning(
            f"Spread too wide for {symbol}: {spread_pct:.3f}% "
            f"(max {settings.MAX_SPREAD_PCT}%) bid={bid:.2f} ask={ask:.2f}"
        )
        return False, spread_pct, f"spread_{spread_pct:.3f}pct"

    return True, spread_pct, "OK"


def _limit_buy_price(exchange: ccxt.binance, symbol: str) -> float:
    """Aggressive limit inside the spread — slightly below ask."""
    spread_pct, bid, ask = get_spread_pct(exchange, symbol)
    if ask <= 0:
        ticker = exchange.fetch_ticker(symbol)
        ask = float(ticker["last"])
    offset = settings.LIMIT_ORDER_OFFSET_PCT / 100
    price = ask * (1 - offset)
    if bid > 0:
        price = max(price, bid)
    return float(exchange.price_to_precision(symbol, price))


def _limit_sell_price(exchange: ccxt.binance, symbol: str) -> float:
    """Aggressive limit inside the spread — slightly above bid."""
    spread_pct, bid, ask = get_spread_pct(exchange, symbol)
    if bid <= 0:
        ticker = exchange.fetch_ticker(symbol)
        bid = float(ticker["last"])
    offset = settings.LIMIT_ORDER_OFFSET_PCT / 100
    price = bid * (1 + offset)
    if ask > 0:
        price = min(price, ask)
    return float(exchange.price_to_precision(symbol, price))


def execute_buy(
    exchange: ccxt.binance,
    symbol: str,
    amount_usdt: float,
    check_spread_gate: bool = True,
) -> ExecutionResult:
    """Enter long with limit order (optional spread gate) and market fallback."""
    if amount_usdt <= 0:
        return ExecutionResult(
            success=False, symbol=symbol, side="buy", order_id=None,
            requested_qty=0.0, filled_qty=0.0, avg_price=0.0,
            notional_usdt=0.0, order_type="none", reason="zero_notional",
        )

    spread_pct = 0.0
    if check_spread_gate:
        ok, spread_pct, reason = check_spread(exchange, symbol)
        if not ok:
            return ExecutionResult(
                success=False, symbol=symbol, side="buy", order_id=None,
                requested_qty=0.0, filled_qty=0.0, avg_price=0.0,
                notional_usdt=0.0, order_type="blocked", spread_pct=spread_pct,
                reason=reason,
            )

    ticker = exchange.fetch_ticker(symbol)
    ref_price = float(ticker["last"])
    requested_qty = float(
        exchange.amount_to_precision(symbol, amount_usdt / ref_price)
    )

    if not settings.USE_LIMIT_ORDERS:
        order = place_market_buy(exchange, symbol, amount_usdt)
        fill = normalize_fill(order, ref_price)
        return ExecutionResult(
            success=fill["filled_qty"] > 0,
            symbol=symbol, side="buy", order_id=fill["order_id"],
            requested_qty=requested_qty,
            filled_qty=fill["filled_qty"],
            avg_price=fill["avg_price"],
            notional_usdt=fill["filled_qty"] * fill["avg_price"],
            order_type="market", spread_pct=spread_pct, reason="market_buy",
        )

    limit_price = _limit_buy_price(exchange, symbol)
    order = place_limit_buy(exchange, symbol, amount_usdt, limit_price)
    order_id = str(order.get("id", ""))
    filled = wait_for_order_fill(
        exchange, symbol, order_id,
        timeout_sec=settings.LIMIT_ORDER_TIMEOUT_SEC,
    )

    if filled["filled_qty"] > 0 and filled["status"] == "closed":
        return ExecutionResult(
            success=True, symbol=symbol, side="buy", order_id=order_id,
            requested_qty=requested_qty,
            filled_qty=filled["filled_qty"],
            avg_price=filled["avg_price"],
            notional_usdt=filled["filled_qty"] * filled["avg_price"],
            order_type="limit", spread_pct=spread_pct, reason="limit_filled",
        )

    remaining_qty = max(0.0, requested_qty - filled["filled_qty"])
    if filled["filled_qty"] > 0:
        cancel_order_safe(exchange, order_id, symbol)

    if remaining_qty > 0 and settings.LIMIT_FALLBACK_TO_MARKET:
        log.info(
            f"Limit buy partial/timeout — market fallback for "
            f"{remaining_qty:.6f} {symbol}"
        )
        remaining_usdt = remaining_qty * ref_price
        market_order = place_market_buy(exchange, symbol, remaining_usdt)
        market_fill = normalize_fill(market_order, ref_price)
        total_qty = filled["filled_qty"] + market_fill["filled_qty"]
        if total_qty > 0:
            total_cost = (
                filled["filled_qty"] * filled["avg_price"]
                + market_fill["filled_qty"] * market_fill["avg_price"]
            )
            avg_price = total_cost / total_qty
            return ExecutionResult(
                success=True, symbol=symbol, side="buy", order_id=order_id,
                requested_qty=requested_qty,
                filled_qty=total_qty,
                avg_price=avg_price,
                notional_usdt=total_qty * avg_price,
                order_type="limit+market", spread_pct=spread_pct,
                reason="limit_partial_market_fallback",
            )

    if filled["filled_qty"] > 0:
        return ExecutionResult(
            success=True, symbol=symbol, side="buy", order_id=order_id,
            requested_qty=requested_qty,
            filled_qty=filled["filled_qty"],
            avg_price=filled["avg_price"],
            notional_usdt=filled["filled_qty"] * filled["avg_price"],
            order_type="limit", spread_pct=spread_pct, reason="limit_partial",
        )

    cancel_order_safe(exchange, order_id, symbol)
    return ExecutionResult(
        success=False, symbol=symbol, side="buy", order_id=order_id,
        requested_qty=requested_qty, filled_qty=0.0, avg_price=0.0,
        notional_usdt=0.0, order_type="limit", spread_pct=spread_pct,
        reason="limit_timeout_no_fill",
    )


def execute_sell(
    exchange: ccxt.binance,
    symbol: str,
    quantity: float,
    urgent: bool = True,
) -> ExecutionResult:
    """
    Exit long. Urgent exits (SL/TP) use market by default.
  Optional limit exits when USE_LIMIT_EXITS=true and urgent=False.
    """
    if quantity <= 0:
        return ExecutionResult(
            success=False, symbol=symbol, side="sell", order_id=None,
            requested_qty=0.0, filled_qty=0.0, avg_price=0.0,
            notional_usdt=0.0, order_type="none", reason="zero_qty",
        )

    qty = float(exchange.amount_to_precision(symbol, quantity))
    ticker = exchange.fetch_ticker(symbol)
    ref_price = float(ticker["last"])

    use_limit = settings.USE_LIMIT_EXITS and not urgent
    if not use_limit:
        order = place_market_sell(exchange, symbol, qty)
        fill = normalize_fill(order, ref_price)
        return ExecutionResult(
            success=fill["filled_qty"] > 0,
            symbol=symbol, side="sell", order_id=fill["order_id"],
            requested_qty=qty,
            filled_qty=fill["filled_qty"],
            avg_price=fill["avg_price"],
            notional_usdt=fill["filled_qty"] * fill["avg_price"],
            order_type="market", reason="market_sell",
        )

    limit_price = _limit_sell_price(exchange, symbol)
    order = place_limit_sell(exchange, symbol, qty, limit_price)
    order_id = str(order.get("id", ""))
    filled = wait_for_order_fill(
        exchange, symbol, order_id,
        timeout_sec=settings.LIMIT_ORDER_TIMEOUT_SEC,
    )

    if filled["filled_qty"] >= qty * 0.99:
        return ExecutionResult(
            success=True, symbol=symbol, side="sell", order_id=order_id,
            requested_qty=qty,
            filled_qty=filled["filled_qty"],
            avg_price=filled["avg_price"],
            notional_usdt=filled["filled_qty"] * filled["avg_price"],
            order_type="limit", reason="limit_filled",
        )

    remaining = max(0.0, qty - filled["filled_qty"])
    if remaining > 0:
        cancel_order_safe(exchange, order_id, symbol)
        if settings.LIMIT_FALLBACK_TO_MARKET:
            market_order = place_market_sell(exchange, symbol, remaining)
            market_fill = normalize_fill(market_order, ref_price)
            total_qty = filled["filled_qty"] + market_fill["filled_qty"]
            if total_qty > 0:
                total_proceeds = (
                    filled["filled_qty"] * filled["avg_price"]
                    + market_fill["filled_qty"] * market_fill["avg_price"]
                )
                avg_price = total_proceeds / total_qty
                return ExecutionResult(
                    success=True, symbol=symbol, side="sell", order_id=order_id,
                    requested_qty=qty,
                    filled_qty=total_qty,
                    avg_price=avg_price,
                    notional_usdt=total_qty * avg_price,
                    order_type="limit+market", reason="limit_partial_market_fallback",
                )

    return ExecutionResult(
        success=filled["filled_qty"] > 0,
        symbol=symbol, side="sell", order_id=order_id,
        requested_qty=qty,
        filled_qty=filled["filled_qty"],
        avg_price=filled["avg_price"],
        notional_usdt=filled["filled_qty"] * filled["avg_price"],
        order_type="limit", reason="limit_partial_or_timeout",
    )
