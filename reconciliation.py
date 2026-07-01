"""
reconciliation.py
Periodic reconciliation between exchange state and local position/order tracking.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

import ccxt

from config import settings
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class ReconciliationReport:
    positions_kept: int = 0
    positions_dropped: int = 0
    positions_adopted: int = 0
    stale_orders_cancelled: int = 0
    open_orders_found: int = 0
    untracked_balance: float = 0.0
    warnings: list = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


def _order_age_sec(order: dict) -> float:
    ts = order.get("timestamp") or order.get("datetime")
    if ts is None:
        return 0.0
    if isinstance(ts, (int, float)):
        order_ms = float(ts)
        if order_ms < 1e12:
            order_ms *= 1000
    else:
        try:
            order_ms = datetime.fromisoformat(
                str(ts).replace("Z", "+00:00")
            ).timestamp() * 1000
        except ValueError:
            return 0.0
    return (datetime.now(timezone.utc).timestamp() * 1000 - order_ms) / 1000


def cancel_stale_orders(
    exchange: ccxt.binance,
    symbol: str,
    max_age_sec: int = None,
) -> int:
    """Cancel open orders older than max_age_sec."""
    max_age_sec = max_age_sec or settings.CANCEL_STALE_ORDERS_SEC
    cancelled = 0
    try:
        open_orders = exchange.fetch_open_orders(symbol)
    except Exception as e:
        log.warning(f"fetch_open_orders failed: {e}")
        return 0

    for order in open_orders:
        age = _order_age_sec(order)
        if age >= max_age_sec:
            try:
                exchange.cancel_order(order["id"], symbol)
                cancelled += 1
                log.info(
                    f"Cancelled stale order {order['id']} "
                    f"({order.get('side')} {order.get('amount')} @ {order.get('price')}, "
                    f"age={age:.0f}s)"
                )
            except Exception as e:
                log.warning(f"Failed to cancel order {order.get('id')}: {e}")

    return cancelled


def reconcile_positions(
    exchange: ccxt.binance,
    risk_manager,
    symbol: str = None,
    adopt_untracked: bool = False,
) -> ReconciliationReport:
    """
    Full position reconciliation loop:
    - Drop positions with no matching exchange balance
    - Warn on untracked balance
    - Optionally adopt untracked balance as a new position
    - Cancel stale open orders
    """
    from risk import Position, BALANCE_TOLERANCE
    from risk_sizing import barriers_for_entry

    symbol = symbol or settings.SYMBOL
    base_asset = symbol.split("/")[0]
    report = ReconciliationReport()

    try:
        balance = exchange.fetch_balance()
        free_base = float(balance["free"].get(base_asset, 0.0))
    except Exception as e:
        report.warnings.append(f"balance_fetch_failed: {e}")
        log.warning(f"Reconciliation balance fetch failed: {e}")
        return report

    try:
        open_orders = exchange.fetch_open_orders(symbol)
        report.open_orders_found = len(open_orders)
    except Exception as e:
        report.warnings.append(f"open_orders_fetch_failed: {e}")
        open_orders = []

    report.stale_orders_cancelled = cancel_stale_orders(exchange, symbol)

    kept = []
    for pos in risk_manager.open_positions:
        if pos.symbol != symbol:
            kept.append(pos)
            report.positions_kept += 1
            continue

        if free_base >= pos.quantity * BALANCE_TOLERANCE:
            kept.append(pos)
            report.positions_kept += 1
            if free_base > pos.quantity * 1.05:
                drift = free_base - pos.quantity
                report.warnings.append(
                    f"qty_drift_{symbol}: tracked={pos.quantity:.6f} "
                    f"exchange={free_base:.6f} (+{drift:.6f})"
                )
                pos.quantity = float(
                    exchange.amount_to_precision(symbol, free_base)
                )
        else:
            report.positions_dropped += 1
            report.warnings.append(
                f"dropped_stale_{symbol}: tracked={pos.quantity:.6f} "
                f"exchange={free_base:.6f}"
            )
            log.warning(
                f"Reconciliation dropped stale position {pos.symbol}: "
                f"tracked qty={pos.quantity:.6f}, exchange free={free_base:.6f}"
            )

    risk_manager.open_positions = kept
    tracked_qty = sum(p.quantity for p in kept if p.symbol == symbol)

    if free_base > 0 and tracked_qty < free_base * BALANCE_TOLERANCE:
        report.untracked_balance = free_base - tracked_qty
        msg = (
            f"Untracked {base_asset}: exchange={free_base:.6f} "
            f"tracked={tracked_qty:.6f}"
        )
        report.warnings.append(msg)
        log.warning(msg)

        if adopt_untracked and report.untracked_balance > 0:
            try:
                ticker = exchange.fetch_ticker(symbol)
                entry = float(ticker["last"])
                barriers = barriers_for_entry(entry)
                adopted = Position(
                    symbol=symbol,
                    entry_price=entry,
                    quantity=float(
                        exchange.amount_to_precision(symbol, report.untracked_balance)
                    ),
                    stop_loss=barriers.stop_loss,
                    take_profit=barriers.take_profit,
                    trail_distance_pct=barriers.trail_distance_pct,
                )
                risk_manager.open_positions.append(adopted)
                report.positions_adopted += 1
                log.info(
                    f"Adopted untracked {base_asset} as position: "
                    f"qty={adopted.quantity:.6f} @ {entry:.2f}"
                )
            except Exception as e:
                report.warnings.append(f"adopt_failed: {e}")

    risk_manager._persist()

    log.info(
        f"Reconciliation {symbol}: kept={report.positions_kept} "
        f"dropped={report.positions_dropped} adopted={report.positions_adopted} "
        f"stale_cancelled={report.stale_orders_cancelled} "
        f"open_orders={report.open_orders_found}"
    )
    return report


def save_reconciliation_log(report: ReconciliationReport, path: str = None) -> None:
    path = path or "logs/reconciliation.jsonl"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    entry = asdict(report)
    entry["ts"] = datetime.now(timezone.utc).isoformat()
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")
