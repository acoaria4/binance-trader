"""
risk_sizing.py
Volatility-based barriers, position sizing, and portfolio heat.
"""
from __future__ import annotations

from dataclasses import dataclass

from config import settings


@dataclass
class TradeBarriers:
    stop_loss: float
    take_profit: float
    stop_loss_pct: float
    take_profit_pct: float
    trail_distance_pct: float


def barriers_for_entry(entry_price: float, atr: float = None) -> TradeBarriers:
    """
    Compute SL/TP/trail from ATR multiples or fixed % settings.
    Clamps ATR-derived % to configured min/max bounds.
    """
    if settings.USE_ATR_STOPS and atr is not None and atr > 0 and entry_price > 0:
        sl_dist = atr * settings.ATR_SL_MULT
        tp_dist = atr * settings.ATR_TP_MULT
        sl = max(entry_price - sl_dist, entry_price * 0.5)
        tp = entry_price + tp_dist
        sl_pct = (entry_price - sl) / entry_price * 100
        tp_pct = (tp - entry_price) / entry_price * 100
        trail_pct = (atr * settings.ATR_TRAIL_MULT / entry_price) * 100
    else:
        sl_pct = settings.STOP_LOSS_PCT
        tp_pct = settings.TAKE_PROFIT_PCT
        trail_pct = settings.TRAIL_DISTANCE_PCT
        sl = entry_price * (1 - sl_pct / 100)
        tp = entry_price * (1 + tp_pct / 100)

    sl_pct = float(max(settings.MIN_STOP_LOSS_PCT, min(sl_pct, settings.MAX_STOP_LOSS_PCT)))
    tp_pct = float(max(settings.MIN_TAKE_PROFIT_PCT, min(tp_pct, settings.MAX_TAKE_PROFIT_PCT)))
    trail_pct = float(max(trail_pct, settings.MIN_TRAIL_DISTANCE_PCT))

    sl = entry_price * (1 - sl_pct / 100)
    tp = entry_price * (1 + tp_pct / 100)

    return TradeBarriers(
        stop_loss=sl,
        take_profit=tp,
        stop_loss_pct=sl_pct,
        take_profit_pct=tp_pct,
        trail_distance_pct=trail_pct,
    )


def calculate_position_size(
    equity: float,
    entry_price: float,
    stop_loss: float,
    size_multiplier: float = 1.0,
) -> dict:
    """
    Risk-based sizing: risk RISK_PCT_PER_TRADE of equity per trade.
    Capped at MAX_POSITION_USDT and 95% of available equity.
    """
    stop_distance = entry_price - stop_loss
    if stop_distance <= 0 or entry_price <= 0 or equity <= 0:
        return dict(qty=0.0, trade_usdt=0.0, risk_usdt=0.0)

    risk_budget = equity * (settings.RISK_PCT_PER_TRADE / 100) * size_multiplier
    qty         = risk_budget / stop_distance
    trade_usdt  = qty * entry_price

    cap = min(settings.MAX_POSITION_USDT, equity * 0.95)
    if trade_usdt > cap:
        trade_usdt = cap
        qty = trade_usdt / entry_price

    if settings.TRADE_AMOUNT_USDT > 0 and not settings.USE_RISK_SIZING:
        trade_usdt = min(settings.TRADE_AMOUNT_USDT, cap)
        qty = trade_usdt / entry_price
        risk_budget = stop_distance * qty

    return dict(
        qty=qty,
        trade_usdt=trade_usdt,
        risk_usdt=risk_budget,
        stop_loss=stop_loss,
    )


def portfolio_heat_pct(positions, equity: float) -> float:
    """Sum of open risk (entry to SL) as % of equity."""
    if equity <= 0:
        return 0.0
    total_risk = sum(
        max(0.0, (p.entry_price - p.stop_loss) * p.quantity)
        for p in positions
    )
    return total_risk / equity * 100
