"""
ev_gate.py
Expected-value gating and conviction scoring for high-conviction entries.
"""
from __future__ import annotations

from dataclasses import dataclass

from config import settings


@dataclass
class SignalEvaluation:
    signal: str
    p_sell: float
    p_hold: float
    p_buy: float
    p_win: float
    ev: float
    conviction: float
    regime_ok: bool
    mtf_ok: bool
    should_enter: bool
    should_exit: bool
    block_reason: str = ""


def round_trip_cost_pct() -> float:
    """Total friction % (fees + slippage both sides)."""
    per_side = settings.FEE_PCT + settings.SLIPPAGE_PCT
    return 2.0 * per_side


def compute_ev(p_win: float) -> float:
    """
    Expected value of a long trade as a fraction (e.g. 0.01 = 1%).

    EV = P(win)*TP - P(loss)*SL - round_trip_costs
    """
    tp = settings.TAKE_PROFIT_PCT / 100
    sl = settings.STOP_LOSS_PCT / 100
    cost = round_trip_cost_pct() / 100
    return p_win * tp - (1.0 - p_win) * sl - cost


def compute_conviction(
    p_win: float,
    ev: float,
    regime_ok: bool,
    mtf_ok: bool,
) -> float:
    """Master conviction score in [0, 1] combining model and context."""
    ev_norm = max(0.0, min(1.0, ev / max(settings.EV_MIN * 2, 1e-9)))
    score = (
        0.40 * p_win +
        0.30 * ev_norm +
        0.15 * (1.0 if regime_ok else 0.0) +
        0.15 * (1.0 if mtf_ok else 0.0)
    )
    return float(max(0.0, min(1.0, score)))


def evaluate_signal(
    p_sell: float,
    p_hold: float,
    p_buy: float,
    regime_ok: bool,
    mtf_ok: bool,
) -> SignalEvaluation:
    """Evaluate a long entry from calibrated class probabilities."""
    probs = [p_sell, p_hold, p_buy]
    cls_id = int(max(range(3), key=lambda i: probs[i]))
    signal = {0: "SELL", 1: "HOLD", 2: "BUY"}[cls_id]
    confidence = probs[cls_id]

    p_win = float(p_buy)
    ev = compute_ev(p_win)
    conviction = compute_conviction(p_win, ev, regime_ok, mtf_ok)

    should_exit = (
        signal == "SELL" and confidence >= settings.MIN_SIGNAL_CONFIDENCE
    )

    should_enter = False
    block_reason = ""

    if signal != "BUY":
        block_reason = f"signal_{signal.lower()}"
    elif confidence < settings.MIN_SIGNAL_CONFIDENCE:
        block_reason = "low_confidence"
    elif p_win < settings.P_WIN_MIN:
        block_reason = f"p_win_below_{settings.P_WIN_MIN}"
    elif ev < settings.EV_MIN:
        block_reason = f"ev_below_{settings.EV_MIN}"
    elif not regime_ok:
        block_reason = "regime_filter"
    elif settings.REQUIRE_MTF and not mtf_ok:
        block_reason = "mtf_filter"
    elif conviction < settings.CONVICTION_MIN:
        block_reason = f"conviction_below_{settings.CONVICTION_MIN}"
    else:
        should_enter = True

    return SignalEvaluation(
        signal=signal,
        p_sell=p_sell,
        p_hold=p_hold,
        p_buy=p_buy,
        p_win=p_win,
        ev=ev,
        conviction=conviction,
        regime_ok=regime_ok,
        mtf_ok=mtf_ok,
        should_enter=should_enter,
        should_exit=should_exit,
        block_reason=block_reason,
    )
