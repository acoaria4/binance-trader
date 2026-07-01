"""
simulation.py
Shared long-trade exit simulation with trailing stops.

Used by labeling, backtesting, live exit checks, and retrain validation
so every path uses identical SL / TP / trailing rules.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from config import settings


@dataclass
class LongTradeState:
    entry_price: float
    stop_loss: float
    take_profit: float
    highest_price: float
    trailing_active: bool = False

    @classmethod
    def from_entry(cls, entry_price: float) -> "LongTradeState":
        return cls(
            entry_price=entry_price,
            stop_loss=entry_price * (1 - settings.STOP_LOSS_PCT / 100),
            take_profit=entry_price * (1 + settings.TAKE_PROFIT_PCT / 100),
            highest_price=entry_price,
        )


def update_trailing(state: LongTradeState, bar_high: float) -> None:
    """Mirror live Position.update_trailing_stop using the bar high."""
    if bar_high > state.highest_price:
        state.highest_price = bar_high

    gain_pct = (state.highest_price - state.entry_price) / state.entry_price * 100
    if gain_pct >= settings.TRAIL_ACTIVATE_PCT:
        state.trailing_active = True

    if state.trailing_active:
        trail_distance = settings.TRAIL_DISTANCE_PCT
        new_sl = state.highest_price * (1 - trail_distance / 100)
        if new_sl > state.stop_loss:
            state.stop_loss = new_sl


def check_bar_exit(
    state: LongTradeState,
    bar_high: float,
    bar_low: float,
) -> tuple[Optional[str], Optional[float]]:
    """
    Process one bar (update trail, then check intrabar TP/SL).
    Returns (reason, exit_price). SL wins if both trigger same bar.
    """
    update_trailing(state, bar_high)

    tp_hit = bar_high >= state.take_profit
    sl_hit = bar_low <= state.stop_loss

    if sl_hit and tp_hit:
        return ("trailing_stop" if state.trailing_active else "stop_loss"), state.stop_loss
    if sl_hit:
        reason = "trailing_stop" if state.trailing_active else "stop_loss"
        return reason, state.stop_loss
    if tp_hit:
        return "take_profit", state.take_profit
    return None, None


def pnl_at_price(entry_price: float, exit_price: float) -> float:
    return (exit_price - entry_price) / entry_price


def apply_entry_costs(price: float) -> float:
    """Effective buy price after slippage + fee."""
    pct = (settings.SLIPPAGE_PCT + settings.FEE_PCT) / 100
    return price * (1 + pct)


def apply_exit_costs(price: float) -> float:
    """Effective sell price after slippage + fee."""
    pct = (settings.SLIPPAGE_PCT + settings.FEE_PCT) / 100
    return price * (1 - pct)


def net_pnl_fraction(entry_price: float, exit_price: float) -> float:
    """Net fractional PnL after round-trip costs."""
    entry_eff = apply_entry_costs(entry_price)
    exit_eff  = apply_exit_costs(exit_price)
    return (exit_eff - entry_eff) / entry_eff


def label_from_pnl(pnl: float) -> int:
    """Map trade outcome to ML label: BUY=+1, HOLD=0, SELL=-1."""
    if pnl > 0.001:
        return 1
    if pnl < -0.001:
        return -1
    return 0


def simulate_forward_trade(
    entry_price: float,
    highs: np.ndarray,
    lows: np.ndarray,
    horizon: int = None,
) -> tuple[int, float, Optional[str]]:
    """
    Walk forward through bars after entry.
    Returns (label, pnl_fraction, exit_reason).
    """
    horizon = horizon or settings.FORWARD_CANDLES
    state = LongTradeState.from_entry(entry_price)

    for j in range(min(horizon, len(highs))):
        reason, exit_px = check_bar_exit(state, float(highs[j]), float(lows[j]))
        if reason:
            pnl = pnl_at_price(entry_price, exit_px)
            return label_from_pnl(pnl), pnl, reason

    return 0, 0.0, None


def make_labels(df: pd.DataFrame, forward_candles: int = None) -> pd.Series:
    """
    Label each candle by simulating a long entry with trailing SL/TP rules.
    """
    horizon = forward_candles or settings.FORWARD_CANDLES
    close = df["close"].values
    high  = df["high"].values
    low   = df["low"].values
    labels = []

    for i in range(len(df)):
        if i + 1 >= len(df):
            labels.append(np.nan)
            continue

        end = min(i + 1 + horizon, len(df))
        label, _, _ = simulate_forward_trade(
            float(close[i]),
            high[i + 1:end],
            low[i + 1:end],
            horizon=end - i - 1,
        )
        labels.append(label)

    return pd.Series(labels, index=df.index)


def run_trading_simulation(
    df_raw: pd.DataFrame,
    row_indices: np.ndarray,
    predict_fn,
    regime_check_fn=None,
    evaluate_fn=None,
    min_window: int = 250,
) -> dict:
    """
    Lightweight long-only backtest over selected row indices.
    predict_fn(window_df) -> (signal, confidence)
    regime_check_fn(window_df) -> bool (True = allow entry)
  """
    from features import compute_features, is_trending

    in_trade = False
    state: Optional[LongTradeState] = None
    pnls: list[float] = []

    for abs_i in row_indices:
        if abs_i < min_window:
            continue

        window = df_raw.iloc[max(0, abs_i - min_window):abs_i + 1]
        row = df_raw.iloc[abs_i]
        bar_high = float(row["high"])
        bar_low  = float(row["low"])

        if in_trade and state is not None:
            reason, exit_px = check_bar_exit(state, bar_high, bar_low)
            if reason is None:
                if evaluate_fn is not None:
                    ev = evaluate_fn(window)
                    if ev.should_exit:
                        reason, exit_px = "signal", apply_exit_costs(float(row["close"]))
                else:
                    signal, conf = predict_fn(window)
                    if signal == "SELL" and conf >= settings.MIN_SIGNAL_CONFIDENCE:
                        reason, exit_px = "signal", apply_exit_costs(float(row["close"]))

            if reason:
                pnls.append(net_pnl_fraction(state.entry_price, exit_px))
                in_trade = False
                state = None
            continue

        if regime_check_fn is not None:
            feat = compute_features(window)
            if not regime_check_fn(feat):
                continue
        elif settings.REQUIRE_TREND:
            feat = compute_features(window)
            if not is_trending(feat, settings.ADX_THRESHOLD):
                continue

        if evaluate_fn is not None:
            ev = evaluate_fn(window)
            if not ev.should_enter:
                continue
            in_trade = True
            state = LongTradeState.from_entry(apply_entry_costs(float(row["close"])))
            continue

        signal, conf = predict_fn(window)
        if signal == "BUY" and conf >= settings.MIN_SIGNAL_CONFIDENCE:
            in_trade = True
            state = LongTradeState.from_entry(apply_entry_costs(float(row["close"])))

    return compute_trade_metrics(pnls)


def compute_trade_metrics(pnls: list[float]) -> dict:
    """Aggregate closed-trade stats from fractional PnL list."""
    if not pnls:
        return {
            "n_trades": 0,
            "win_rate": 0.0,
            "return_pct": 0.0,
            "sharpe": 0.0,
            "score": 0.0,
        }

    rets = pd.Series(pnls)
    wins = (rets > 0).sum()
    capital = 1000.0
    for r in pnls:
        capital *= (1 + r)
    return_pct = (capital - 1000.0) / 1000.0 * 100

    sharpe = 0.0
    if len(rets) > 1 and rets.std() > 0:
        sharpe = float(rets.mean() / rets.std() * np.sqrt(252))

    # Composite deploy score: return% plus scaled Sharpe
    score = return_pct + sharpe * 5.0

    return {
        "n_trades": len(pnls),
        "win_rate": wins / len(pnls) * 100,
        "return_pct": return_pct,
        "sharpe": sharpe,
        "score": score,
    }
