"""
backtest/engine.py
Unified backtest engine used by CLI, validation, and research.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol

import pandas as pd

from config import settings
from features import compute_features, is_trending, is_mtf_aligned
from simulation import LongTradeState, check_bar_exit, net_pnl_fraction, apply_entry_costs, apply_exit_costs


class StrategyProtocol(Protocol):
    def predict(self, df_raw: pd.DataFrame) -> tuple[str, float]: ...
    def evaluate(self, df_raw: pd.DataFrame) -> object: ...
    def on_new_candle(self, df_raw: pd.DataFrame) -> bool: ...


@dataclass
class BacktestConfig:
    min_window: int = 250
    train_fetch_limit: int = 2000
    initial_capital: float = 1000.0
    periodic_retrain: bool = True
    apply_costs: bool = True
    symbol: str = field(default_factory=lambda: settings.SYMBOL)
    timeframe: str = field(default_factory=lambda: settings.TIMEFRAME)


@dataclass
class BacktestResult:
    trades: list
    equity: list
    gross_equity: list
    skipped: dict
    signal_counts: dict
    retrain_count: int
    symbol: str
    timeframe: str
    n_candles: int

    def to_report_dict(self) -> dict:
        gross_start = self.gross_equity[0] if self.gross_equity else 1000.0
        gross_end   = self.gross_equity[-1] if self.gross_equity else 1000.0
        net_start   = self.equity[0] if self.equity else 1000.0
        net_end     = self.equity[-1] if self.equity else 1000.0
        return {
            "trades": self.trades,
            "equity": self.equity,
            "gross_return_pct": (gross_end - gross_start) / gross_start * 100,
            "cost_drag_pct": (gross_end - gross_start) / gross_start * 100 - (net_end - net_start) / net_start * 100,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "n_candles": self.n_candles,
            "skipped": self.skipped,
            "signal_counts": self.signal_counts,
            "_equity": self.equity,
        }


class BacktestEngine:
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()

    def run(
        self,
        df_raw: pd.DataFrame,
        strategy: StrategyProtocol,
        test_start_idx: int = None,
        test_end_idx: int = None,
    ) -> BacktestResult:
        cfg = self.config
        n = len(df_raw)
        start = test_start_idx if test_start_idx is not None else int(n * 0.6)
        end   = test_end_idx if test_end_idx is not None else n

        capital       = cfg.initial_capital
        gross_capital = cfg.initial_capital
        equity        = [capital]
        gross_equity  = [gross_capital]
        trades        = []
        in_trade      = False
        trade_state: Optional[LongTradeState] = None
        skipped = {
            "regime": 0, "ev_gate": 0, "confidence": 0,
            "sell_flat": 0, "mtf": 0,
        }
        signal_counts = {}
        retrain_count = 0

        test_slice = df_raw.iloc[start:end]

        for i, (ts, row) in enumerate(test_slice.iterrows()):
            abs_i = start + i
            window = df_raw.iloc[
                max(0, abs_i - max(settings.LOOKBACK_CANDLES, cfg.min_window)):abs_i + 1
            ]
            if len(window) < cfg.min_window:
                continue

            bar_high  = float(row["high"])
            bar_low   = float(row["low"])
            bar_close = float(row["close"])

            if cfg.periodic_retrain:
                train_window = df_raw.iloc[max(0, abs_i + 1 - cfg.train_fetch_limit):abs_i + 1]
                if strategy.on_new_candle(train_window):
                    retrain_count += 1

            evaluation = strategy.evaluate(window)
            signal_counts[evaluation.signal] = signal_counts.get(evaluation.signal, 0) + 1

            if in_trade and trade_state is not None:
                reason, exit_px = check_bar_exit(trade_state, bar_high, bar_low)
                if reason is None and evaluation.should_exit:
                    reason, exit_px = "signal", apply_exit_costs(bar_close) if cfg.apply_costs else bar_close

                if reason:
                    gross_pnl = (exit_px - trade_state.entry_price) / trade_state.entry_price
                    net_pnl   = net_pnl_fraction(trade_state.entry_price, exit_px) if cfg.apply_costs else gross_pnl
                    capital       *= (1 + net_pnl)
                    gross_capital *= (1 + gross_pnl)
                    equity.append(capital)
                    gross_equity.append(gross_capital)
                    trades.append({
                        "ts": ts, "action": f"SELL_{reason.upper()}",
                        "price": exit_px, "pnl_pct": net_pnl * 100,
                        "gross_pnl_pct": gross_pnl * 100,
                    })
                    in_trade = False
                    trade_state = None
                continue

            if not evaluation.regime_ok:
                skipped["regime"] += 1
                continue

            if settings.REQUIRE_MTF and not evaluation.mtf_ok:
                skipped["mtf"] += 1
                continue

            if evaluation.should_enter:
                entry_eff = apply_entry_costs(bar_close) if cfg.apply_costs else bar_close
                in_trade = True
                trade_state = LongTradeState.from_entry(entry_eff)
                trades.append({
                    "ts": ts, "action": "BUY", "price": bar_close,
                    "p_win": evaluation.p_win, "ev": evaluation.ev,
                    "conviction": evaluation.conviction,
                })
            else:
                if evaluation.signal == "BUY":
                    if evaluation.block_reason.startswith("ev_") or evaluation.block_reason.startswith("conviction") or evaluation.block_reason.startswith("p_win"):
                        skipped["ev_gate"] += 1
                    else:
                        skipped["confidence"] += 1
                elif evaluation.should_exit:
                    skipped["sell_flat"] += 1

        return BacktestResult(
            trades=trades,
            equity=equity,
            gross_equity=gross_equity,
            skipped=skipped,
            signal_counts=signal_counts,
            retrain_count=retrain_count,
            symbol=cfg.symbol,
            timeframe=cfg.timeframe,
            n_candles=len(test_slice),
        )
