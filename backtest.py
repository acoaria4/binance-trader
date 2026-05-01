"""
backtest.py
Simple walk-forward backtester for the ML strategy.

Usage:
    python backtest.py --symbol BTC/USDT --timeframe 1h --limit 1000

Shows:
  - Total return, win rate, Sharpe ratio
  - Trade-by-trade log
  - Equity curve (ASCII)
"""
import argparse
import pandas as pd
import numpy as np

from exchange  import get_exchange, fetch_ohlcv
from features  import compute_features, FEATURE_COLS
from strategy  import MLStrategy, _label, FORWARD_CANDLES
from config    import settings
from utils.logger import get_logger

log = get_logger("backtest")


def run_backtest(symbol: str, timeframe: str, limit: int,
                 train_frac: float = 0.6) -> None:

    exchange = get_exchange()
    df_raw   = fetch_ohlcv(exchange, symbol=symbol, timeframe=timeframe, limit=limit)
    df       = compute_features(df_raw)
    df["label"] = _label(df)

    split = int(len(df) * train_frac)
    df_train = df.iloc[:split]
    df_test  = df.iloc[split:]

    log.info(f"Train candles: {len(df_train)} | Test candles: {len(df_test)}")

    strat = MLStrategy()
    strat.train(df_train.drop(columns=["label"]))

    # Walk-forward simulation
    capital    = 1000.0   # USDT
    equity     = [capital]
    trades     = []
    in_trade   = False
    entry_p    = 0.0

    for i, (ts, row) in enumerate(df_test.iterrows()):
        if i < 1:
            continue

        window = df_test.iloc[max(0, i-settings.LOOKBACK_CANDLES):i]
        if len(window) < 52:
            continue

        signal, conf = strat.predict(window)
        price = row["close"]

        if not in_trade and signal == "BUY" and conf >= settings.MIN_SIGNAL_CONFIDENCE:
            entry_p  = price
            in_trade = True
            trades.append({"ts": ts, "action": "BUY", "price": price, "conf": conf})

        elif in_trade:
            sl = entry_p * (1 - settings.STOP_LOSS_PCT   / 100)
            tp = entry_p * (1 + settings.TAKE_PROFIT_PCT / 100)
            reason = None
            if price <= sl:   reason = "SL"
            elif price >= tp: reason = "TP"
            elif signal == "SELL" and conf >= settings.MIN_SIGNAL_CONFIDENCE:
                reason = "SIGNAL"

            if reason:
                pnl_pct  = (price - entry_p) / entry_p
                capital *= (1 + pnl_pct)
                equity.append(capital)
                trades.append({
                    "ts": ts, "action": f"SELL_{reason}",
                    "price": price, "conf": conf,
                    "pnl_pct": pnl_pct * 100,
                })
                in_trade = False

    # ── Summary ──────────────────────────────────────────────────────────────
    closed = [t for t in trades if "pnl_pct" in t]
    wins   = [t for t in closed if t["pnl_pct"] > 0]
    losses = [t for t in closed if t["pnl_pct"] <= 0]
    total_return = (capital - 1000) / 1000 * 100

    rets = pd.Series([t["pnl_pct"] / 100 for t in closed])
    sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0

    print("\n" + "─" * 52)
    print(f"  Backtest: {symbol} | {timeframe} | {len(df_test)} candles")
    print("─" * 52)
    print(f"  Trades:       {len(closed)}")
    print(f"  Win rate:     {len(wins)/len(closed)*100:.1f}%" if closed else "  Win rate: N/A")
    print(f"  Total return: {total_return:+.2f}%")
    print(f"  Sharpe:       {sharpe:.2f}")
    print(f"  Final equity: {capital:.2f} USDT (start: 1000)")
    print("─" * 52)

    # ASCII equity curve
    if equity:
        mn, mx = min(equity), max(equity)
        height = 8
        rows   = []
        for row_i in range(height, 0, -1):
            line = ""
            for val in equity[::max(1, len(equity)//60)]:
                norm = (val - mn) / (mx - mn + 1e-9)
                line += "█" if norm >= row_i / height else " "
            label = f"{mn + (mx - mn) * row_i / height:>8.0f}" if row_i in (1, height) else " " * 8
            rows.append(f"  {label} │{line}")
        print("\n  Equity curve (USDT)")
        print("\n".join(rows))
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",    default=settings.SYMBOL)
    parser.add_argument("--timeframe", default=settings.TIMEFRAME)
    parser.add_argument("--limit",     default=1000, type=int)
    args = parser.parse_args()
    run_backtest(args.symbol, args.timeframe, args.limit)
