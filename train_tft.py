"""
train_tft.py
Standalone TFT training and backtest.
Run this independently of backtest.py.

Usage:
    python train_tft.py --timeframe 4h --limit 10000
"""
import argparse
import numpy as np
import pandas as pd

from exchange  import get_data_exchange, fetch_ohlcv
from features  import compute_features, is_trending
from strategy  import TFTStrategy
from config    import settings
from utils.logger import get_logger

log = get_logger("train_tft")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",    default=settings.SYMBOL)
    parser.add_argument("--timeframe", default="4h")
    parser.add_argument("--limit",     default=10000, type=int)
    args = parser.parse_args()

    # Fetch data
    log.info(f"Fetching {args.limit} {args.timeframe} candles ...")
    exchange = get_data_exchange()
    df_raw   = fetch_ohlcv(exchange, symbol=args.symbol,
                           timeframe=args.timeframe, limit=args.limit)
    log.info(f"Got {len(df_raw)} candles: {df_raw.index[0]} to {df_raw.index[-1]}")

    # Train on 70%
    split    = int(len(df_raw) * 0.7)
    df_train = df_raw.iloc[:split]
    df_test  = df_raw.iloc[split:]

    tft = TFTStrategy()
    tft.train(df_train)

    # Backtest on remaining 30%
    log.info(f"Backtesting on {len(df_test)} candles ...")
    df       = compute_features(df_raw)
    df_test  = df.iloc[split:]

    capital    = 1000.0
    equity     = [capital]
    trades     = []
    in_trade   = False
    entry_p    = 0.0
    current_sl = 0.0
    highest_p  = 0.0
    MIN_WIN    = 100   # TFT needs SEQ_LEN=60 + buffer
    test_start = len(df_raw) - len(df_test)

    for i, (ts, row) in enumerate(df_test.iterrows()):
        abs_i  = test_start + i
        window = df_raw.iloc[max(0, abs_i - 200):abs_i]
        if len(window) < MIN_WIN:
            continue
        price = row["close"]

        if in_trade:
            if price > highest_p: highest_p = price
            gain = (highest_p - entry_p) / entry_p * 100
            if gain >= 1.0:
                trail = highest_p * (1 - settings.STOP_LOSS_PCT / 100)
                if trail > current_sl: current_sl = trail
            tp = entry_p * (1 + settings.TAKE_PROFIT_PCT / 100)
            reason = None
            if price >= tp:            reason = "TP"
            elif price <= current_sl:  reason = "SL"
            else:
                sig, conf = tft.predict(window)
                if sig == "SELL" and conf >= 0.40:  # Lower threshold for TFT
                    reason = "SIGNAL"
            if reason:
                pnl = (price - entry_p) / entry_p
                capital *= (1 + pnl)
                equity.append(capital)
                trades.append({"pnl": pnl * 100, "reason": reason})
                in_trade = False
            continue

        wfeat = compute_features(window)
        if settings.REQUIRE_TREND and not is_trending(wfeat, settings.ADX_THRESHOLD):
            continue
        sig, conf = tft.predict(window)
        if sig == "BUY" and conf >= 0.40:  # Lower threshold for TFT
            entry_p    = price
            current_sl = price * (1 - settings.STOP_LOSS_PCT / 100)
            highest_p  = price
            in_trade   = True
            trades.append({"buy": price, "conf": conf})

    closed = [t for t in trades if "pnl" in t]
    wins   = [t for t in closed if t["pnl"] > 0]
    total  = (capital - 1000) / 1000 * 100
    rets   = pd.Series([t["pnl"]/100 for t in closed])
    sharpe = (rets.mean()/rets.std()*np.sqrt(252)) if len(rets)>1 and rets.std()>0 else 0
    mxdd   = 0.0
    peak   = equity[0]
    for v in equity:
        if v > peak: peak = v
        dd = (peak - v) / peak * 100
        if dd > mxdd: mxdd = dd

    print(f"\n{'─'*52}")
    print(f"  TFT Backtest: {args.symbol} | {args.timeframe} | {len(df_test)} candles")
    print(f"{'─'*52}")
    print(f"  Trades:       {len(closed)}")
    print(f"  Win rate:     {len(wins)/len(closed)*100:.1f}%" if closed else "  Win rate: N/A")
    print(f"  Total return: {total:+.2f}%")
    print(f"  Max drawdown: -{mxdd:.2f}%")
    print(f"  Sharpe:       {sharpe:.2f}")
    print(f"  Final equity: {capital:.2f} USDT")
    print(f"{'─'*52}")
