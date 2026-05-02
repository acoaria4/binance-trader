"""
fetch_history.py
Fetch large historical datasets from Binance public API
and save to CSV for TFT training.

Usage:
    python fetch_history.py --symbol BTC/USDT --timeframe 4h --limit 10000
    python fetch_history.py --symbol ETH/USDT --timeframe 1d --limit 2000

Output: data/{symbol}_{timeframe}.csv
"""
import argparse
import os
import pandas as pd
from exchange import get_data_exchange, fetch_ohlcv
from features import compute_features
from utils.logger import get_logger

log = get_logger("fetch_history")


def fetch_and_save(symbol: str, timeframe: str, limit: int) -> str:
    os.makedirs("data", exist_ok=True)
    filename = f"data/{symbol.replace('/', '_')}_{timeframe}.csv"

    log.info(f"Fetching {limit} candles for {symbol} [{timeframe}] …")
    exchange = get_data_exchange()
    df_raw   = fetch_ohlcv(exchange, symbol=symbol, timeframe=timeframe, limit=limit)

    log.info(f"Computing features …")
    df = compute_features(df_raw)

    df.to_csv(filename)
    log.info(f"Saved {len(df)} rows → {filename}")
    log.info(f"Date range: {df.index[0]} → {df.index[-1]}")
    return filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",    default="BTC/USDT")
    parser.add_argument("--timeframe", default="4h")
    parser.add_argument("--limit",     default=10000, type=int)
    args = parser.parse_args()

    # Fetch multiple assets for richer training data
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
    for sym in symbols:
        fetch_and_save(sym, args.timeframe, args.limit)

    log.info("All done! Run: python backtest_tft.py to train and evaluate.")
