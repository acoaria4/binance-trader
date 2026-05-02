"""
fetch_all_history.py
Fetch maximum available historical data from Binance public API.
Stores locally in data/ folder as CSV files.

Binance has data going back to:
  BTC/USDT 1h  — August 2017 (~75,000 candles)
  ETH/USDT 1h  — May 2018    (~60,000 candles)
  BNB/USDT 1h  — November 2017 (~70,000 candles)

Strategy: fetch in batches of 1000, paginate backwards from today.

Usage:
    python fetch_all_history.py --timeframe 1h
    python fetch_all_history.py --timeframe 4h
"""
import argparse
import os
import time
import pandas as pd
from datetime import datetime, timezone
from exchange import get_data_exchange
from features import compute_features
from utils.logger import get_logger

log = get_logger("fetch_all")

# How far back each asset's data goes on Binance
ASSET_START_DATES = {
    "BTC/USDT": "2017-08-17",
    "ETH/USDT": "2017-08-17",
    "BNB/USDT": "2017-11-06",
    "SOL/USDT": "2020-08-11",
    "ADA/USDT": "2018-04-17",
}

TIMEFRAME_MS = {
    "1m":  60_000,
    "5m":  300_000,
    "15m": 900_000,
    "1h":  3_600_000,
    "4h":  14_400_000,
    "1d":  86_400_000,
}


def fetch_all_candles(exchange, symbol: str, timeframe: str,
                      start_date: str = None) -> pd.DataFrame:
    """
    Fetch ALL available candles for a symbol by paginating
    from the start date to now in batches of 1000.
    """
    BATCH = 1000
    ms_per_candle = TIMEFRAME_MS.get(timeframe, 3_600_000)

    # Convert start date to milliseconds
    if start_date:
        since = int(datetime.strptime(start_date, "%Y-%m-%d")
                    .replace(tzinfo=timezone.utc).timestamp() * 1000)
    else:
        # Default: go back as far as possible (10 years)
        since = exchange.milliseconds() - (10 * 365 * 24 * 3600 * 1000)

    all_candles = []
    batch_num   = 0

    log.info(f"Fetching all {timeframe} candles for {symbol} from {start_date} ...")

    while True:
        try:
            raw = exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=since,
                limit=BATCH,
            )
        except Exception as e:
            log.error(f"Fetch error: {e}")
            time.sleep(5)
            continue

        if not raw:
            break

        all_candles.extend(raw)
        batch_num += 1
        last_ts = raw[-1][0]

        # Progress update every 10 batches
        if batch_num % 10 == 0:
            last_dt = datetime.fromtimestamp(last_ts/1000, tz=timezone.utc)
            log.info(f"  Batch {batch_num}: {len(all_candles):,} candles fetched, up to {last_dt.date()}")

        # If we got less than a full batch, we've reached the end
        if len(raw) < BATCH:
            break

        # Advance since to just after last candle
        since = last_ts + ms_per_candle

        # Respect rate limits
        time.sleep(0.2)

    if not all_candles:
        log.warning(f"No candles returned for {symbol}")
        return pd.DataFrame()

    df = pd.DataFrame(all_candles,
                      columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df = df[~df.index.duplicated(keep="last")].sort_index()

    log.info(f"Total: {len(df):,} candles for {symbol} "
             f"[{df.index[0].date()} to {df.index[-1].date()}]")
    return df


def save_with_features(df: pd.DataFrame, symbol: str, timeframe: str) -> str:
    """Compute features and save to CSV."""
    os.makedirs("data", exist_ok=True)
    clean_symbol = symbol.replace("/", "_")

    # Save raw first
    raw_path = f"data/{clean_symbol}_{timeframe}_raw.csv"
    df.to_csv(raw_path)
    log.info(f"Raw data saved: {raw_path} ({len(df):,} rows)")

    # Compute features
    log.info("Computing features ...")
    df_feat = compute_features(df)
    feat_path = f"data/{clean_symbol}_{timeframe}.csv"
    df_feat.to_csv(feat_path)
    log.info(f"Feature data saved: {feat_path} ({len(df_feat):,} rows)")

    return feat_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--symbols",   default="BTC/USDT,ETH/USDT,BNB/USDT")
    args = parser.parse_args()

    exchange = get_data_exchange()
    symbols  = [s.strip() for s in args.symbols.split(",")]

    total_candles = 0
    saved_files   = []

    for symbol in symbols:
        start = ASSET_START_DATES.get(symbol, "2018-01-01")
        df    = fetch_all_candles(exchange, symbol, args.timeframe, start)
        if df.empty:
            continue
        path = save_with_features(df, symbol, args.timeframe)
        saved_files.append(path)
        total_candles += len(df)

    print(f"\n{'='*52}")
    print(f"  Data collection complete!")
    print(f"  Total candles: {total_candles:,}")
    print(f"  Files saved:")
    for f in saved_files:
        size = os.path.getsize(f) / 1024 / 1024
        print(f"    {f} ({size:.1f} MB)")
    print(f"\n  Next: python train_tft.py --use-saved")
    print(f"{'='*52}")
