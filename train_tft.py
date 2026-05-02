"""
train_tft.py
Train TFT on maximum available historical data.

Two modes:
  1. Fetch fresh + train:
     python train_tft.py --timeframe 1h

  2. Use pre-fetched saved data (recommended after fetch_all_history.py):
     python train_tft.py --use-saved --timeframe 1h

The --use-saved mode loads all CSV files from data/ folder,
concatenates them, and trains on the full dataset.
"""
import argparse
import os
import numpy as np
import pandas as pd

from exchange  import get_data_exchange, fetch_ohlcv
from features  import compute_features, is_trending
from strategy  import TFTStrategy
from config    import settings
from utils.logger import get_logger

log = get_logger("train_tft")


def load_saved_data(timeframe: str, symbols: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load pre-fetched CSVs from data/ folder.
    Returns (df_train_combined, df_test_btc)
    """
    dfs = []
    df_btc = None

    for sym in symbols:
        clean = sym.replace("/", "_")
        path  = f"data/{clean}_{timeframe}.csv"
        if not os.path.exists(path):
            log.warning(f"Missing: {path} — run fetch_all_history.py first")
            continue
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        log.info(f"Loaded {len(df):,} rows from {path}")
        dfs.append(df)
        if sym == "BTC/USDT":
            df_btc = df

    if not dfs:
        raise FileNotFoundError("No data files found. Run: python fetch_all_history.py")

    df_all = pd.concat(dfs)
    log.info(f"Combined dataset: {len(df_all):,} candles across {len(dfs)} assets")
    return df_all, df_btc


def run_backtest(tft: TFTStrategy, df_raw: pd.DataFrame) -> None:
    df       = compute_features(df_raw)
    split    = int(len(df) * 0.7)
    df_test  = df.iloc[split:]

    capital    = 1000.0
    equity     = [capital]
    trades     = []
    in_trade   = False
    entry_p    = 0.0
    current_sl = 0.0
    highest_p  = 0.0
    MIN_WIN    = 100
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
                if sig == "SELL" and conf >= 0.40:
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
        if sig == "BUY" and conf >= 0.40:
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
    peak   = equity[0] if equity else 1000
    for v in equity:
        if v > peak: peak = v
        dd = (peak - v) / peak * 100
        if dd > mxdd: mxdd = dd

    print(f"\n{'─'*52}")
    print(f"  TFT Backtest | {len(df_test)} test candles")
    print(f"{'─'*52}")
    print(f"  Trades:       {len(closed)}")
    print(f"  Win rate:     {len(wins)/len(closed)*100:.1f}%" if closed else "  Win rate:     N/A")
    print(f"  Total return: {total:+.2f}%")
    print(f"  Max drawdown: -{mxdd:.2f}%")
    print(f"  Sharpe:       {sharpe:.2f}")
    print(f"  Final equity: {capital:.2f} USDT")
    print(f"{'─'*52}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",     default=settings.SYMBOL)
    parser.add_argument("--timeframe",  default="1h")
    parser.add_argument("--limit",      default=5000, type=int)
    parser.add_argument("--use-saved",  action="store_true",
                        help="Load pre-fetched data from data/ folder")
    parser.add_argument("--symbols",    default="BTC/USDT,ETH/USDT,BNB/USDT")
    args = parser.parse_args()

    if args.use_saved:
        symbols = [s.strip() for s in args.symbols.split(",")]
        df_train, df_btc = load_saved_data(args.timeframe, symbols)
        df_test_raw = df_btc  # Backtest on BTC only
    else:
        log.info(f"Fetching {args.limit} {args.timeframe} candles ...")
        exchange    = get_data_exchange()
        df_raw      = fetch_ohlcv(exchange, symbol=args.symbol,
                                  timeframe=args.timeframe, limit=args.limit)
        log.info(f"Fetching ETH and BNB ...")
        df_eth      = fetch_ohlcv(exchange, symbol="ETH/USDT",
                                  timeframe=args.timeframe, limit=args.limit)
        df_bnb      = fetch_ohlcv(exchange, symbol="BNB/USDT",
                                  timeframe=args.timeframe, limit=args.limit)
        df_train    = pd.concat([df_raw, df_eth, df_bnb])
        df_test_raw = df_raw
        log.info(f"Combined: {len(df_train):,} candles")

    # Train TFT
    tft = TFTStrategy()
    split = int(len(df_train) * 0.7)
    tft.train(df_train.iloc[:split])

    # Backtest on BTC test set
    log.info("Running backtest ...")
    split_btc = int(len(df_test_raw) * 0.7)
    run_backtest(tft, df_test_raw)
