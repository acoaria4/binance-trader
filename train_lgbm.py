"""
train_lgbm.py
Train LightGBM on maximum historical data and backtest.

Uses the same saved CSV files from fetch_all_history.py
BTC: 75,000+ candles (2017-2026)
ETH: 75,000+ candles
BNB: 74,000+ candles

Usage:
    python train_lgbm.py --use-saved --timeframe 1h
    python train_lgbm.py --timeframe 1h --limit 5000
"""
import argparse
import os
import numpy as np
import pandas as pd

from exchange  import get_data_exchange, fetch_ohlcv
from features  import compute_features, FEATURE_COLS, is_trending
from simulation  import LongTradeState, check_bar_exit, pnl_at_price
from strategy  import MLStrategy
from config    import settings
from utils.logger import get_logger

log = get_logger("train_lgbm")


def load_saved_data(timeframe: str, symbols: list) -> tuple:
    dfs    = []
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
    log.info(f"Combined: {len(df_all):,} candles across {len(dfs)} assets")
    return df_all, df_btc


def train_lgbm(df_train: pd.DataFrame) -> tuple:
    """Train via MLStrategy (triple-barrier labels + walk-forward validation)."""
    log.info(f"Training on {len(df_train):,} candles via MLStrategy ...")
    strat = MLStrategy()
    strat.train(df_train, force=True)
    return strat.model, strat.scaler


def run_backtest(model, scaler, df_test_raw: pd.DataFrame) -> None:
    """Backtest the trained model on test data using shared exit simulation."""
    df_feat = compute_features(df_test_raw)
    capital = 1000.0
    equity  = [capital]
    trades  = []
    in_trade = False
    trade_state = None
    MIN_WIN = 250
    test_start = len(df_test_raw) - len(df_feat)

    for i, (ts, row) in enumerate(df_feat.iterrows()):
        abs_i = test_start + i
        window_raw = df_test_raw.iloc[
            max(0, abs_i - max(settings.LOOKBACK_CANDLES, MIN_WIN)):abs_i + 1
        ]
        if len(window_raw) < MIN_WIN:
            continue

        raw_row = df_test_raw.iloc[abs_i]
        bar_high  = float(raw_row["high"])
        bar_low   = float(raw_row["low"])
        bar_close = float(raw_row["close"])

        feat = row[FEATURE_COLS].values.reshape(1, -1)
        proba = model.predict_proba(scaler.transform(feat))[0]
        pred  = int(np.argmax(proba))
        conf  = float(proba[pred])
        signal = {0: "SELL", 1: "HOLD", 2: "BUY"}[pred]

        if in_trade and trade_state is not None:
            reason, exit_px = check_bar_exit(trade_state, bar_high, bar_low)
            if reason is None and signal == "SELL" and conf >= settings.MIN_SIGNAL_CONFIDENCE:
                reason, exit_px = "signal", bar_close
            if reason:
                pnl = pnl_at_price(trade_state.entry_price, exit_px)
                capital *= (1 + pnl)
                equity.append(capital)
                trades.append({"pnl": pnl * 100, "reason": reason})
                in_trade = False
                trade_state = None
            continue

        wfeat = compute_features(window_raw)
        if settings.REQUIRE_TREND and not is_trending(wfeat, settings.ADX_THRESHOLD):
            continue

        if signal == "BUY" and conf >= settings.MIN_SIGNAL_CONFIDENCE:
            in_trade = True
            trade_state = LongTradeState.from_entry(bar_close)
            trades.append({"buy": bar_close, "conf": conf})

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
    print(f"  LightGBM Backtest | {len(df_feat)} test candles")
    print(f"  Settings: SL={settings.STOP_LOSS_PCT}% | TP={settings.TAKE_PROFIT_PCT}% | MinConf={settings.MIN_SIGNAL_CONFIDENCE:.0%}")
    print(f"{'─'*52}")
    print(f"  Trades:       {len(closed)}")
    print(f"  Win rate:     {len(wins)/len(closed)*100:.1f}%" if closed else "  Win rate:     N/A")
    print(f"  Total return: {total:+.2f}%")
    print(f"  Max drawdown: -{mxdd:.2f}%")
    print(f"  Sharpe:       {sharpe:.2f}")
    print(f"  Final equity: {capital:.2f} USDT")
    print(f"{'─'*52}")

    # Feature importance
    print(f"\n  Top 10 most important features:")
    importance = dict(zip(FEATURE_COLS, model.feature_importances_))
    for feat, imp in sorted(importance.items(), key=lambda x: -x[1])[:10]:
        print(f"    {feat:25} {imp:.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",    default=settings.SYMBOL)
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--limit",     default=5000, type=int)
    parser.add_argument("--use-saved", action="store_true")
    parser.add_argument("--symbols",   default="BTC/USDT,ETH/USDT,BNB/USDT")
    args = parser.parse_args()

    if args.use_saved:
        symbols          = [s.strip() for s in args.symbols.split(",")]
        df_all, df_btc   = load_saved_data(args.timeframe, symbols)
        split            = int(len(df_btc) * 0.7)
        df_train         = df_all.iloc[:int(len(df_all) * 0.7)]
        df_test_raw      = df_btc.iloc[split:]
    else:
        exchange    = get_data_exchange()
        df_raw      = fetch_ohlcv(exchange, symbol=args.symbol,
                                  timeframe=args.timeframe, limit=args.limit)
        df_eth      = fetch_ohlcv(exchange, symbol="ETH/USDT",
                                  timeframe=args.timeframe, limit=args.limit)
        df_bnb      = fetch_ohlcv(exchange, symbol="BNB/USDT",
                                  timeframe=args.timeframe, limit=args.limit)
        df_train    = pd.concat([df_raw, df_eth, df_bnb])
        df_test_raw = df_raw.iloc[int(len(df_raw)*0.7):]

    # Train
    model, scaler = train_lgbm(df_train)

    # Backtest on BTC test set
    log.info("Running backtest on BTC test set ...")
    if args.use_saved:
        run_backtest(model, scaler, df_btc.iloc[int(len(df_btc)*0.7):])
    else:
        run_backtest(model, scaler, df_test_raw)
