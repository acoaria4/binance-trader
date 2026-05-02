"""
backtest_tft.py
Train and backtest the TFT strategy on historical data.

Usage:
    # First fetch data:
    python fetch_history.py --timeframe 4h --limit 10000

    # Then train and backtest:
    python backtest_tft.py --timeframe 4h

The TFT is trained on 70% of data and tested on remaining 30%.
Results are compared against LightGBM baseline.
"""
import argparse
import os
import numpy as np
import pandas as pd

from exchange   import get_data_exchange, fetch_ohlcv
from features   import compute_features, FEATURE_COLS, is_trending
from strategy   import MLStrategy          # LightGBM baseline
from strategy_tft import TFTStrategy, prepare_tft_dataframe
from config     import settings
from utils.logger import get_logger

log = get_logger("backtest_tft")


def run_backtest(strategy, df_raw: pd.DataFrame,
                 label: str = "Strategy") -> dict:
    """
    Generic walk-forward backtest — works with any strategy
    that implements predict(df_raw) -> (signal, confidence).
    """
    df       = compute_features(df_raw)
    split    = int(len(df) * 0.7)
    df_train = df.iloc[:split]
    df_test  = df.iloc[split:]

    log.info(f"[{label}] Training on {len(df_train)} candles …")
    strategy.train(df_train)

    capital      = 1000.0
    equity       = [capital]
    trades       = []
    in_trade     = False
    entry_p      = 0.0
    current_sl   = 0.0
    highest_price = 0.0
    MIN_WINDOW   = 250
    test_start   = len(df_raw) - len(df_test)

    for i, (ts, row) in enumerate(df_test.iterrows()):
        abs_i  = test_start + i
        window = df_raw.iloc[max(0, abs_i - max(settings.LOOKBACK_CANDLES, MIN_WINDOW)):abs_i]
        if len(window) < MIN_WINDOW:
            continue

        price = row["close"]

        if in_trade:
            if price > highest_price:
                highest_price = price
            gain = (highest_price - entry_p) / entry_p * 100
            if gain >= 1.0:
                trail = highest_price * (1 - settings.STOP_LOSS_PCT / 100)
                if trail > current_sl:
                    current_sl = trail

            tp     = entry_p * (1 + settings.TAKE_PROFIT_PCT / 100)
            reason = None
            if price >= tp:           reason = "TP"
            elif price <= current_sl: reason = "SL"
            else:
                sig, conf = strategy.predict(window)
                if sig == "SELL" and conf >= settings.MIN_SIGNAL_CONFIDENCE:
                    reason = "SIGNAL"

            if reason:
                pnl     = (price - entry_p) / entry_p
                capital *= (1 + pnl)
                equity.append(capital)
                trades.append({"action": f"SELL_{reason}", "pnl": pnl * 100})
                in_trade = False
            continue

        # Regime filter
        wfeat = compute_features(window)
        if settings.REQUIRE_TREND and not is_trending(wfeat, settings.ADX_THRESHOLD):
            continue

        sig, conf = strategy.predict(window)
        if sig != "BUY" or conf < settings.MIN_SIGNAL_CONFIDENCE:
            continue

        entry_p       = price
        current_sl    = price * (1 - settings.STOP_LOSS_PCT / 100)
        highest_price = price
        in_trade      = True
        trades.append({"action": "BUY", "price": price})

    closed = [t for t in trades if "pnl" in t]
    wins   = [t for t in closed if t["pnl"] > 0]
    rets   = pd.Series([t["pnl"] / 100 for t in closed])
    sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if len(rets) > 1 and rets.std() > 0 else 0
    total  = (capital - 1000) / 1000 * 100
    mxdd   = 0.0
    if equity:
        peak = equity[0]
        for v in equity:
            if v > peak: peak = v
            dd = (peak - v) / peak * 100
            if dd > mxdd: mxdd = dd

    result = dict(
        label=label,
        trades=len(closed),
        win_rate=len(wins)/len(closed)*100 if closed else 0,
        total_return=total,
        max_drawdown=mxdd,
        sharpe=sharpe,
        final_equity=capital,
    )

    print(f"\n{'─'*52}")
    print(f"  {label}")
    print(f"{'─'*52}")
    print(f"  Trades:       {result['trades']}")
    print(f"  Win rate:     {result['win_rate']:.1f}%")
    print(f"  Total return: {result['total_return']:+.2f}%")
    print(f"  Max drawdown: -{result['max_drawdown']:.2f}%")
    print(f"  Sharpe:       {result['sharpe']:.2f}")
    print(f"  Final equity: {result['final_equity']:.2f} USDT")
    print(f"{'─'*52}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",    default=settings.SYMBOL)
    parser.add_argument("--timeframe", default="4h")
    parser.add_argument("--limit",     default=10000, type=int)
    parser.add_argument("--tft-only",  action="store_true")
    args = parser.parse_args()

    log.info(f"Fetching {args.limit} candles [{args.timeframe}] …")
    exchange = get_data_exchange()
    df_raw   = fetch_ohlcv(
        exchange, symbol=args.symbol,
        timeframe=args.timeframe, limit=args.limit
    )
    log.info(f"Got {len(df_raw)} candles: {df_raw.index[0]} → {df_raw.index[-1]}")

    results = []

    # LightGBM baseline
    if not args.tft_only:
        lgbm = MLStrategy()
        r1   = run_backtest(lgbm, df_raw.copy(), label="LightGBM (baseline)")
        results.append(r1)

    # TFT
    try:
        tft = TFTStrategy()
        r2  = run_backtest(tft, df_raw.copy(), label="TFT (new)")
        results.append(r2)
    except Exception as e:
        log.error(f"TFT failed: {e}")
        log.info("Install dependencies: pip install pytorch-forecasting pytorch-lightning torch")

    # Comparison
    if len(results) == 2:
        print("\n── Head-to-head comparison ────────────────────────")
        metrics = ["trades", "win_rate", "total_return", "max_drawdown", "sharpe"]
        for m in metrics:
            v1, v2 = results[0][m], results[1][m]
            winner = "TFT ✓" if (
                (m in ["win_rate", "total_return", "sharpe", "trades"] and v2 > v1) or
                (m == "max_drawdown" and v2 < v1)
            ) else "LightGBM ✓"
            print(f"  {m:15} LightGBM={v1:.2f}  TFT={v2:.2f}  → {winner}")
        print()
