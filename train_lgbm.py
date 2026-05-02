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
import lightgbm as lgb
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from exchange  import get_data_exchange, fetch_ohlcv
from features  import compute_features, FEATURE_COLS, is_trending
from config    import settings
from utils.logger import get_logger

log = get_logger("train_lgbm")

MODEL_PATH  = "models/lgbm_model.pkl"
SCALER_PATH = "models/scaler.pkl"
FORWARD_CANDLES = 10


def make_labels(df: pd.DataFrame) -> pd.Series:
    """Percentile-based labels — same as our proven strategy."""
    close = df["close"]
    scores = []
    for i in range(len(close)):
        if i + FORWARD_CANDLES >= len(close):
            scores.append(np.nan)
            continue
        fut     = close.iloc[i+1:i+1+FORWARD_CANDLES]
        max_ret = (fut.max() - close.iloc[i]) / close.iloc[i]
        min_ret = (fut.min() - close.iloc[i]) / close.iloc[i]
        scores.append(max_ret + min_ret)
    s = pd.Series(scores, index=df.index)
    buy_t  = s.quantile(0.75)
    sell_t = s.quantile(0.25)
    labels = []
    for v in scores:
        if pd.isna(v):       labels.append(1)
        elif v >= buy_t:     labels.append(2)   # BUY
        elif v <= sell_t:    labels.append(0)   # SELL
        else:                labels.append(1)   # HOLD
    return pd.Series(labels, index=df.index)


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
    """Train LightGBM on the provided dataframe."""
    log.info(f"Computing features for {len(df_train):,} candles ...")
    df = compute_features(df_train)
    df["label"] = make_labels(df)
    df.dropna(inplace=True)

    X = df[FEATURE_COLS].values
    y = df["label"].values
    y_mapped = y + 1   # -1->0, 0->1, 1->2... wait labels are already 0/1/2

    log.info(f"Label distribution: SELL={int((y==0).sum())} HOLD={int((y==1).sum())} BUY={int((y==2).sum())}")

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    model = lgb.LGBMClassifier(
        n_estimators=500,        # More trees with more data
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
        n_jobs=-1,               # Use all CPU cores
    )

    log.info("Training LightGBM ...")
    model.fit(X_sc, y)

    # Validation report on last 20% of training data
    split   = int(len(X_sc) * 0.8)
    val_pred = model.predict(X_sc[split:])
    log.info("\n" + classification_report(
        y[split:], val_pred,
        target_names=["SELL", "HOLD", "BUY"],
        zero_division=0,
    ))

    # Save model
    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH,  "wb") as f: pickle.dump(model,  f)
    with open(SCALER_PATH, "wb") as f: pickle.dump(scaler, f)
    log.info(f"Model saved to {MODEL_PATH}")

    return model, scaler


def run_backtest(model, scaler, df_test_raw: pd.DataFrame) -> None:
    """Backtest the trained model on test data."""
    df = compute_features(df_test_raw)
    df["label"] = make_labels(df)
    df.dropna(inplace=True)

    capital    = 1000.0
    equity     = [capital]
    trades     = []
    in_trade   = False
    entry_p    = 0.0
    current_sl = 0.0
    highest_p  = 0.0
    MIN_WIN    = 250
    test_start = len(df_test_raw) - len(df)

    for i, (ts, row) in enumerate(df.iterrows()):
        abs_i  = test_start + i
        window_raw = df_test_raw.iloc[max(0, abs_i - max(settings.LOOKBACK_CANDLES, MIN_WIN)):abs_i]
        if len(window_raw) < MIN_WIN:
            continue
        price = row["close"]

        # Get features for this candle
        feat = row[FEATURE_COLS].values.reshape(1, -1)
        feat_sc = scaler.transform(feat)
        proba   = model.predict_proba(feat_sc)[0]
        pred    = int(np.argmax(proba))
        conf    = float(proba[pred])
        sig_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
        signal  = sig_map[pred]

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
            elif signal == "SELL" and conf >= settings.MIN_SIGNAL_CONFIDENCE:
                reason = "SIGNAL"
            if reason:
                pnl = (price - entry_p) / entry_p
                capital *= (1 + pnl)
                equity.append(capital)
                trades.append({"pnl": pnl * 100, "reason": reason})
                in_trade = False
            continue

        # Regime filter
        wfeat = compute_features(window_raw)
        if settings.REQUIRE_TREND and not is_trending(wfeat, settings.ADX_THRESHOLD):
            continue

        if signal == "BUY" and conf >= settings.MIN_SIGNAL_CONFIDENCE:
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
    print(f"  LightGBM Backtest | {len(df)} test candles")
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
