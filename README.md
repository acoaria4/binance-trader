# AI Trading Bot — Binance Testnet

An AI/ML-powered crypto trading bot using LightGBM signal classification,
connected to Binance Testnet via `ccxt`.

---

## Project Structure

```
kew_trading_bot/
├── bot.py              ← Main live trading loop
├── backtest.py         ← Walk-forward backtester
├── exchange.py         ← Binance Testnet connector (ccxt)
├── features.py         ← Technical indicator feature engineering
├── strategy.py         ← LightGBM ML strategy engine
├── risk.py             ← Position sizing + stop-loss/take-profit
├── train_lgbm.py       ← Offline training on large/multi-asset datasets
├── fetch_all_history.py← Bulk historical data download
├── diagnose.py         ← Inspect model probability distributions
├── config/
│   └── settings.py     ← All settings loaded from .env
├── utils/
│   ├── logger.py       ← Rich-enhanced logging
│   └── trade_logger.py ← CSV trade journal
├── models/             ← Saved model + scaler (auto-created)
├── logs/               ← Log + trade CSV (auto-created)
├── data/               ← Saved OHLCV/feature CSVs (auto-created)
├── .env.example        ← Copy to .env and fill in your keys
└── requirements.txt
```

---

## Quick Start

### 1. Get Binance Testnet API Keys
- Go to https://testnet.binance.vision/
- Log in with GitHub and generate API + Secret keys
- You'll get free testnet funds automatically

### 2. Install dependencies
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure
```bash
cp .env.example .env
# Edit .env — paste your API_KEY and SECRET_KEY
```

### 4. Backtest first (recommended)
```bash
python backtest.py --symbol BTC/USDT --timeframe 1h --limit 1000
```

### 5. Run the live bot
```bash
python bot.py
```

Press **Ctrl+C** to stop cleanly.

---

## How It Works

```
Public Binance OHLCV (data)          Binance Testnet (orders)
        │                                      ▲
        ▼                                      │
Feature Engineering ───────────────────────────┤
  RSI, MACD, Bollinger Bands, ATR, ADX, OBV,
  stochastics, candle structure, returns
  (37 features total — see features.FEATURE_COLS)
        │
        ▼
LightGBM Classifier
  Predicts: BUY / HOLD / SELL
  Triple-barrier labels (aligned with live SL/TP):
    BUY  = take-profit hit before stop-loss within 10 candles
    SELL = stop-loss hit before take-profit
    HOLD = neither barrier hit in time
  Retrain only deploys if walk-forward macro-F1 >= RETRAIN_MIN_F1
        │
        ▼
Regime Filter (optional, REQUIRE_TREND=true)
  BUY only when ADX strong AND ATR healthy AND structure trending
        │
        ▼
Risk Manager
  - Checks MAX_OPEN_TRADES
  - Sizes position from TRADE_AMOUNT_USDT
  - Sets stop-loss, take-profit, and trailing stop
  - Persists open positions to logs/positions.json
        │
        ▼
Order Executor
  Market orders via ccxt → Binance Testnet
        │
        ▼
Trade Logger (CSV) + console status display
```

**Data vs execution:** Historical candles are fetched from the public live
Binance API (full depth). Orders are placed on Binance Testnet only.

---

## Key Parameters (in .env)

| Parameter               | Default   | Description                                  |
|-------------------------|-----------|----------------------------------------------|
| `SYMBOL`                | BTC/USDT  | Trading pair                                 |
| `TIMEFRAME`             | 1h        | Candle interval                              |
| `TRADE_AMOUNT_USDT`     | 50        | USDT per trade                               |
| `MAX_OPEN_TRADES`       | 3         | Concurrent positions cap                     |
| `STOP_LOSS_PCT`         | 1.5       | % below entry to cut loss                    |
| `TAKE_PROFIT_PCT`       | 5.0       | % above entry to take profit                 |
| `MIN_SIGNAL_CONFIDENCE` | 0.62      | Min ML probability to act (0–1)              |
| `RETRAIN_EVERY_N`       | 50        | Retrain model every N new candles            |
| `RETRAIN_MIN_F1`        | 0.30      | Min walk-forward F1 to deploy a retrained model |
| `REQUIRE_TREND`         | true      | Skip BUY when regime filter says ranging     |
| `ADX_THRESHOLD`         | 20.0      | Min ADX for regime filter                    |
| `MIN_ATR_RATIO`         | 0.8       | Min ATR ratio for regime filter              |
| `POSITIONS_FILE`        | logs/positions.json | Persisted open positions path        |

---

## Offline Training (optional)

Download full history and train on multiple assets:

```bash
python fetch_all_history.py --timeframe 1h
python train_lgbm.py --use-saved --timeframe 1h
```

---

## Upgrading the Strategy

The strategy lives in `strategy.py`. To upgrade:
- **LSTM/Transformer**: Replace LightGBM with a PyTorch sequence model
- **More features**: Add to `features.py` → `FEATURE_COLS`
- **Regime detection**: Tune `is_trending()` in `features.py`
- **Shorts**: Enable `side='short'` in `risk.py` and use Binance Futures testnet

---

## Disclaimer

This bot is for **educational and testnet use only**.
Never run it with real funds without extensive backtesting, risk auditing,
and understanding of the financial risks involved.
