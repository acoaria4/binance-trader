# KEW AI Trading Bot — Binance Testnet

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
├── config/
│   └── settings.py     ← All settings loaded from .env
├── utils/
│   ├── logger.py       ← Rich-enhanced logging
│   └── trade_logger.py ← CSV trade journal
├── models/             ← Saved model + scaler (auto-created)
├── logs/               ← Log + trade CSV (auto-created)
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
Binance Testnet OHLCV
        │
        ▼
Feature Engineering
  RSI, MACD, Bollinger Bands, ATR, OBV, stochastics,
  candle structure, N-period returns (19 features total)
        │
        ▼
LightGBM Classifier
  Predicts: BUY / HOLD / SELL
  Trained on forward-looking labels:
    BUY  = price rises > take_profit% in next 5 candles
    SELL = price drops > stop_loss%  in next 5 candles
    HOLD = neither
        │
        ▼
Risk Manager
  - Checks MAX_OPEN_TRADES
  - Sizes position from TRADE_AMOUNT_USDT
  - Sets stop-loss and take-profit prices
        │
        ▼
Order Executor
  Market orders via ccxt → Binance Testnet
        │
        ▼
Trade Logger (CSV) + console status display
```

---

## Key Parameters (in .env)

| Parameter               | Default   | Description                                  |
|-------------------------|-----------|----------------------------------------------|
| `SYMBOL`                | BTC/USDT  | Trading pair                                 |
| `TIMEFRAME`             | 1h        | Candle interval                              |
| `TRADE_AMOUNT_USDT`     | 50        | USDT per trade                               |
| `MAX_OPEN_TRADES`       | 3         | Concurrent positions cap                     |
| `STOP_LOSS_PCT`         | 2.0       | % below entry to cut loss                    |
| `TAKE_PROFIT_PCT`       | 4.0       | % above entry to take profit                 |
| `MIN_SIGNAL_CONFIDENCE` | 0.6       | Min ML probability to act (0–1)              |
| `RETRAIN_EVERY_N`       | 50        | Retrain model every N new candles            |

---

## Upgrading the Strategy

The strategy lives entirely in `strategy.py`. To upgrade:
- **LSTM/Transformer**: Replace LightGBM with a PyTorch sequence model
- **More features**: Add to `features.py` → `FEATURE_COLS`
- **Regime detection**: Add a volatility-regime classifier before the signal model
- **Shorts**: Enable `side='short'` in `risk.py` and use Binance Futures testnet

---

## Disclaimer

This bot is for **educational and testnet use only**.
Never run it with real funds without extensive backtesting, risk auditing,
and understanding of the financial risks involved.
