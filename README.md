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
├── ev_gate.py          ← EV + conviction scoring
├── risk_sizing.py      ← Volatility sizing + ATR barriers
├── portfolio_state.py  ← Daily loss limits + consecutive-loss tracking
├── simulation.py       ← Shared trailing-stop + label simulation
├── backtest/
│   └── engine.py       ← Unified backtest engine
├── evaluation/
│   └── purged_cv.py    ← Purged walk-forward splits
├── reports/
│   └── backtest_report.py ← JSON backtest artifacts
├── risk.py             ← Position sizing + stop-loss/take-profit
├── train_lgbm.py       ← Offline training on large/multi-asset datasets
├── fetch_all_history.py← Bulk historical data download
├── diagnose.py         ← Inspect model probability distributions
├── config/
│   └── settings.py     ← All settings loaded from .env
├── utils/
│   ├── logger.py       ← Rich-enhanced logging
│   ├── trade_logger.py ← CSV trade journal
│   └── signal_logger.py← CSV signal audit log (every skip/enter)
├── models/             ← Saved model + scaler (auto-created)
├── logs/               ← Log + trade CSV (auto-created)
├── reports/            ← Backtest JSON reports (auto-created)
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
  Labels simulate full trade path with trailing stops (intrabar high/low):
    BUY  = profitable exit (TP or trailing lock-in)
    SELL = stopped out at a loss
    HOLD = no exit within forward window
  Retrain deploys only when walk-forward validation passes:
    macro-F1, validation return %, Sharpe, and composite score
        │
        ▼
Regime Filter (optional, REQUIRE_TREND=true)
  BUY only when ADX strong AND ATR healthy AND structure trending
        │
        ▼
EV + Conviction Gate (ev_gate.py)
  - Dynamic EV from ATR-derived SL/TP when USE_ATR_STOPS=true
  - P(win), EV, and conviction thresholds must all pass
        │
        ▼
Risk Manager (risk.py + risk_sizing.py + portfolio_state.py)
  - Volatility-based sizing: risk RISK_PCT_PER_TRADE of equity per trade
  - ATR multiples for SL/TP/trail (clamped to min/max %)
  - Portfolio heat cap, daily loss halt, consecutive-loss size reduction
  - Persists open positions + portfolio state to logs/
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
| `RETRAIN_MIN_F1`        | 0.25      | Min macro-F1 on validation fold              |
| `RETRAIN_MIN_VAL_RETURN_PCT` | 0.0  | Min simulated return % on validation fold    |
| `RETRAIN_MIN_VAL_SHARPE` | 0.0      | Min Sharpe on validation fold                |
| `TRAIL_ACTIVATE_PCT`    | 1.0       | Gain % before trailing stop activates        |
| `TRAIL_DISTANCE_PCT`    | 1.5       | Trail distance below high (defaults to SL)   |
| `FORWARD_CANDLES`       | 10        | Label / forward simulation horizon           |
| `FEE_PCT` / `SLIPPAGE_PCT` | 0.10 / 0.05 | Per-side costs in backtest + EV gate     |
| `EV_MIN`                | 0.005     | Min expected value (fraction) to enter       |
| `P_WIN_MIN`             | 0.55      | Min calibrated P(BUY) to enter               |
| `CONVICTION_MIN`        | 0.65      | Master conviction score threshold            |
| `REQUIRE_MTF`           | true      | Require 4h timeframe alignment for entries   |
| `ADX_THRESHOLD`         | 20.0      | Min ADX for regime filter                    |
| `MIN_ATR_RATIO`         | 0.8       | Min ATR ratio for regime filter              |
| `POSITIONS_FILE`        | logs/positions.json | Persisted open positions path        |
| `USE_RISK_SIZING`       | true      | Size from risk % instead of fixed USDT       |
| `RISK_PCT_PER_TRADE`    | 0.5       | % of equity risked per trade                 |
| `MAX_POSITION_USDT`     | 200       | Hard cap on position notional                  |
| `MAX_PORTFOLIO_HEAT_PCT`| 2.0     | Max aggregate open risk as % of equity         |
| `DAILY_LOSS_LIMIT_PCT`  | 3.0       | Halt new entries after this daily drawdown     |
| `USE_ATR_STOPS`         | true      | Derive SL/TP/trail from ATR multiples          |
| `ATR_SL_MULT` / `ATR_TP_MULT` | 1.5 / 3.0 | ATR multipliers for barriers           |

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
