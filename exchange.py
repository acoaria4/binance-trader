"""
exchange.py
Thin wrapper around ccxt for Binance.

Two exchange instances:
  - live_exchange  : public Binance (no keys) — used for historical data only
  - trade_exchange : Binance Testnet (with keys) — used for order execution only

This gives us full historical data depth while keeping all trades on testnet.
"""
import time
import ccxt
import pandas as pd
from config import settings
from utils.logger import get_logger

log = get_logger(__name__)


def get_exchange() -> ccxt.binance:
    """
    Returns the TESTNET exchange for order execution.
    Used by bot.py for live trading.
    """
    exchange = ccxt.binance({
        "apiKey": settings.API_KEY,
        "secret": settings.SECRET_KEY,
        "enableRateLimit": True,
        "options": {
            "defaultType": "spot",
            "adjustForTimeDifference": True,
        },
    })
    exchange.set_sandbox_mode(True)
    log.info("Trade exchange: Binance Testnet (sandbox mode ON)")
    return exchange


def get_data_exchange() -> ccxt.binance:
    """
    Returns a PUBLIC Binance exchange (no API keys needed).
    Used only for fetching historical OHLCV data.
    Full historical depth available — no testnet limitations.
    """
    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {
            "defaultType": "spot",
            "adjustForTimeDifference": True,
        },
    })
    log.info("Data exchange: Binance Live (public, read-only)")
    return exchange


def fetch_ohlcv(exchange: ccxt.binance,
                symbol: str = settings.SYMBOL,
                timeframe: str = settings.TIMEFRAME,
                limit: int = settings.LOOKBACK_CANDLES) -> pd.DataFrame:
    """
    Fetch OHLCV candles with automatic pagination.
    Fetches in batches of 1000 and stitches results together.

    Pass get_data_exchange() for historical depth.
    Pass get_exchange() for live bot use.
    """
    BATCH_SIZE = 1000

    tf_ms = {
        "1m": 60_000,    "3m": 180_000,   "5m": 300_000,
        "15m": 900_000,  "30m": 1_800_000,
        "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000,
        "6h": 21_600_000,"12h": 43_200_000,
        "1d": 86_400_000,"1w": 604_800_000,
    }
    ms_per_candle = tf_ms.get(timeframe, 3_600_000)
    since = exchange.milliseconds() - (limit * ms_per_candle)

    all_candles = []
    fetched = 0

    while fetched < limit:
        batch_limit = min(BATCH_SIZE, limit - fetched)
        try:
            raw = exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=since,
                limit=batch_limit,
            )
        except Exception as e:
            log.error(f"fetch_ohlcv error: {e}")
            break

        if not raw:
            break

        all_candles.extend(raw)
        fetched += len(raw)
        since = raw[-1][0] + ms_per_candle

        if len(raw) < batch_limit:
            break

        time.sleep(exchange.rateLimit / 1000)

    if not all_candles:
        log.warning("No candles returned.")
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.DataFrame(all_candles,
                      columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df = df[~df.index.duplicated(keep="last")].sort_index()

    log.info(f"Fetched {len(df)} candles for {symbol} [{timeframe}] "
             f"from {df.index[0]} to {df.index[-1]}")
    return df


def get_balance(exchange: ccxt.binance, currency: str = "USDT") -> float:
    balance = exchange.fetch_balance()
    free = balance["free"].get(currency, 0.0)
    log.info(f"Balance: {free:.2f} {currency}")
    return free


def place_market_order(exchange: ccxt.binance,
                       symbol: str,
                       side: str,
                       amount_usdt: float) -> dict:
    ticker = exchange.fetch_ticker(symbol)
    price  = ticker["last"]
    qty    = amount_usdt / price
    qty    = exchange.amount_to_precision(symbol, qty)
    order  = exchange.create_order(
        symbol=symbol, type="market", side=side, amount=float(qty),
    )
    log.info(f"Order placed -> {side.upper()} {qty} {symbol} @ ~{price:.2f} USDT")
    return order


def place_limit_order(exchange: ccxt.binance,
                      symbol: str,
                      side: str,
                      amount_usdt: float,
                      price: float) -> dict:
    qty   = amount_usdt / price
    qty   = exchange.amount_to_precision(symbol, qty)
    order = exchange.create_order(
        symbol=symbol, type="limit", side=side,
        amount=float(qty), price=price,
    )
    log.info(f"Limit order -> {side.upper()} {qty} {symbol} @ {price:.2f} USDT")
    return order


def cancel_order(exchange: ccxt.binance, order_id: str, symbol: str) -> None:
    exchange.cancel_order(order_id, symbol)
    log.info(f"Cancelled order {order_id}")
