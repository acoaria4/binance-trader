"""
exchange.py
Thin wrapper around ccxt for Binance Testnet.
All exchange calls go through this module.
"""
import ccxt
import pandas as pd
from config import settings
from utils.logger import get_logger

log = get_logger(__name__)


def get_exchange() -> ccxt.binance:
    """Return an authenticated Binance Testnet exchange instance."""
    exchange = ccxt.binance({
        "apiKey": settings.API_KEY,
        "secret": settings.SECRET_KEY,
        "enableRateLimit": True,
        "options": {
            "defaultType": "spot",
            "adjustForTimeDifference": True,
        },
    })
    # Point to testnet endpoints
    exchange.set_sandbox_mode(True)
    log.info("Exchange: Binance Testnet (sandbox mode ON)")
    return exchange


def fetch_ohlcv(exchange: ccxt.binance,
                symbol: str = settings.SYMBOL,
                timeframe: str = settings.TIMEFRAME,
                limit: int = settings.LOOKBACK_CANDLES) -> pd.DataFrame:
    """
    Fetch OHLCV candles and return a tidy DataFrame.

    Columns: timestamp, open, high, low, close, volume
    """
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    log.debug(f"Fetched {len(df)} candles for {symbol} [{timeframe}]")
    return df


def get_balance(exchange: ccxt.binance, currency: str = "USDT") -> float:
    """Return free balance for the given currency."""
    balance = exchange.fetch_balance()
    free = balance["free"].get(currency, 0.0)
    log.info(f"Balance: {free:.2f} {currency}")
    return free


def place_market_order(exchange: ccxt.binance,
                       symbol: str,
                       side: str,
                       amount_usdt: float) -> dict:
    """
    Place a market order.

    side: 'buy' | 'sell'
    amount_usdt: value in USDT to spend (converted to base qty internally)
    """
    ticker = exchange.fetch_ticker(symbol)
    price  = ticker["last"]
    qty    = amount_usdt / price

    # Round to exchange precision
    qty = exchange.amount_to_precision(symbol, qty)

    order = exchange.create_order(
        symbol=symbol,
        type="market",
        side=side,
        amount=float(qty),
    )
    log.info(f"Order placed → {side.upper()} {qty} {symbol} @ ~{price:.2f} USDT")
    return order


def place_limit_order(exchange: ccxt.binance,
                      symbol: str,
                      side: str,
                      amount_usdt: float,
                      price: float) -> dict:
    """Place a limit order at the specified price."""
    qty = amount_usdt / price
    qty = exchange.amount_to_precision(symbol, qty)

    order = exchange.create_order(
        symbol=symbol,
        type="limit",
        side=side,
        amount=float(qty),
        price=price,
    )
    log.info(f"Limit order → {side.upper()} {qty} {symbol} @ {price:.2f} USDT")
    return order


def cancel_order(exchange: ccxt.binance, order_id: str, symbol: str) -> None:
    """Cancel an open order."""
    exchange.cancel_order(order_id, symbol)
    log.info(f"Cancelled order {order_id}")
