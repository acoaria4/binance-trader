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


def get_spread_pct(exchange: ccxt.binance, symbol: str) -> tuple[float, float, float]:
    """
    Returns (spread_pct, bid, ask).
    spread_pct = (ask - bid) / mid * 100
    """
    try:
        book = exchange.fetch_order_book(symbol, limit=5)
        bid = float(book["bids"][0][0]) if book.get("bids") else 0.0
        ask = float(book["asks"][0][0]) if book.get("asks") else 0.0
    except Exception as e:
        log.warning(f"Order book fetch failed for {symbol}: {e}")
        ticker = exchange.fetch_ticker(symbol)
        last = float(ticker.get("last") or ticker.get("close") or 0.0)
        return 0.0, last, last

    mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0.0
    if mid <= 0:
        return 0.0, bid, ask
    spread_pct = (ask - bid) / mid * 100
    return spread_pct, bid, ask


def normalize_fill(order: dict, fallback_price: float = 0.0) -> dict:
    """Extract fill details from a ccxt order response."""
    filled_qty = float(order.get("filled") or order.get("amount") or 0.0)
    avg_price = float(order.get("average") or order.get("price") or fallback_price)
    if filled_qty <= 0 and order.get("cost") and avg_price > 0:
        filled_qty = float(order["cost"]) / avg_price
    return {
        "order_id": str(order.get("id", "")),
        "filled_qty": filled_qty,
        "avg_price": avg_price,
        "status": order.get("status", "unknown"),
        "cost": float(order.get("cost") or filled_qty * avg_price),
    }


def wait_for_order_fill(
    exchange: ccxt.binance,
    symbol: str,
    order_id: str,
    timeout_sec: int = 30,
    poll_interval_sec: float = 1.0,
) -> dict:
    """Poll order until closed/canceled or timeout. Returns normalize_fill-compatible dict."""
    deadline = time.time() + timeout_sec
    last = {"order_id": order_id, "filled_qty": 0.0, "avg_price": 0.0, "status": "open"}

    while time.time() < deadline:
        try:
            order = exchange.fetch_order(order_id, symbol)
            last = normalize_fill(order)
            last["status"] = order.get("status", "unknown")
            if last["status"] in ("closed", "canceled", "expired"):
                return last
            if last["filled_qty"] > 0 and last["status"] == "open":
                time.sleep(poll_interval_sec)
                continue
        except Exception as e:
            log.warning(f"fetch_order {order_id} failed: {e}")
        time.sleep(poll_interval_sec)

    try:
        order = exchange.fetch_order(order_id, symbol)
        last = normalize_fill(order)
        last["status"] = order.get("status", "timeout")
    except Exception:
        last["status"] = "timeout"
    return last


def cancel_order_safe(exchange: ccxt.binance, order_id: str, symbol: str) -> bool:
    if not order_id:
        return False
    try:
        exchange.cancel_order(order_id, symbol)
        log.info(f"Cancelled order {order_id}")
        return True
    except Exception as e:
        log.warning(f"Cancel order {order_id} failed: {e}")
        return False


def place_market_buy(exchange: ccxt.binance,
                     symbol: str,
                     amount_usdt: float) -> dict:
    """Market buy using a USDT notional."""
    ticker = exchange.fetch_ticker(symbol)
    price  = ticker["last"]
    qty    = exchange.amount_to_precision(symbol, amount_usdt / price)
    order  = exchange.create_order(
        symbol=symbol, type="market", side="buy", amount=float(qty),
    )
    log.info(f"Market BUY {qty} {symbol} @ ~{price:.2f} USDT (~{amount_usdt:.2f} USDT)")
    return order


def place_market_sell(exchange: ccxt.binance,
                      symbol: str,
                      quantity: float) -> dict:
    """Market sell using base-asset quantity."""
    ticker = exchange.fetch_ticker(symbol)
    price  = ticker["last"]
    qty    = exchange.amount_to_precision(symbol, quantity)
    order  = exchange.create_order(
        symbol=symbol, type="market", side="sell", amount=float(qty),
    )
    log.info(f"Market SELL {qty} {symbol} @ ~{price:.2f}")
    return order


def place_market_order(exchange: ccxt.binance,
                       symbol: str,
                       side: str,
                       amount_usdt: float) -> dict:
    """Backward-compatible wrapper — USDT notional for both sides."""
    if side.lower() == "buy":
        return place_market_buy(exchange, symbol, amount_usdt)
    raise ValueError(
        "place_market_order(sell) is deprecated — use place_market_sell(symbol, quantity)"
    )


def place_limit_buy(exchange: ccxt.binance,
                    symbol: str,
                    amount_usdt: float,
                    price: float) -> dict:
    """Limit buy at a specific price using USDT notional."""
    price = float(exchange.price_to_precision(symbol, price))
    qty   = float(exchange.amount_to_precision(symbol, amount_usdt / price))
    order = exchange.create_order(
        symbol=symbol, type="limit", side="buy",
        amount=qty, price=price,
    )
    log.info(f"Limit BUY {qty} {symbol} @ {price:.2f} (~{amount_usdt:.2f} USDT)")
    return order


def place_limit_sell(exchange: ccxt.binance,
                     symbol: str,
                     quantity: float,
                     price: float) -> dict:
    """Limit sell at a specific price using base-asset quantity."""
    price = float(exchange.price_to_precision(symbol, price))
    qty   = float(exchange.amount_to_precision(symbol, quantity))
    order = exchange.create_order(
        symbol=symbol, type="limit", side="sell",
        amount=qty, price=price,
    )
    log.info(f"Limit SELL {qty} {symbol} @ {price:.2f}")
    return order


def place_limit_order(exchange: ccxt.binance,
                      symbol: str,
                      side: str,
                      amount_usdt: float,
                      price: float) -> dict:
    """Backward-compatible limit order wrapper."""
    if side.lower() == "buy":
        return place_limit_buy(exchange, symbol, amount_usdt, price)
    ticker = exchange.fetch_ticker(symbol)
    qty = amount_usdt / float(ticker["last"])
    return place_limit_sell(exchange, symbol, qty, price)


def cancel_order(exchange: ccxt.binance, order_id: str, symbol: str) -> None:
    cancel_order_safe(exchange, order_id, symbol)
