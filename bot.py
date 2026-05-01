"""
bot.py
Main trading bot loop.

Run with:
    python bot.py

The bot will:
  1. Connect to Binance Testnet
  2. Fetch historical candles and train the ML model
  3. Every new candle close → predict signal → execute trade if confident
  4. Monitor open positions for stop-loss / take-profit exits
  5. Log every action to CSV

Press Ctrl+C to stop cleanly.
"""
import time
import sys
from datetime import datetime, timezone

from rich.console import Console
from rich.table import Table
from rich import box

from exchange      import get_exchange, fetch_ohlcv, get_balance, place_market_order
from strategy      import MLStrategy
from risk          import RiskManager
from utils.trade_logger import TradeLogger
from utils.logger  import get_logger
from config        import settings

log     = get_logger("bot")
console = Console()


def print_banner():
    console.print("""
[bold cyan]╔══════════════════════════════════════════╗
║   KEW AI Trading Bot  •  Binance Testnet ║
║   Strategy: LightGBM Signal Classifier   ║
╚══════════════════════════════════════════╝[/bold cyan]
""")


def print_status(balance: float, n_positions: int, last_signal: str,
                 last_conf: float, last_price: float):
    table = Table(box=box.SIMPLE, show_header=False)
    table.add_column("Key",   style="dim")
    table.add_column("Value", style="bold")
    table.add_row("Symbol",     settings.SYMBOL)
    table.add_row("Timeframe",  settings.TIMEFRAME)
    table.add_row("Balance",    f"{balance:.2f} USDT")
    table.add_row("Open trades",f"{n_positions} / {settings.MAX_OPEN_TRADES}")
    table.add_row("Last price", f"{last_price:.2f}")
    table.add_row("Signal",     last_signal)
    table.add_row("Confidence", f"{last_conf:.1%}")
    table.add_row("Time (UTC)", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
    console.print(table)


def run():
    print_banner()

    exchange     = get_exchange()
    strategy     = MLStrategy()
    risk         = RiskManager()
    trade_logger = TradeLogger()

    # ── Initial training ──────────────────────────────────────────────────────
    log.info("Fetching historical data for initial training …")
    df_init = fetch_ohlcv(exchange)
    if strategy.model is None:
        strategy.train(df_init)

    last_signal  = "—"
    last_conf    = 0.0
    last_candle  = None   # track last closed candle timestamp

    log.info("Bot live. Entering main loop. Press Ctrl+C to stop.")

    while True:
        try:
            df = fetch_ohlcv(exchange)
            current_price = float(df["close"].iloc[-1])
            current_ts    = df.index[-1]

            # ── Exit checks (every tick) ──────────────────────────────────────
            exits = risk.check_exits(current_price)
            for pos, reason in exits:
                pnl_pct = (current_price - pos.entry_price) / pos.entry_price * 100
                place_market_order(exchange, pos.symbol, "sell", pos.quantity * current_price)
                risk.close_position(pos.symbol)
                trade_logger.log_trade(
                    symbol=pos.symbol, action=f"SELL_{reason.upper()}",
                    price=current_price, quantity=pos.quantity,
                    reason=reason, pnl_pct=pnl_pct,
                )

            # ── New candle check ──────────────────────────────────────────────
            if last_candle is None or current_ts != last_candle:
                last_candle = current_ts
                log.info(f"New candle closed: {current_ts} @ {current_price:.2f}")

                strategy.on_new_candle(df)       # triggers retrain if due
                signal, conf = strategy.predict(df)
                last_signal, last_conf = signal, conf

                # ── BUY logic ─────────────────────────────────────────────────
                if signal == "BUY" and conf >= settings.MIN_SIGNAL_CONFIDENCE:
                    allowed, reason = risk.can_open_trade(settings.SYMBOL)
                    if allowed:
                        balance = get_balance(exchange)
                        sizing  = risk.calculate_position(current_price, balance)
                        order   = place_market_order(
                            exchange, settings.SYMBOL, "buy",
                            settings.TRADE_AMOUNT_USDT
                        )
                        risk.open_position(
                            symbol=settings.SYMBOL,
                            entry_price=current_price,
                            quantity=sizing["qty"],
                            order_id=order.get("id"),
                        )
                        trade_logger.log_trade(
                            symbol=settings.SYMBOL, action="BUY",
                            price=current_price, quantity=sizing["qty"],
                            signal_confidence=conf, reason="ml_signal",
                        )
                    else:
                        log.info(f"BUY blocked: {reason}")

                # ── SELL signal (optional long-exit) ──────────────────────────
                elif signal == "SELL" and conf >= settings.MIN_SIGNAL_CONFIDENCE:
                    for pos in list(risk.open_positions):
                        if pos.symbol == settings.SYMBOL:
                            pnl = (current_price - pos.entry_price) / pos.entry_price * 100
                            place_market_order(exchange, pos.symbol, "sell",
                                               pos.quantity * current_price)
                            risk.close_position(pos.symbol)
                            trade_logger.log_trade(
                                symbol=pos.symbol, action="SELL_SIGNAL",
                                price=current_price, quantity=pos.quantity,
                                signal_confidence=conf, reason="ml_sell_signal",
                                pnl_pct=pnl,
                            )

            # ── Status display ────────────────────────────────────────────────
            balance = get_balance(exchange)
            print_status(balance, len(risk.open_positions),
                         last_signal, last_conf, current_price)
            console.print(risk.summary())

            # Wait before next poll (30s — adapt to timeframe)
            time.sleep(30)

        except KeyboardInterrupt:
            log.info("Stopping bot … Bye!")
            sys.exit(0)
        except Exception as e:
            log.error(f"Error in main loop: {e}", exc_info=True)
            time.sleep(60)   # back-off on error


if __name__ == "__main__":
    run()
