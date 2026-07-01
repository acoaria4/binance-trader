"""
bot.py
Main trading bot loop.

Run with:
    python bot.py

Press Ctrl+C to stop cleanly.
"""
import time
import sys
from datetime import datetime, timezone

from rich.console import Console
from rich.table import Table
from rich import box

from exchange import (
    get_exchange, get_data_exchange, fetch_ohlcv,
    get_balance, place_market_buy, place_market_sell,
)
from strategy           import MLStrategy
from risk               import RiskManager
from utils.trade_logger import TradeLogger
from utils.logger       import get_logger
from features           import compute_features, is_trending
from config             import settings

log     = get_logger("bot")
console = Console()

DATA_FETCH_LIMIT  = 300
TRAIN_FETCH_LIMIT = 2000


def print_banner():
    console.print("""
[bold cyan]╔══════════════════════════════════════════╗
║   KEW AI Trading Bot  •  Binance Testnet ║
║   Strategy: LightGBM Triple-Barrier      ║
║   v3: Validated retrain + persistence    ║
╚══════════════════════════════════════════╝[/bold cyan]
""")


def print_status(balance: float, n_positions: int, last_signal: str,
                 last_conf: float, last_price: float):
    table = Table(box=box.SIMPLE, show_header=False)
    table.add_column("Key",   style="dim")
    table.add_column("Value", style="bold")
    table.add_row("Symbol",      settings.SYMBOL)
    table.add_row("Timeframe",   settings.TIMEFRAME)
    table.add_row("Balance",     f"{balance:.2f} USDT")
    table.add_row("Open trades", f"{n_positions} / {settings.MAX_OPEN_TRADES}")
    table.add_row("Last price",  f"{last_price:.2f}")
    table.add_row("Signal",      last_signal)
    table.add_row("Confidence",  f"{last_conf:.1%}")
    table.add_row("Time (UTC)",  datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
    console.print(table)


def run():
    print_banner()

    trade_exchange = get_exchange()
    data_exchange  = get_data_exchange()
    strategy       = MLStrategy()
    risk           = RiskManager()
    trade_logger   = TradeLogger()

    risk.reconcile(trade_exchange, settings.SYMBOL)

    log.info("Fetching historical data for initial training …")
    df_init = fetch_ohlcv(data_exchange, limit=TRAIN_FETCH_LIMIT)
    if strategy.model is None:
        strategy.train(df_init, force=True)

    last_signal = "—"
    last_conf   = 0.0
    last_candle = None

    log.info("Bot live. Entering main loop. Press Ctrl+C to stop.")

    while True:
        try:
            df            = fetch_ohlcv(data_exchange, limit=DATA_FETCH_LIMIT)
            current_price = float(df["close"].iloc[-1])
            current_ts    = df.index[-1]

            exits = risk.check_exits(current_price)
            for pos, reason in exits:
                pnl_pct = pos.pnl_pct(current_price)
                place_market_sell(trade_exchange, pos.symbol, pos.quantity)
                risk.close_position(pos.symbol)
                trade_logger.log_trade(
                    symbol=pos.symbol, action=f"SELL_{reason.upper()}",
                    price=current_price, quantity=pos.quantity,
                    reason=reason, pnl_pct=pnl_pct,
                )
                log.info(f"Exit: {reason.upper()} | PnL: {pnl_pct:+.2f}%")

            if last_candle is None or current_ts != last_candle:
                last_candle = current_ts
                log.info(f"New candle: {current_ts} @ {current_price:.2f}")

                df_train = fetch_ohlcv(data_exchange, limit=TRAIN_FETCH_LIMIT)
                strategy.on_new_candle(df_train)

                signal, conf = strategy.predict(df)
                last_signal, last_conf = signal, conf

                df_feat  = compute_features(df)
                trending = not settings.REQUIRE_TREND or \
                           is_trending(df_feat, settings.ADX_THRESHOLD)

                if signal == "SELL" and conf >= settings.MIN_SIGNAL_CONFIDENCE:
                    symbol_positions = [
                        p for p in risk.open_positions if p.symbol == settings.SYMBOL
                    ]
                    if not symbol_positions:
                        log.info(
                            "SELL signal (long-only): no open position — standing aside"
                        )
                    for pos in symbol_positions:
                        pnl = pos.pnl_pct(current_price)
                        place_market_sell(trade_exchange, pos.symbol, pos.quantity)
                        risk.close_position(pos.symbol)
                        trade_logger.log_trade(
                            symbol=pos.symbol, action="SELL_SIGNAL",
                            price=current_price, quantity=pos.quantity,
                            signal_confidence=conf,
                            reason="ml_sell_signal", pnl_pct=pnl,
                        )
                        log.info(f"SELL signal exit | PnL: {pnl:+.2f}%")

                elif signal == "BUY" and conf >= settings.MIN_SIGNAL_CONFIDENCE:
                    if not trending:
                        log.info("Regime filter: not trending — BUY skipped")
                    else:
                        allowed, block_reason = risk.can_open_trade(settings.SYMBOL)
                        if allowed:
                            balance = get_balance(trade_exchange)
                            sizing  = risk.calculate_position(current_price, balance)
                            order   = place_market_buy(
                                trade_exchange, settings.SYMBOL,
                                settings.TRADE_AMOUNT_USDT,
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
                            log.info(f"BUY blocked: {block_reason}")

            balance = get_balance(trade_exchange)
            print_status(balance, len(risk.open_positions),
                         last_signal, last_conf, current_price)
            console.print(risk.summary())

            time.sleep(30)

        except KeyboardInterrupt:
            log.info("Stopping bot … Bye!")
            sys.exit(0)
        except Exception as e:
            log.error(f"Error in main loop: {e}", exc_info=True)
            time.sleep(60)


if __name__ == "__main__":
    run()
