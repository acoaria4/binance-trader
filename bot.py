"""
bot.py
Main trading bot loop with EV gating, volatility sizing, portfolio risk,
and Phase 3 execution fidelity (limit orders, spread checks, reconciliation).
"""
import time
import sys
from datetime import datetime, timezone

from rich.console import Console
from rich.table import Table
from rich import box

from exchange import get_exchange, get_data_exchange, fetch_ohlcv, get_balance
from execution import execute_buy, execute_sell
from reconciliation import reconcile_positions, save_reconciliation_log
from strategy           import MLStrategy
from risk               import RiskManager
from features           import compute_features
from utils.trade_logger import TradeLogger
from utils.signal_logger import SignalLogger
from utils.logger       import get_logger
from config             import settings

log     = get_logger("bot")
console = Console()

DATA_FETCH_LIMIT  = 300
TRAIN_FETCH_LIMIT = 2000


def print_banner():
    console.print("""
[bold cyan]╔══════════════════════════════════════════╗
║   KEW AI Trading Bot  •  Binance Testnet ║
║   v7: Limit Orders + Spread + Reconcile    ║
╚══════════════════════════════════════════╝[/bold cyan]
""")


def print_status(balance: float, risk: RiskManager, evaluation, last_price: float):
    table = Table(box=box.SIMPLE, show_header=False)
    table.add_column("Key",   style="dim")
    table.add_column("Value", style="bold")
    table.add_row("Symbol",      settings.SYMBOL)
    table.add_row("Timeframe",   settings.TIMEFRAME)
    table.add_row("Balance",     f"{balance:.2f} USDT")
    table.add_row("Open trades", f"{len(risk.open_positions)} / {settings.MAX_OPEN_TRADES}")
    table.add_row("Last price",  f"{last_price:.2f}")
    table.add_row("Execution",   f"limit={settings.USE_LIMIT_ORDERS} max_spread={settings.MAX_SPREAD_PCT}%")
    if evaluation:
        table.add_row("Signal",     evaluation.signal)
        table.add_row("P(win)",     f"{evaluation.p_win:.1%}")
        table.add_row("EV",         f"{evaluation.ev:.4f}")
        table.add_row("Conviction", f"{evaluation.conviction:.2f}")
    table.add_row("Portfolio",   risk.portfolio_summary(balance))
    table.add_row("Time (UTC)",  datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
    console.print(table)


def _get_atr(df) -> float:
    feat = compute_features(df)
    if feat.empty or "atr_14" not in feat.columns:
        return None
    return float(feat["atr_14"].iloc[-1])


def _close_position(
    trade_exchange, risk, trade_logger, pos, current_price, balance,
    action: str, reason: str, signal_confidence: float = 0.0,
    urgent: bool = True,
):
    """Execute sell and update risk/trade logs."""
    result = execute_sell(trade_exchange, pos.symbol, pos.quantity, urgent=urgent)
    if not result.success:
        log.error(f"Sell failed for {pos.symbol}: {result.reason}")
        return False

    exit_price = result.avg_price or current_price
    pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * 100
    risk.close_position(pos.symbol)
    risk.record_closed_trade(pnl_pct, balance)
    trade_logger.log_trade(
        symbol=pos.symbol, action=action,
        price=exit_price, quantity=result.filled_qty,
        signal_confidence=signal_confidence,
        reason=f"{reason}|{result.order_type}", pnl_pct=pnl_pct,
    )
    log.info(
        f"Exit: {reason} | {result.order_type} | "
        f"fill={result.filled_qty:.6f} @ {exit_price:.2f} | PnL: {pnl_pct:+.2f}%"
    )
    return True


def run():
    print_banner()

    trade_exchange = get_exchange()
    data_exchange  = get_data_exchange()
    strategy       = MLStrategy()
    risk           = RiskManager()
    trade_logger   = TradeLogger()
    signal_logger  = SignalLogger()
    loop_count     = 0

    if settings.RECONCILE_ON_START:
        report = reconcile_positions(trade_exchange, risk, settings.SYMBOL)
        save_reconciliation_log(report)

    log.info("Fetching historical data for initial training …")
    df_init = fetch_ohlcv(data_exchange, limit=TRAIN_FETCH_LIMIT)
    if strategy.model is None:
        strategy.train(df_init, force=True)

    last_evaluation = None
    last_candle = None

    log.info("Bot live. Entering main loop. Press Ctrl+C to stop.")

    while True:
        try:
            loop_count += 1
            df            = fetch_ohlcv(data_exchange, limit=DATA_FETCH_LIMIT)
            bar_high      = float(df["high"].iloc[-1])
            bar_low       = float(df["low"].iloc[-1])
            current_price = float(df["close"].iloc[-1])
            current_ts    = df.index[-1]
            atr           = _get_atr(df)

            if loop_count % settings.RECONCILE_EVERY_N_LOOPS == 0:
                report = reconcile_positions(trade_exchange, risk, settings.SYMBOL)
                save_reconciliation_log(report)

            exits = risk.check_exits(bar_high, bar_low)
            balance = get_balance(trade_exchange)
            for pos, reason in exits:
                _close_position(
                    trade_exchange, risk, trade_logger, pos, current_price, balance,
                    action=f"SELL_{reason.upper()}", reason=reason, urgent=True,
                )

            if last_candle is None or current_ts != last_candle:
                last_candle = current_ts
                log.info(f"New candle: {current_ts} @ {current_price:.2f}")

                df_train = fetch_ohlcv(data_exchange, limit=TRAIN_FETCH_LIMIT)
                strategy.on_new_candle(df_train)

                evaluation = strategy.evaluate(df, atr=atr)
                last_evaluation = evaluation
                signal_logger.log(settings.SYMBOL, evaluation, "EVAL")

                if evaluation.should_exit:
                    symbol_positions = [
                        p for p in risk.open_positions if p.symbol == settings.SYMBOL
                    ]
                    if not symbol_positions:
                        log.info("SELL signal (long-only): no open position — standing aside")
                        signal_logger.log(settings.SYMBOL, evaluation, "SKIP_EXIT_FLAT")
                    for pos in symbol_positions:
                        if _close_position(
                            trade_exchange, risk, trade_logger, pos, current_price, balance,
                            action="SELL_SIGNAL", reason="ml_sell_signal",
                            signal_confidence=evaluation.p_buy, urgent=False,
                        ):
                            signal_logger.log(settings.SYMBOL, evaluation, "EXIT")

                elif evaluation.should_enter:
                    sizing = risk.calculate_position(current_price, balance, atr=atr)
                    allowed, block_reason = risk.can_open_trade(
                        settings.SYMBOL, balance, sizing["risk_usdt"],
                    )
                    if allowed and sizing["trade_usdt"] > 0:
                        result = execute_buy(
                            trade_exchange, settings.SYMBOL, sizing["trade_usdt"],
                        )
                        if result.success and result.filled_qty > 0:
                            entry_price = result.avg_price or current_price
                            risk.open_position(
                                symbol=settings.SYMBOL,
                                entry_price=entry_price,
                                quantity=result.filled_qty,
                                stop_loss=sizing["stop_loss"],
                                take_profit=sizing["take_profit"],
                                trail_distance_pct=sizing["trail_distance_pct"],
                                risk_usdt=sizing["risk_usdt"],
                                order_id=result.order_id,
                            )
                            trade_logger.log_trade(
                                symbol=settings.SYMBOL, action="BUY",
                                price=entry_price, quantity=result.filled_qty,
                                signal_confidence=evaluation.p_win,
                                reason=f"ev_conviction|{result.order_type}|spread={result.spread_pct:.3f}%",
                            )
                            signal_logger.log(settings.SYMBOL, evaluation, "ENTER")
                            log.info(
                                f"BUY {result.notional_usdt:.2f} USDT via {result.order_type} | "
                                f"fill={result.filled_qty:.6f} @ {entry_price:.2f} | "
                                f"P(win)={evaluation.p_win:.1%} EV={evaluation.ev:.4f} | "
                                f"spread={result.spread_pct:.3f}%"
                            )
                        else:
                            skip = result.reason or "execution_failed"
                            log.info(f"BUY blocked: {skip}")
                            signal_logger.log(settings.SYMBOL, evaluation, f"SKIP_{skip}")
                    else:
                        reason = block_reason if not allowed else "zero_size"
                        log.info(f"BUY blocked: {reason}")
                        signal_logger.log(settings.SYMBOL, evaluation, f"SKIP_{reason}")
                else:
                    log.info(f"No entry: {evaluation.block_reason}")
                    signal_logger.log(settings.SYMBOL, evaluation, f"SKIP_{evaluation.block_reason}")

            balance = get_balance(trade_exchange)
            risk.portfolio.update_peak(balance)
            print_status(balance, risk, last_evaluation, current_price)
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
