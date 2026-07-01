"""
backtest.py
CLI wrapper around the unified BacktestEngine.

Usage:
    python backtest.py --symbol BTC/USDT --timeframe 1h --limit 5000
"""
import argparse

from exchange import get_data_exchange, fetch_ohlcv
from strategy import MLStrategy
from backtest.engine import BacktestConfig, BacktestEngine
from reports.backtest_report import save_backtest_report, print_backtest_summary, build_report
from config import settings
from utils.logger import get_logger

log = get_logger("backtest")


def run_backtest(symbol: str, timeframe: str, limit: int,
                 train_frac: float = 0.6) -> dict:

    exchange = get_data_exchange()
    df_raw   = fetch_ohlcv(exchange, symbol=symbol, timeframe=timeframe, limit=limit)

    split = int(len(df_raw) * train_frac)
    log.info(f"Train candles: {split} | Test candles: {len(df_raw) - split}")

    strat = MLStrategy()
    strat.train(df_raw.iloc[:split], force=True)

    config = BacktestConfig(symbol=symbol, timeframe=timeframe)
    engine = BacktestEngine(config)
    result = engine.run(df_raw, strat, test_start_idx=split)

    report_dict = result.to_report_dict()
    report = build_report(report_dict)
    report["_equity"] = result.equity

    print_backtest_summary(report)
    print(f"  Retrains:          {result.retrain_count}")
    print(f"  Signal counts:     {result.signal_counts}")

    path = save_backtest_report(report_dict)
    log.info(f"Report saved → {path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",    default=settings.SYMBOL)
    parser.add_argument("--timeframe", default=settings.TIMEFRAME)
    parser.add_argument("--limit",     default=5000, type=int)
    args = parser.parse_args()
    run_backtest(args.symbol, args.timeframe, args.limit)
