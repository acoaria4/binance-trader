"""
reports/backtest_report.py
Structured backtest report generation (JSON + console summary).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import settings


def _max_drawdown(equity: list[float]) -> float:
    if not equity:
        return 0.0
    peak = equity[0]
    max_dd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100 if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
    return max_dd


def build_report(result: dict) -> dict:
    """Build report dict from BacktestResult-like mapping."""
    trades = result.get("trades", [])
    closed = [t for t in trades if "pnl_pct" in t]
    pnls   = [t["pnl_pct"] / 100 for t in closed]
    equity = result.get("equity", [1000.0])

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    gross_profit = sum(wins) if wins else 0.0
    gross_loss   = abs(sum(losses)) if losses else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    rets = pd.Series(pnls) if pnls else pd.Series([0.0])
    sharpe = 0.0
    sortino = 0.0
    if len(rets) > 1:
        if rets.std() > 0:
            sharpe = float(rets.mean() / rets.std() * np.sqrt(252))
        downside = rets[rets < 0]
        if len(downside) > 0 and downside.std() > 0:
            sortino = float(rets.mean() / downside.std() * np.sqrt(252))

    start = equity[0] if equity else 1000.0
    end   = equity[-1] if equity else 1000.0
    total_return = (end - start) / start * 100

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbol": result.get("symbol", settings.SYMBOL),
        "timeframe": result.get("timeframe", settings.TIMEFRAME),
        "n_candles": result.get("n_candles", 0),
        "n_trades": len(closed),
        "win_rate_pct": len(wins) / len(closed) * 100 if closed else 0.0,
        "total_return_pct": total_return,
        "gross_return_pct": result.get("gross_return_pct", total_return),
        "cost_drag_pct": result.get("cost_drag_pct", 0.0),
        "profit_factor": profit_factor if profit_factor != float("inf") else 999.0,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown_pct": _max_drawdown(equity),
        "final_equity": end,
        "skipped": result.get("skipped", {}),
        "signal_counts": result.get("signal_counts", {}),
        "settings": {
            "sl_pct": settings.STOP_LOSS_PCT,
            "tp_pct": settings.TAKE_PROFIT_PCT,
            "fee_pct": settings.FEE_PCT,
            "slippage_pct": settings.SLIPPAGE_PCT,
            "ev_min": settings.EV_MIN,
            "conviction_min": settings.CONVICTION_MIN,
        },
    }


def save_backtest_report(result: dict, path: str = None) -> str:
    """Save JSON report; returns filepath."""
    report = build_report(result)
    if path is None:
        os.makedirs("reports", exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = f"reports/backtest_{ts}.json"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path


def print_backtest_summary(report: dict) -> None:
    """Print formatted console summary from report dict."""
    print("\n" + "─" * 56)
    print(f"  Backtest: {report['symbol']} | {report['timeframe']} | "
          f"{report['n_candles']} candles")
    print(f"  Costs: fee={report['settings']['fee_pct']}% "
          f"slip={report['settings']['slippage_pct']}% per side")
    print("─" * 56)
    print(f"  Trades:            {report['n_trades']}")
    print(f"  Win rate:          {report['win_rate_pct']:.1f}%")
    print(f"  Net return:        {report['total_return_pct']:+.2f}%")
    if report.get("gross_return_pct") is not None:
        print(f"  Gross return:      {report['gross_return_pct']:+.2f}%")
        print(f"  Cost drag:         {report.get('cost_drag_pct', 0):+.2f}%")
    print(f"  Profit factor:     {report['profit_factor']:.2f}")
    print(f"  Max drawdown:      -{report['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe:            {report['sharpe']:.2f}")
    print(f"  Sortino:           {report['sortino']:.2f}")
    print(f"  Final equity:      {report['final_equity']:.2f} USDT")
    skipped = report.get("skipped", {})
    if skipped:
        print(f"  Skipped:           {skipped}")
    print("─" * 56)

    equity = report.get("_equity")
    if equity and len(equity) > 1:
        mn, mx = min(equity), max(equity)
        height = 8
        step   = max(1, len(equity) // 60)
        rows   = []
        for row_i in range(height, 0, -1):
            line = ""
            for val in equity[::step]:
                norm = (val - mn) / (mx - mn + 1e-9)
                line += "█" if norm >= row_i / height else " "
            label = f"{mn + (mx-mn)*row_i/height:>8.0f}" if row_i in (1, height) else " " * 8
            rows.append(f"  {label} │{line}")
        print("\n  Equity curve (USDT)")
        print("\n".join(rows))
        print()
