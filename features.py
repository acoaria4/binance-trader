"""
features.py
Compute technical indicators used as ML features.
Uses the `ta` library (Technical Analysis Library for Python).
"""
import pandas as pd
import numpy as np
import ta


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes raw OHLCV DataFrame and appends feature columns.

    Features added:
    ─── Trend ───────────────────────────────────────
      ema_9, ema_21, ema_50          Exponential MAs
      macd, macd_signal, macd_diff   MACD triple
    ─── Momentum ────────────────────────────────────
      rsi_14                         Relative Strength Index
      stoch_k, stoch_d               Stochastic oscillator
    ─── Volatility ──────────────────────────────────
      bb_upper, bb_mid, bb_lower     Bollinger Bands
      bb_width                       Band width (volatility proxy)
      atr_14                         Average True Range
    ─── Volume ──────────────────────────────────────
      obv                            On-Balance Volume
      vol_z                          Volume z-score (20-period)
    ─── Price-derived ───────────────────────────────
      candle_body                    |close - open| / open
      upper_shadow                   (high - max(open,close)) / open
      lower_shadow                   (min(open,close) - low) / open
      ret_1, ret_3, ret_5            N-candle returns
    """
    df = df.copy()
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    # ── Trend ──────────────────────────────────────────────
    df["ema_9"]  = ta.trend.ema_indicator(c, window=9)
    df["ema_21"] = ta.trend.ema_indicator(c, window=21)
    df["ema_50"] = ta.trend.ema_indicator(c, window=50)

    macd_obj         = ta.trend.MACD(c)
    df["macd"]       = macd_obj.macd()
    df["macd_signal"]= macd_obj.macd_signal()
    df["macd_diff"]  = macd_obj.macd_diff()

    # ── Momentum ───────────────────────────────────────────
    df["rsi_14"] = ta.momentum.RSIIndicator(c, window=14).rsi()

    stoch = ta.momentum.StochasticOscillator(h, l, c)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # ── Volatility ─────────────────────────────────────────
    bb = ta.volatility.BollingerBands(c, window=20)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_mid"]   = bb.bollinger_mavg()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

    df["atr_14"] = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range()

    # ── Volume ─────────────────────────────────────────────
    df["obv"] = ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume()
    vol_mean  = v.rolling(20).mean()
    vol_std   = v.rolling(20).std()
    df["vol_z"] = (v - vol_mean) / (vol_std + 1e-9)

    # ── Candle structure ───────────────────────────────────
    df["candle_body"]   = (c - df["open"]).abs() / (df["open"] + 1e-9)
    df["upper_shadow"]  = (h - df[["open", "close"]].max(axis=1)) / (df["open"] + 1e-9)
    df["lower_shadow"]  = (df[["open", "close"]].min(axis=1) - l) / (df["open"] + 1e-9)

    # ── N-candle returns ───────────────────────────────────
    df["ret_1"] = c.pct_change(1)
    df["ret_3"] = c.pct_change(3)
    df["ret_5"] = c.pct_change(5)

    df.dropna(inplace=True)
    return df


FEATURE_COLS = [
    "ema_9", "ema_21", "ema_50",
    "macd", "macd_signal", "macd_diff",
    "rsi_14", "stoch_k", "stoch_d",
    "bb_width", "atr_14",
    "obv", "vol_z",
    "candle_body", "upper_shadow", "lower_shadow",
    "ret_1", "ret_3", "ret_5",
]
