"""
features.py
Compute technical indicators used as ML features.

v2 additions:
  - Regime filter features (ADX, EMA alignment, price vs EMA200)
  - Volatility filter (ATR ratio)
  - Additional momentum features (ROC, Williams %R)
"""
import pandas as pd
import numpy as np
import ta

from config import settings


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    # ── Trend ──────────────────────────────────────────────────────────────────
    df["ema_9"]   = ta.trend.ema_indicator(c, window=9)
    df["ema_21"]  = ta.trend.ema_indicator(c, window=21)
    df["ema_50"]  = ta.trend.ema_indicator(c, window=50)
    df["ema_200"] = ta.trend.ema_indicator(c, window=200)

    macd_obj          = ta.trend.MACD(c)
    df["macd"]        = macd_obj.macd()
    df["macd_signal"] = macd_obj.macd_signal()
    df["macd_diff"]   = macd_obj.macd_diff()

    # ── Regime Filter Features ─────────────────────────────────────────────────
    # ADX: measures trend strength. >25 = trending, <20 = ranging
    adx                = ta.trend.ADXIndicator(h, l, c, window=14)
    df["adx"]          = adx.adx()
    df["adx_pos"]      = adx.adx_pos()   # +DI
    df["adx_neg"]      = adx.adx_neg()   # -DI

    # EMA alignment score: how many EMAs are stacked bullishly (9>21>50)
    df["ema_align"]    = (
        (df["ema_9"]  > df["ema_21"]).astype(int) +
        (df["ema_21"] > df["ema_50"]).astype(int) +
        (df["ema_50"] > df["ema_200"]).astype(int)
    )

    # Price position relative to EMA200 (key trend divider)
    df["price_vs_ema200"] = (c - df["ema_200"]) / df["ema_200"]

    # EMA slope (momentum of the trend itself)
    df["ema_50_slope"]  = df["ema_50"].pct_change(5,  fill_method=None)
    df["ema_200_slope"] = df["ema_200"].pct_change(10, fill_method=None)

    # ── Momentum ───────────────────────────────────────────────────────────────
    df["rsi_14"] = ta.momentum.RSIIndicator(c, window=14).rsi()

    stoch = ta.momentum.StochasticOscillator(h, l, c)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # Rate of change
    df["roc_5"]  = ta.momentum.ROCIndicator(c, window=5).roc()
    df["roc_10"] = ta.momentum.ROCIndicator(c, window=10).roc()

    # Williams %R (overbought/oversold)
    df["williams_r"] = ta.momentum.WilliamsRIndicator(h, l, c, lbp=14).williams_r()

    # ── Volatility ─────────────────────────────────────────────────────────────
    bb = ta.volatility.BollingerBands(c, window=20)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_mid"]   = bb.bollinger_mavg()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    df["bb_pct"]   = bb.bollinger_pband()   # Where price sits in the band (0-1)

    df["atr_14"] = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range()

    # ATR ratio: current ATR vs 50-period average ATR (volatility regime)
    df["atr_ratio"] = df["atr_14"] / (df["atr_14"].rolling(50).mean() + 1e-9)

    # ── Volume ─────────────────────────────────────────────────────────────────
    df["obv"] = ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume()
    vol_mean  = v.rolling(20).mean()
    vol_std   = v.rolling(20).std()
    df["vol_z"]     = (v - vol_mean) / (vol_std + 1e-9)
    df["vol_ratio"] = v / (vol_mean + 1e-9)   # Current vs average volume

    # ── Candle structure ───────────────────────────────────────────────────────
    df["candle_body"]  = (c - df["open"]).abs() / (df["open"] + 1e-9)
    df["upper_shadow"] = (h - df[["open", "close"]].max(axis=1)) / (df["open"] + 1e-9)
    df["lower_shadow"] = (df[["open", "close"]].min(axis=1) - l) / (df["open"] + 1e-9)
    df["candle_dir"]   = np.sign(c - df["open"])   # +1 bullish, -1 bearish

    # ── N-candle returns ───────────────────────────────────────────────────────
    df["ret_1"]  = c.pct_change(1,  fill_method=None)
    df["ret_3"]  = c.pct_change(3,  fill_method=None)
    df["ret_5"]  = c.pct_change(5,  fill_method=None)
    df["ret_10"] = c.pct_change(10, fill_method=None)

    df.dropna(inplace=True)
    return df


def is_trending(df: pd.DataFrame,
                adx_threshold: float = 20.0,
                min_atr_ratio: float = None) -> bool:
    """
    Regime filter: returns True if the market is currently trending.
    Uses the last row of a feature-computed DataFrame.

    All conditions must be met:
      - ADX > adx_threshold (trend strength)
      - ATR ratio > min_atr_ratio (not in a volatility collapse)
      - EMA alignment >= 2 OR price is >1% away from EMA200
    """
    if df.empty or "adx" not in df.columns:
        return True

    atr_min = min_atr_ratio if min_atr_ratio is not None else settings.MIN_ATR_RATIO
    last = df.iloc[-1]
    adx_ok       = last["adx"] > adx_threshold
    atr_ok       = last["atr_ratio"] > atr_min
    structure_ok = last["ema_align"] >= 2 or abs(last["price_vs_ema200"]) > 0.01
    return bool(adx_ok and atr_ok and structure_ok)


FEATURE_COLS = [
    # Trend
    "ema_9", "ema_21", "ema_50", "ema_200",
    "macd", "macd_signal", "macd_diff",
    # Regime
    "adx", "adx_pos", "adx_neg",
    "ema_align", "price_vs_ema200",
    "ema_50_slope", "ema_200_slope",
    # Momentum
    "rsi_14", "stoch_k", "stoch_d",
    "roc_5", "roc_10", "williams_r",
    # Volatility
    "bb_width", "bb_pct", "atr_14", "atr_ratio",
    # Volume
    "obv", "vol_z", "vol_ratio",
    # Candle
    "candle_body", "upper_shadow", "lower_shadow", "candle_dir",
    # Returns
    "ret_1", "ret_3", "ret_5", "ret_10",
]
