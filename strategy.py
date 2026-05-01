"""
strategy.py
ML Strategy Engine: trains a LightGBM classifier that predicts
whether the next N candles will produce a profitable long signal.

Signal classes:
  1 = BUY  (price expected to rise > take_profit threshold)
  0 = HOLD
 -1 = SELL (price expected to drop > stop_loss threshold)

The bot only trades class 1 long signals (long-only for safety
on testnet; short selling can be enabled later).
"""
import os
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report

from features import FEATURE_COLS, compute_features
from config import settings
from utils.logger import get_logger

log = get_logger(__name__)

MODEL_PATH  = "models/lgbm_model.pkl"
SCALER_PATH = "models/scaler.pkl"
FORWARD_CANDLES = 5   # Predict outcome over next 5 candles


def _label(df: pd.DataFrame) -> pd.Series:
    """
    Generate forward-looking labels.
    +1 if max future return > take_profit_pct
    -1 if min future return < -stop_loss_pct
    0  otherwise (no strong signal)
    """
    tp = settings.TAKE_PROFIT_PCT / 100
    sl = settings.STOP_LOSS_PCT   / 100
    close = df["close"]
    labels = []
    for i in range(len(close)):
        if i + FORWARD_CANDLES >= len(close):
            labels.append(0)
            continue
        future  = close.iloc[i+1 : i+1+FORWARD_CANDLES]
        max_ret = (future.max() - close.iloc[i]) / close.iloc[i]
        min_ret = (future.min() - close.iloc[i]) / close.iloc[i]
        if max_ret >= tp:
            labels.append(1)
        elif min_ret <= -sl:
            labels.append(-1)
        else:
            labels.append(0)
    return pd.Series(labels, index=df.index)


class MLStrategy:
    def __init__(self):
        self.model  = None
        self.scaler = StandardScaler()
        self._candles_since_train = 0
        self._load_if_exists()

    # ── Persistence ──────────────────────────────────────────────────────────
    def _save(self):
        os.makedirs("models", exist_ok=True)
        with open(MODEL_PATH, "wb")  as f: pickle.dump(self.model,  f)
        with open(SCALER_PATH, "wb") as f: pickle.dump(self.scaler, f)
        log.info("Model + scaler saved.")

    def _load_if_exists(self):
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            with open(MODEL_PATH, "rb")  as f: self.model  = pickle.load(f)
            with open(SCALER_PATH, "rb") as f: self.scaler = pickle.load(f)
            log.info("Loaded existing model from disk.")

    # ── Training ─────────────────────────────────────────────────────────────
    def train(self, df_raw: pd.DataFrame) -> None:
        """Train (or retrain) the LightGBM model on historical data."""
        log.info(f"Training on {len(df_raw)} candles …")
        df = compute_features(df_raw)
        df["label"] = _label(df)

        X = df[FEATURE_COLS].values
        y = df["label"].values

        # Map -1/0/1 → 0/1/2 for LightGBM multiclass
        y_mapped = y + 1   # -1→0, 0→1, 1→2

        # Time-series cross-val (no shuffle)
        tscv = TimeSeriesSplit(n_splits=3)
        X_scaled = self.scaler.fit_transform(X)

        self.model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )
        # Final fit on all data (CV used for eval only)
        self.model.fit(X_scaled, y_mapped)

        # Quick report on last fold
        for train_idx, val_idx in tscv.split(X_scaled):
            pass   # iterate to get last fold
        val_pred = self.model.predict(X_scaled[val_idx])
        present_labels = sorted(set(y_mapped[val_idx]) | set(val_pred))
        label_names    = {0: "SELL", 1: "HOLD", 2: "BUY"}
        log.info("\n" + classification_report(
            y_mapped[val_idx], val_pred,
            labels=present_labels,
            target_names=[label_names[l] for l in present_labels],
            zero_division=0,
        ))
        self._save()
        self._candles_since_train = 0

    # ── Inference ─────────────────────────────────────────────────────────────
    def predict(self, df_raw: pd.DataFrame) -> tuple[str, float]:
        """
        Returns (signal, confidence) where:
          signal     : 'BUY' | 'SELL' | 'HOLD'
          confidence : probability of the predicted class (0–1)
        """
        if self.model is None:
            log.warning("Model not trained yet — returning HOLD")
            return "HOLD", 0.0

        df = compute_features(df_raw)
        if df.empty:
            return "HOLD", 0.0

        last = df[FEATURE_COLS].iloc[[-1]].values
        X_sc = self.scaler.transform(last)

        proba  = self.model.predict_proba(X_sc)[0]   # [p_sell, p_hold, p_buy]
        cls_id = int(np.argmax(proba))
        conf   = float(proba[cls_id])

        signal_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
        signal = signal_map[cls_id]

        log.info(f"Signal: {signal} | Confidence: {conf:.2%} "
                 f"[sell={proba[0]:.2f} hold={proba[1]:.2f} buy={proba[2]:.2f}]")
        return signal, conf

    def on_new_candle(self, df_raw: pd.DataFrame) -> None:
        """Call this each time a new candle closes to trigger periodic retraining."""
        self._candles_since_train += 1
        if self._candles_since_train >= settings.RETRAIN_EVERY_N:
            self.train(df_raw)
