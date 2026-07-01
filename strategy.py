"""
strategy.py
ML Strategy Engine: trains a LightGBM classifier that predicts
whether a long entry would hit take-profit before stop-loss.

Signal classes:
  1 = BUY  (TP barrier hit before SL within forward window)
  0 = HOLD (neither barrier hit in time)
 -1 = SELL (SL barrier hit before TP — avoid / exit long)

Labels use the same STOP_LOSS_PCT / TAKE_PROFIT_PCT as live trading
(triple-barrier method).
"""
import json
import os
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, f1_score

from features import FEATURE_COLS, compute_features
from config import settings
from utils.logger import get_logger

log = get_logger(__name__)

MODEL_PATH      = "models/lgbm_model.pkl"
SCALER_PATH     = "models/scaler.pkl"
META_PATH       = "models/training_meta.json"
FORWARD_CANDLES = 10   # Look ahead up to 10 candles for barrier hits


def make_labels(df: pd.DataFrame,
                stop_loss_pct: float = None,
                take_profit_pct: float = None,
                forward_candles: int = None) -> pd.Series:
    """
    Triple-barrier labels aligned with live SL/TP rules.

    For each candle, walk forward through subsequent highs/lows:
      BUY  (+1): take-profit reached before stop-loss
      SELL (-1): stop-loss reached before take-profit
      HOLD (0):  neither barrier hit within the forward window
    If both barriers could trigger on the same candle, SL wins (conservative).
    """
    sl_pct = stop_loss_pct if stop_loss_pct is not None else settings.STOP_LOSS_PCT
    tp_pct = take_profit_pct if take_profit_pct is not None else settings.TAKE_PROFIT_PCT
    horizon = forward_candles if forward_candles is not None else FORWARD_CANDLES

    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    labels = []

    for i in range(len(df)):
        entry = close.iloc[i]
        sl_price = entry * (1 - sl_pct / 100)
        tp_price = entry * (1 + tp_pct / 100)
        label = 0

        for j in range(1, horizon + 1):
            if i + j >= len(df):
                label = np.nan
                break

            bar_low  = low.iloc[i + j]
            bar_high = high.iloc[i + j]
            sl_hit = bar_low <= sl_price
            tp_hit = bar_high >= tp_price

            if sl_hit and tp_hit:
                label = -1
                break
            if sl_hit:
                label = -1
                break
            if tp_hit:
                label = 1
                break

        labels.append(label)

    return pd.Series(labels, index=df.index)


def _load_meta() -> dict:
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            return json.load(f)
    return {}


def _save_meta(meta: dict) -> None:
    os.makedirs("models", exist_ok=True)
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)


class MLStrategy:
    def __init__(self):
        self.model  = None
        self.scaler = StandardScaler()
        self._candles_since_train = 0
        self._last_val_f1 = _load_meta().get("last_val_f1", 0.0)
        self._load_if_exists()

    # Persistence
    def _save(self):
        os.makedirs("models", exist_ok=True)
        with open(MODEL_PATH, "wb")  as f: pickle.dump(self.model,  f)
        with open(SCALER_PATH, "wb") as f: pickle.dump(self.scaler, f)
        _save_meta({"last_val_f1": self._last_val_f1})
        log.info("Model + scaler saved.")

    def _load_if_exists(self):
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            with open(MODEL_PATH, "rb")  as f: self.model  = pickle.load(f)
            with open(SCALER_PATH, "rb") as f: self.scaler = pickle.load(f)
            self._last_val_f1 = _load_meta().get("last_val_f1", self._last_val_f1)
            log.info("Loaded existing model from disk.")

    def _build_classifier(self) -> lgb.LGBMClassifier:
        return lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.03,
            num_leaves=16,
            max_depth=4,
            min_child_samples=50,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=0.1,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )

    def _cross_val_f1(self, X: np.ndarray, y_mapped: np.ndarray) -> tuple[float, int, int]:
        """Walk-forward CV: train on past folds only, score on validation fold."""
        tscv = TimeSeriesSplit(n_splits=3)
        val_scores = []
        last_val_idx = None

        for train_idx, val_idx in tscv.split(X):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_val   = scaler.transform(X[val_idx])

            model = self._build_classifier()
            model.fit(X_train, y_mapped[train_idx])
            val_pred = model.predict(X_val)

            val_scores.append(f1_score(
                y_mapped[val_idx], val_pred, average="macro", zero_division=0,
            ))
            last_val_idx = val_idx

        return float(np.mean(val_scores)), last_val_idx, len(val_scores)

    # Training
    def train(self, df_raw: pd.DataFrame, force: bool = False) -> bool:
        """
        Train with walk-forward validation. Fits on all data only when:
          - no model exists yet, or force=True, or
          - validation macro-F1 meets RETRAIN_MIN_F1 and does not regress >5%
            vs the currently deployed model.
        Returns True if a new model was deployed.
        """
        log.info(f"Training on {len(df_raw)} candles ...")
        df = compute_features(df_raw)
        df["label"] = make_labels(df)
        df = df.dropna(subset=["label"])

        X = df[FEATURE_COLS].values
        y = df["label"].values.astype(int)
        y_mapped = y + 1   # SELL/HOLD/BUY: -1/0/1 -> LightGBM classes 0/1/2

        log.info(
            f"Label distribution: SELL={int(np.sum(y==-1))} "
            f"HOLD={int(np.sum(y==0))} BUY={int(np.sum(y==1))}"
        )

        if len(np.unique(y_mapped)) < 2:
            log.warning("Not enough label diversity to train — skipping.")
            return False

        avg_f1, last_val_idx, n_folds = self._cross_val_f1(X, y_mapped)
        log.info(f"Walk-forward validation macro-F1: {avg_f1:.3f} ({n_folds} folds)")

        should_deploy = (
            force
            or self.model is None
            or (
                avg_f1 >= settings.RETRAIN_MIN_F1
                and avg_f1 >= self._last_val_f1 * 0.95
            )
        )

        if not should_deploy:
            log.warning(
                f"Retrain skipped — val F1 {avg_f1:.3f} "
                f"(min={settings.RETRAIN_MIN_F1:.3f}, "
                f"prev={self._last_val_f1:.3f})"
            )
            self._candles_since_train = 0
            return False

        # Report on the last validation fold (no leakage into training set)
        scaler = StandardScaler()
        train_idx = np.arange(0, last_val_idx[0])
        val_idx   = last_val_idx
        X_train   = scaler.fit_transform(X[train_idx])
        X_val     = scaler.transform(X[val_idx])

        report_model = self._build_classifier()
        report_model.fit(X_train, y_mapped[train_idx])
        val_pred = report_model.predict(X_val)
        present_labels = sorted(set(y_mapped[val_idx]) | set(val_pred))
        label_names    = {0: "SELL", 1: "HOLD", 2: "BUY"}
        log.info("\n" + classification_report(
            y_mapped[val_idx], val_pred,
            labels=present_labels,
            target_names=[label_names[l] for l in present_labels],
            zero_division=0,
        ))

        # Deploy: fit final model on all data
        X_scaled = self.scaler.fit_transform(X)
        self.model = self._build_classifier()
        self.model.fit(X_scaled, y_mapped)
        self._last_val_f1 = avg_f1
        self._save()
        self._candles_since_train = 0
        log.info(f"Model deployed (val macro-F1={avg_f1:.3f})")
        return True

    # Inference
    def predict(self, df_raw: pd.DataFrame) -> tuple[str, float]:
        if self.model is None:
            log.warning("Model not trained yet — returning HOLD")
            return "HOLD", 0.0

        df = compute_features(df_raw)
        if df.empty or len(df) < 5:
            log.debug(f"predict(): empty features after dropna (window={len(df_raw)} rows)")
            return "HOLD", 0.0

        last = df[FEATURE_COLS].iloc[[-1]].values
        X_sc = self.scaler.transform(last)

        proba     = self.model.predict_proba(X_sc)[0]
        n_classes = len(proba)

        if n_classes == 3:
            p_sell, p_hold, p_buy = proba
        elif n_classes == 2:
            p_sell, p_hold, p_buy = proba[0], proba[1], 0.0
        else:
            p_sell, p_hold, p_buy = 0.0, 1.0, 0.0

        cls_id     = int(np.argmax([p_sell, p_hold, p_buy]))
        conf       = float(max(p_sell, p_hold, p_buy))
        signal_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
        signal     = signal_map[cls_id]

        log.info(f"Signal: {signal} | Confidence: {conf:.2%} "
                 f"[sell={p_sell:.2f} hold={p_hold:.2f} buy={p_buy:.2f}]")
        return signal, conf

    def on_new_candle(self, df_raw: pd.DataFrame) -> bool:
        """Increment candle counter; retrain with validation when due."""
        self._candles_since_train += 1
        if self._candles_since_train >= settings.RETRAIN_EVERY_N:
            return self.train(df_raw)
        return False


# Backward-compatible alias used by backtest.py and diagnose.py
_label = make_labels
