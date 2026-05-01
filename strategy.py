"""
strategy.py
ML Strategy Engine: trains a LightGBM classifier that predicts
whether the next N candles will produce a profitable long signal.

Signal classes:
  1 = BUY  (top 25% forward return opportunities)
  0 = HOLD (middle 50%)
 -1 = SELL (bottom 25% — worst forward return windows)

Uses percentile-based labelling so BUY/SELL classes are always
present in training data regardless of dataset size or TP/SL settings.
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

MODEL_PATH      = "models/lgbm_model.pkl"
SCALER_PATH     = "models/scaler.pkl"
FORWARD_CANDLES = 10   # Look ahead 10 candles


def _label(df: pd.DataFrame) -> pd.Series:
    """
    Percentile-based forward-looking labels.

    For each candle, compute a net forward score:
        score = max_return_over_next_N + min_return_over_next_N

    High score = good opportunity (price went up, didn't dip much)
    Low score  = bad window (price dropped hard)

    Top 25% scores -> BUY  (+1)
    Bottom 25%     -> SELL (-1)
    Middle 50%     -> HOLD (0)

    This guarantees all 3 classes exist in every training batch.
    """
    close = df["close"]
    scores = []

    for i in range(len(close)):
        if i + FORWARD_CANDLES >= len(close):
            scores.append(np.nan)
            continue
        future  = close.iloc[i+1 : i+1+FORWARD_CANDLES]
        max_ret = (future.max() - close.iloc[i]) / close.iloc[i]
        min_ret = (future.min() - close.iloc[i]) / close.iloc[i]
        scores.append(max_ret + min_ret)

    score_series = pd.Series(scores, index=df.index)
    buy_thresh   = score_series.quantile(0.75)
    sell_thresh  = score_series.quantile(0.25)

    labels = []
    for score in scores:
        if pd.isna(score):
            labels.append(0)
        elif score >= buy_thresh:
            labels.append(1)
        elif score <= sell_thresh:
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

    # Persistence
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

    # Training
    def train(self, df_raw: pd.DataFrame) -> None:
        log.info(f"Training on {len(df_raw)} candles ...")
        df = compute_features(df_raw)
        df["label"] = _label(df)

        X = df[FEATURE_COLS].values
        y = df["label"].values
        y_mapped = y + 1   # -1->0, 0->1, 1->2

        unique_classes = np.unique(y_mapped)
        log.info(f"Label distribution: SELL={int(np.sum(y==-1))} HOLD={int(np.sum(y==0))} BUY={int(np.sum(y==1))}")

        if len(unique_classes) < 2:
            log.warning("Not enough label diversity to train — skipping.")
            return

        tscv     = TimeSeriesSplit(n_splits=3)
        X_scaled = self.scaler.fit_transform(X)

        self.model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.03,      # Slower learning = less overfitting
            num_leaves=16,           # Reduced from 31 = simpler trees
            max_depth=4,             # Shallower = less memorisation
            min_child_samples=50,    # Raised from 20 = needs more samples per leaf
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.1,           # L1 regularisation
            reg_lambda=0.1,          # L2 regularisation
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )
        self.model.fit(X_scaled, y_mapped)

        for train_idx, val_idx in tscv.split(X_scaled):
            pass
        val_pred       = self.model.predict(X_scaled[val_idx])
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

    # Inference
    def predict(self, df_raw: pd.DataFrame) -> tuple[str, float]:
        if self.model is None:
            log.warning("Model not trained yet — returning HOLD")
            return "HOLD", 0.0

        df = compute_features(df_raw)
        if df.empty or len(df) < 5:
            # Window too small after dropna — extend lookback in caller
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

    def on_new_candle(self, df_raw: pd.DataFrame) -> None:
        self._candles_since_train += 1
        if self._candles_since_train >= settings.RETRAIN_EVERY_N:
            self.train(df_raw)
