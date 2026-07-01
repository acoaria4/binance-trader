"""
strategy.py
ML Strategy Engine: LightGBM classifier trained on trailing-stop-aligned labels.
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

from features import FEATURE_COLS, compute_features, is_trending
from simulation import make_labels, run_trading_simulation
from config import settings
from utils.logger import get_logger

log = get_logger(__name__)

MODEL_PATH  = "models/lgbm_model.pkl"
SCALER_PATH = "models/scaler.pkl"
META_PATH   = "models/training_meta.json"


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
        meta = _load_meta()
        self._last_val_f1    = meta.get("last_val_f1", 0.0)
        self._last_val_score = meta.get("last_val_score", 0.0)
        self._load_if_exists()

    def _save(self):
        os.makedirs("models", exist_ok=True)
        with open(MODEL_PATH, "wb")  as f: pickle.dump(self.model,  f)
        with open(SCALER_PATH, "wb") as f: pickle.dump(self.scaler, f)
        _save_meta({
            "last_val_f1":    self._last_val_f1,
            "last_val_score": self._last_val_score,
        })
        log.info("Model + scaler saved.")

    def _load_if_exists(self):
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            with open(MODEL_PATH, "rb")  as f: self.model  = pickle.load(f)
            with open(SCALER_PATH, "rb") as f: self.scaler = pickle.load(f)
            meta = _load_meta()
            self._last_val_f1    = meta.get("last_val_f1", self._last_val_f1)
            self._last_val_score = meta.get("last_val_score", self._last_val_score)
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

    def _predict_with_model(self, model, scaler, df_raw: pd.DataFrame) -> tuple[str, float]:
        df = compute_features(df_raw)
        if df.empty:
            return "HOLD", 0.0
        last = df[FEATURE_COLS].iloc[[-1]].values
        proba = model.predict_proba(scaler.transform(last))[0]
        if len(proba) == 3:
            p_sell, p_hold, p_buy = proba
        elif len(proba) == 2:
            p_sell, p_hold, p_buy = proba[0], proba[1], 0.0
        else:
            return "HOLD", 0.0
        probs = [p_sell, p_hold, p_buy]
        cls_id = int(np.argmax(probs))
        return {0: "SELL", 1: "HOLD", 2: "BUY"}[cls_id], float(max(probs))

    def _cross_validate(self, X, y_mapped, df_raw, df_index) -> tuple[dict, np.ndarray]:
        """
        Walk-forward CV: classification F1 + trading simulation on each val fold.
        Returns aggregate metrics and the last validation row indices.
        """
        tscv = TimeSeriesSplit(n_splits=3)
        f1_scores = []
        trade_scores = []
        trade_returns = []
        trade_sharpes = []
        last_val_idx = None

        raw_indices = df_raw.index.get_indexer(df_index)

        for train_idx, val_idx in tscv.split(X):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_val   = scaler.transform(X[val_idx])

            model = self._build_classifier()
            model.fit(X_train, y_mapped[train_idx])
            val_pred = model.predict(X_val)
            f1_scores.append(f1_score(
                y_mapped[val_idx], val_pred, average="macro", zero_division=0,
            ))

            val_raw_idx = raw_indices[val_idx]
            val_raw_idx = val_raw_idx[val_raw_idx >= 0]

            def predict_fn(window, _m=model, _s=scaler):
                return self._predict_with_model(_m, _s, window)

            metrics = run_trading_simulation(
                df_raw, val_raw_idx, predict_fn,
                regime_check_fn=lambda feat: is_trending(feat, settings.ADX_THRESHOLD),
            )
            trade_scores.append(metrics["score"])
            trade_returns.append(metrics["return_pct"])
            trade_sharpes.append(metrics["sharpe"])
            last_val_idx = val_idx

        return {
            "f1":        float(np.mean(f1_scores)),
            "score":     float(np.mean(trade_scores)),
            "return_pct": float(np.mean(trade_returns)),
            "sharpe":    float(np.mean(trade_sharpes)),
        }, last_val_idx

    def _passes_deploy_gate(self, metrics: dict, force: bool) -> bool:
        if force or self.model is None:
            return True

        f1_ok = metrics["f1"] >= settings.RETRAIN_MIN_F1
        return_ok = metrics["return_pct"] >= settings.RETRAIN_MIN_VAL_RETURN_PCT
        sharpe_ok = metrics["sharpe"] >= settings.RETRAIN_MIN_VAL_SHARPE
        score_ok = metrics["score"] >= self._last_val_score * 0.95

        return f1_ok and return_ok and sharpe_ok and score_ok

    def train(self, df_raw: pd.DataFrame, force: bool = False) -> bool:
        """
        Train with walk-forward validation. Deploy only when validation trading
        metrics (return, Sharpe, composite score) and macro-F1 all pass gates.
        """
        log.info(f"Training on {len(df_raw)} candles ...")
        df = compute_features(df_raw)
        df["label"] = make_labels(df)
        df = df.dropna(subset=["label"])

        X = df[FEATURE_COLS].values
        y = df["label"].values.astype(int)
        y_mapped = y + 1

        log.info(
            f"Label distribution: SELL={int(np.sum(y==-1))} "
            f"HOLD={int(np.sum(y==0))} BUY={int(np.sum(y==1))}"
        )

        if len(np.unique(y_mapped)) < 2:
            log.warning("Not enough label diversity to train — skipping.")
            return False

        metrics, last_val_idx = self._cross_validate(X, y_mapped, df_raw, df.index)
        log.info(
            f"Validation — F1={metrics['f1']:.3f} | "
            f"return={metrics['return_pct']:+.2f}% | "
            f"Sharpe={metrics['sharpe']:.2f} | "
            f"score={metrics['score']:.2f}"
        )

        if not self._passes_deploy_gate(metrics, force):
            log.warning(
                f"Retrain skipped — val score {metrics['score']:.2f} "
                f"(prev={self._last_val_score:.2f}), "
                f"return={metrics['return_pct']:+.2f}% "
                f"(min={settings.RETRAIN_MIN_VAL_RETURN_PCT:.2f}%), "
                f"Sharpe={metrics['sharpe']:.2f} "
                f"(min={settings.RETRAIN_MIN_VAL_SHARPE:.2f}), "
                f"F1={metrics['f1']:.3f} (min={settings.RETRAIN_MIN_F1:.3f})"
            )
            self._candles_since_train = 0
            return False

        # Classification report on last fold (train past, validate future)
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

        X_scaled = self.scaler.fit_transform(X)
        self.model = self._build_classifier()
        self.model.fit(X_scaled, y_mapped)
        self._last_val_f1    = metrics["f1"]
        self._last_val_score = metrics["score"]
        self._save()
        self._candles_since_train = 0
        log.info(
            f"Model deployed — val return={metrics['return_pct']:+.2f}%, "
            f"Sharpe={metrics['sharpe']:.2f}, F1={metrics['f1']:.3f}"
        )
        return True

    def predict(self, df_raw: pd.DataFrame) -> tuple[str, float]:
        if self.model is None:
            log.warning("Model not trained yet — returning HOLD")
            return "HOLD", 0.0

        df = compute_features(df_raw)
        if df.empty or len(df) < 5:
            log.debug(f"predict(): empty features after dropna (window={len(df_raw)} rows)")
            return "HOLD", 0.0

        signal, conf = self._predict_with_model(self.model, self.scaler, df_raw)
        log.info(f"Signal: {signal} | Confidence: {conf:.2%}")
        return signal, conf

    def on_new_candle(self, df_raw: pd.DataFrame) -> bool:
        self._candles_since_train += 1
        if self._candles_since_train >= settings.RETRAIN_EVERY_N:
            return self.train(df_raw)
        return False


_label = make_labels
