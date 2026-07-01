"""
strategy.py
ML Strategy Engine with calibrated probabilities, EV gating, and purged validation.
"""
import json
import os
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, f1_score

from features import compute_features, FEATURE_COLS, is_trending, is_mtf_aligned
from risk_sizing import barriers_for_entry
from simulation import make_labels, run_trading_simulation
from ev_gate import evaluate_signal, SignalEvaluation
from evaluation.purged_cv import purged_time_series_split
from reports.backtest_report import save_backtest_report
from backtest.engine import BacktestConfig, BacktestEngine
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

    def _save(self, extra_meta: dict = None):
        os.makedirs("models", exist_ok=True)
        with open(MODEL_PATH, "wb")  as f: pickle.dump(self.model,  f)
        with open(SCALER_PATH, "wb") as f: pickle.dump(self.scaler, f)
        meta = {
            "last_val_f1":    self._last_val_f1,
            "last_val_score": self._last_val_score,
            "updated_at":     datetime.now(timezone.utc).isoformat(),
        }
        if extra_meta:
            meta.update(extra_meta)
        _save_meta(meta)
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

    def _get_probs(self, df_raw: pd.DataFrame) -> tuple[float, float, float]:
        if self.model is None:
            return 0.0, 1.0, 0.0
        df = compute_features(df_raw)
        if df.empty:
            return 0.0, 1.0, 0.0
        last = df[FEATURE_COLS].iloc[[-1]].values
        proba = self.model.predict_proba(self.scaler.transform(last))[0]
        if len(proba) == 3:
            return float(proba[0]), float(proba[1]), float(proba[2])
        if len(proba) == 2:
            return float(proba[0]), float(proba[1]), 0.0
        return 0.0, 1.0, 0.0

    def evaluate(self, df_raw: pd.DataFrame, atr: float = None) -> SignalEvaluation:
        p_sell, p_hold, p_buy = self._get_probs(df_raw)
        df_feat = compute_features(df_raw)
        regime_ok = not settings.REQUIRE_TREND or is_trending(df_feat, settings.ADX_THRESHOLD)
        mtf_ok = is_mtf_aligned(df_feat)

        if atr is None and not df_feat.empty and "atr_14" in df_feat.columns:
            atr = float(df_feat["atr_14"].iloc[-1])

        entry = float(df_raw["close"].iloc[-1]) if len(df_raw) else 0.0
        barriers = barriers_for_entry(entry, atr) if entry > 0 else None
        sl_pct = barriers.stop_loss_pct if barriers else None
        tp_pct = barriers.take_profit_pct if barriers else None

        return evaluate_signal(
            p_sell, p_hold, p_buy, regime_ok, mtf_ok,
            sl_pct=sl_pct, tp_pct=tp_pct,
        )

    def _predict_with_model(self, model, scaler, df_raw: pd.DataFrame) -> tuple[str, float]:
        df = compute_features(df_raw)
        if df.empty:
            return "HOLD", 0.0
        proba = model.predict_proba(scaler.transform(df[FEATURE_COLS].iloc[[-1]].values))[0]
        if len(proba) == 3:
            p_sell, p_hold, p_buy = proba
        elif len(proba) == 2:
            p_sell, p_hold, p_buy = proba[0], proba[1], 0.0
        else:
            return "HOLD", 0.0
        probs = [p_sell, p_hold, p_buy]
        cls_id = int(np.argmax(probs))
        return {0: "SELL", 1: "HOLD", 2: "BUY"}[cls_id], float(max(probs))

    def _evaluate_with_model(self, model, scaler, df_raw: pd.DataFrame) -> SignalEvaluation:
        df = compute_features(df_raw)
        if df.empty:
            return evaluate_signal(0, 1, 0, False, False)
        proba = model.predict_proba(scaler.transform(df[FEATURE_COLS].iloc[[-1]].values))[0]
        if len(proba) == 3:
            p_sell, p_hold, p_buy = float(proba[0]), float(proba[1]), float(proba[2])
        elif len(proba) == 2:
            p_sell, p_hold, p_buy = float(proba[0]), float(proba[1]), 0.0
        else:
            p_sell, p_hold, p_buy = 0.0, 1.0, 0.0
        regime_ok = not settings.REQUIRE_TREND or is_trending(df, settings.ADX_THRESHOLD)
        mtf_ok = is_mtf_aligned(df)
        atr = float(df["atr_14"].iloc[-1]) if "atr_14" in df.columns else None
        entry = float(df_raw["close"].iloc[-1])
        barriers = barriers_for_entry(entry, atr)
        return evaluate_signal(
            p_sell, p_hold, p_buy, regime_ok, mtf_ok,
            sl_pct=barriers.stop_loss_pct, tp_pct=barriers.take_profit_pct,
        )

    def _cross_validate(self, X, y_mapped, df_raw, df_index) -> tuple[dict, np.ndarray]:
        f1_scores, trade_scores, trade_returns, trade_sharpes = [], [], [], []
        last_val_idx = None
        raw_indices = df_raw.index.get_indexer(df_index)

        for train_idx, val_idx in purged_time_series_split(len(X), n_splits=3):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_val   = scaler.transform(X[val_idx])

            base = self._build_classifier()
            base.fit(X_train, y_mapped[train_idx])

            calibrated = CalibratedClassifierCV(base, method="isotonic", cv="prefit")
            calibrated.fit(X_val, y_mapped[val_idx])

            val_pred = calibrated.predict(X_val)
            f1_scores.append(f1_score(
                y_mapped[val_idx], val_pred, average="macro", zero_division=0,
            ))

            val_raw_idx = raw_indices[val_idx]
            val_raw_idx = val_raw_idx[val_raw_idx >= 0]

            def evaluate_fn(window, _m=calibrated, _s=scaler):
                return self._evaluate_with_model(_m, _s, window)

            metrics = run_trading_simulation(
                df_raw, val_raw_idx,
                predict_fn=lambda w, _m=calibrated, _s=scaler: self._predict_with_model(_m, _s, w),
                regime_check_fn=lambda feat: is_trending(feat, settings.ADX_THRESHOLD),
                evaluate_fn=evaluate_fn,
            )
            trade_scores.append(metrics["score"])
            trade_returns.append(metrics["return_pct"])
            trade_sharpes.append(metrics["sharpe"])
            last_val_idx = val_idx

        if not f1_scores:
            from sklearn.model_selection import TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=3)
            for train_idx, val_idx in tscv.split(X):
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X[train_idx])
                X_val   = scaler.transform(X[val_idx])
                base = self._build_classifier()
                base.fit(X_train, y_mapped[train_idx])
                val_pred = base.predict(X_val)
                f1_scores.append(f1_score(
                    y_mapped[val_idx], val_pred, average="macro", zero_division=0,
                ))
                last_val_idx = val_idx
            return {
                "f1": float(np.mean(f1_scores)) if f1_scores else 0,
                "score": 0, "return_pct": 0, "sharpe": 0,
            }, last_val_idx if last_val_idx is not None else np.array([])

        return {
            "f1":         float(np.mean(f1_scores)),
            "score":      float(np.mean(trade_scores)),
            "return_pct": float(np.mean(trade_returns)),
            "sharpe":     float(np.mean(trade_sharpes)),
        }, last_val_idx

    def _passes_deploy_gate(self, metrics: dict, force: bool) -> bool:
        if force or self.model is None:
            return True
        return (
            metrics["f1"] >= settings.RETRAIN_MIN_F1
            and metrics["return_pct"] >= settings.RETRAIN_MIN_VAL_RETURN_PCT
            and metrics["sharpe"] >= settings.RETRAIN_MIN_VAL_SHARPE
            and metrics["score"] >= self._last_val_score * 0.95
        )

    def _fit_final_model(self, X, y_mapped) -> None:
        cal_split = int(len(X) * 0.85)
        X_tr, X_cal = X[:cal_split], X[cal_split:]
        y_tr, y_cal = y_mapped[:cal_split], y_mapped[cal_split:]

        X_tr_s = self.scaler.fit_transform(X_tr)
        X_cal_s = self.scaler.transform(X_cal)

        base = self._build_classifier()
        base.fit(X_tr_s, y_tr)

        if len(X_cal) >= 30 and len(np.unique(y_cal)) >= 2:
            calibrated = CalibratedClassifierCV(base, method="isotonic", cv="prefit")
            calibrated.fit(X_cal_s, y_cal)
            self.model = calibrated
        else:
            self.model = base

    def train(self, df_raw: pd.DataFrame, force: bool = False) -> bool:
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
            f"Purged validation — F1={metrics['f1']:.3f} | "
            f"return={metrics['return_pct']:+.2f}% | "
            f"Sharpe={metrics['sharpe']:.2f} | score={metrics['score']:.2f}"
        )

        if not self._passes_deploy_gate(metrics, force):
            log.warning(f"Retrain skipped — metrics failed deploy gate: {metrics}")
            self._candles_since_train = 0
            return False

        if len(last_val_idx) > 0:
            train_idx = np.arange(0, last_val_idx[0])
            val_idx   = last_val_idx
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_val   = scaler.transform(X[val_idx])
            report_model = self._build_classifier()
            report_model.fit(X_train, y_mapped[train_idx])
            val_pred = report_model.predict(X_val)
            present_labels = sorted(set(y_mapped[val_idx]) | set(val_pred))
            label_names = {0: "SELL", 1: "HOLD", 2: "BUY"}
            log.info("\n" + classification_report(
                y_mapped[val_idx], val_pred,
                labels=present_labels,
                target_names=[label_names[l] for l in present_labels],
                zero_division=0,
            ))

        self._fit_final_model(X, y_mapped)
        self._last_val_f1    = metrics["f1"]
        self._last_val_score = metrics["score"]

        report_path = self._save_validation_report(df_raw, last_val_idx)
        self._save(extra_meta={"last_report": report_path, "val_metrics": metrics})
        self._candles_since_train = 0
        log.info(
            f"Model deployed — return={metrics['return_pct']:+.2f}%, "
            f"Sharpe={metrics['sharpe']:.2f}, F1={metrics['f1']:.3f}"
        )
        return True

    def _save_validation_report(self, df_raw: pd.DataFrame, val_idx: np.ndarray) -> str:
        split = int(len(df_raw) * 0.8)
        engine = BacktestEngine(BacktestConfig(periodic_retrain=False))
        result = engine.run(df_raw, self, test_start_idx=split)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = save_backtest_report(result.to_report_dict(), f"reports/validation_{ts}.json")
        log.info(f"Validation report saved → {path}")
        return path

    def predict(self, df_raw: pd.DataFrame) -> tuple[str, float]:
        if self.model is None:
            log.warning("Model not trained yet — returning HOLD")
            return "HOLD", 0.0
        ev = self.evaluate(df_raw)
        conf = max(ev.p_sell, ev.p_hold, ev.p_buy)
        log.info(
            f"Signal: {ev.signal} | conf={conf:.2%} | "
            f"P(win)={ev.p_win:.2%} | EV={ev.ev:.4f} | conviction={ev.conviction:.2f}"
        )
        return ev.signal, conf

    def on_new_candle(self, df_raw: pd.DataFrame) -> bool:
        self._candles_since_train += 1
        if self._candles_since_train >= settings.RETRAIN_EVERY_N:
            return self.train(df_raw)
        return False


_label = make_labels

