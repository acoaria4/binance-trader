"""
strategy_tft.py
Temporal Fusion Transformer (TFT) Strategy Engine.

Replaces LightGBM with a proper time series deep learning model.

Key advantages over LightGBM:
  - Understands temporal order and long-range dependencies
  - Multi-horizon forecasting (predicts next 1, 3, 5, 10 candles)
  - Attention mechanism shows WHICH past candles influenced the decision
  - Uncertainty quantification — knows when it doesn't know
  - Handles multiple assets/symbols as covariates

Architecture:
  - Variable Selection Networks (VSN) — learns which features matter
  - Gated Residual Networks (GRN) — non-linear feature processing
  - LSTM encoder/decoder — captures temporal dynamics
  - Multi-head attention — long-range pattern recognition
  - Quantile output heads — probabilistic predictions

Uses PyTorch + PyTorch Forecasting library.

Install:
  pip install pytorch-forecasting pytorch-lightning torch
"""
import os
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    import torch
    import pytorch_lightning as pl
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import QuantileLoss
    TFT_AVAILABLE = True
except ImportError:
    TFT_AVAILABLE = False

from features import FEATURE_COLS, compute_features
from config import settings
from utils.logger import get_logger

log = get_logger(__name__)

MODEL_PATH   = "models/tft_model.ckpt"
SCALER_PATH  = "models/tft_scaler.pkl"

# TFT hyperparameters
MAX_ENCODER_LENGTH  = 72    # Look back 72 candles (3 days on 1h)
MAX_PREDICTION_LENGTH = 10  # Predict next 10 candles
HIDDEN_SIZE         = 64    # Model width
ATTENTION_HEAD_SIZE = 4
DROPOUT             = 0.1
HIDDEN_CONT_SIZE    = 32
BATCH_SIZE          = 64
MAX_EPOCHS          = 30
LEARNING_RATE       = 0.001

# Quantile targets — we predict the distribution, not just a point
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]


def prepare_tft_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform OHLCV + features DataFrame into TFT-compatible format.

    TFT requires:
      - time_idx: integer time index (0, 1, 2, ...)
      - group_id: series identifier (symbol name)
      - target: what to predict (future return)
      - known_reals: features known in the future (time features)
      - unknown_reals: features only known up to current time
    """
    df = df.copy()

    # Integer time index required by TFT
    df["time_idx"] = np.arange(len(df))

    # Group identifier (can extend to multi-asset later)
    df["symbol"] = settings.SYMBOL.replace("/", "_")

    # Target: forward 1-candle return (what we want to predict)
    df["target"] = df["close"].pct_change(1).shift(-1)

    # Time-based known features (cyclic encoding)
    if hasattr(df.index, 'hour'):
        df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
        df["dow_sin"]  = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df["dow_cos"]  = np.cos(2 * np.pi * df.index.dayofweek / 7)
    else:
        df["hour_sin"] = df["dow_sin"] = df["hour_cos"] = df["dow_cos"] = 0.0

    df.dropna(inplace=True)
    return df


def build_dataset(df: pd.DataFrame, training: bool = True):
    """Build PyTorch Forecasting TimeSeriesDataSet."""
    known_reals    = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "time_idx"]
    unknown_reals  = [f for f in FEATURE_COLS if f in df.columns]

    max_idx = df["time_idx"].max()
    cutoff  = max_idx - MAX_PREDICTION_LENGTH

    dataset = TimeSeriesDataSet(
        df[df["time_idx"] <= cutoff] if training else df,
        time_idx="time_idx",
        target="target",
        group_ids=["symbol"],
        min_encoder_length=MAX_ENCODER_LENGTH // 2,
        max_encoder_length=MAX_ENCODER_LENGTH,
        min_prediction_length=1,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        static_categoricals=["symbol"],
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=unknown_reals,
        target_normalizer=GroupNormalizer(groups=["symbol"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    return dataset


class TFTStrategy:
    """
    Temporal Fusion Transformer trading strategy.

    Drop-in replacement for MLStrategy — same interface:
      - train(df_raw)
      - predict(df_raw) -> (signal, confidence)
      - on_new_candle(df_raw)
    """

    def __init__(self):
        self.model  = None
        self.dataset_params = None
        self._candles_since_train = 0

        if not TFT_AVAILABLE:
            log.warning(
                "PyTorch Forecasting not installed. "
                "Run: pip install pytorch-forecasting pytorch-lightning torch\n"
                "Falling back to LightGBM strategy."
            )
        else:
            self._load_if_exists()

    # ── Persistence ───────────────────────────────────────────────────────────
    def _save(self, trainer, model):
        os.makedirs("models", exist_ok=True)
        trainer.save_checkpoint(MODEL_PATH)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(self.dataset_params, f)
        log.info("TFT model saved.")

    def _load_if_exists(self):
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            try:
                self.model = TemporalFusionTransformer.load_from_checkpoint(MODEL_PATH)
                with open(SCALER_PATH, "rb") as f:
                    self.dataset_params = pickle.load(f)
                log.info("TFT model loaded from disk.")
            except Exception as e:
                log.warning(f"Could not load TFT model: {e}")

    # ── Training ──────────────────────────────────────────────────────────────
    def train(self, df_raw: pd.DataFrame) -> None:
        if not TFT_AVAILABLE:
            log.error("Cannot train — PyTorch Forecasting not installed.")
            return

        log.info(f"Training TFT on {len(df_raw)} candles …")
        df = compute_features(df_raw)
        df = prepare_tft_dataframe(df)

        if len(df) < MAX_ENCODER_LENGTH * 3:
            log.warning(f"Need at least {MAX_ENCODER_LENGTH * 3} candles. Got {len(df)}.")
            return

        # Build datasets
        train_dataset = build_dataset(df, training=True)
        val_dataset   = TimeSeriesDataSet.from_dataset(
            train_dataset, df, predict=True, stop_randomization=True
        )
        self.dataset_params = train_dataset.get_parameters()

        train_loader = train_dataset.to_dataloader(
            train=True, batch_size=BATCH_SIZE, num_workers=0
        )
        val_loader = val_dataset.to_dataloader(
            train=False, batch_size=BATCH_SIZE * 2, num_workers=0
        )

        # Build model
        tft = TemporalFusionTransformer.from_dataset(
            train_dataset,
            learning_rate=LEARNING_RATE,
            hidden_size=HIDDEN_SIZE,
            attention_head_size=ATTENTION_HEAD_SIZE,
            dropout=DROPOUT,
            hidden_continuous_size=HIDDEN_CONT_SIZE,
            loss=QuantileLoss(quantiles=QUANTILES),
            log_interval=10,
            reduce_on_plateau_patience=4,
        )

        log.info(f"TFT parameters: {tft.size() / 1e3:.1f}k")

        # Train
        trainer = pl.Trainer(
            max_epochs=MAX_EPOCHS,
            accelerator="cpu",
            enable_progress_bar=True,
            gradient_clip_val=0.1,
            limit_train_batches=50,
            logger=False,
            enable_checkpointing=False,
        )
        trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

        self.model = tft
        self._save(trainer, tft)
        self._candles_since_train = 0
        log.info("TFT training complete.")

    # ── Inference ─────────────────────────────────────────────────────────────
    def predict(self, df_raw: pd.DataFrame) -> tuple[str, float]:
        """
        Returns (signal, confidence):
          signal     : 'BUY' | 'SELL' | 'HOLD'
          confidence : 0.0–1.0

        Uses the median (q=0.5) prediction for direction and
        the spread between q0.1 and q0.9 for confidence.
        """
        if self.model is None or not TFT_AVAILABLE:
            return "HOLD", 0.0

        try:
            df = compute_features(df_raw)
            df = prepare_tft_dataframe(df)

            if len(df) < MAX_ENCODER_LENGTH:
                return "HOLD", 0.0

            # Use last MAX_ENCODER_LENGTH rows for inference
            df_infer = df.iloc[-MAX_ENCODER_LENGTH:].copy()
            df_infer["time_idx"] = np.arange(len(df_infer))

            infer_dataset = TimeSeriesDataSet.from_parameters(
                self.dataset_params, df_infer, predict=True
            )
            infer_loader = infer_dataset.to_dataloader(
                train=False, batch_size=1, num_workers=0
            )

            # Get quantile predictions
            predictions = self.model.predict(
                infer_loader, mode="quantiles", return_x=False
            )
            # predictions shape: (batch, time_steps, quantiles)
            pred = predictions[0]  # first (only) batch

            # Use 1-step ahead prediction
            q10  = float(pred[0, 0])   # pessimistic
            q50  = float(pred[0, 2])   # median
            q90  = float(pred[0, 4])   # optimistic

            # Direction from median
            # Spread between q90 and q10 normalised as confidence proxy
            spread = abs(q90 - q10)
            conf   = min(float(spread / (abs(q50) + 1e-6 + spread)), 1.0)
            conf   = max(conf, 0.0)

            tp_thresh = settings.TAKE_PROFIT_PCT / 100 / MAX_PREDICTION_LENGTH
            sl_thresh = settings.STOP_LOSS_PCT   / 100 / MAX_PREDICTION_LENGTH

            if q50 > tp_thresh:
                signal = "BUY"
            elif q50 < -sl_thresh:
                signal = "SELL"
            else:
                signal = "HOLD"

            log.info(
                f"TFT signal: {signal} | conf={conf:.2%} | "
                f"q10={q10:.4f} q50={q50:.4f} q90={q90:.4f}"
            )
            return signal, conf

        except Exception as e:
            log.error(f"TFT predict error: {e}")
            return "HOLD", 0.0

    def on_new_candle(self, df_raw: pd.DataFrame) -> None:
        self._candles_since_train += 1
        if self._candles_since_train >= settings.RETRAIN_EVERY_N:
            self.train(df_raw)
