"""
strategy.py
Temporal Fusion Transformer — pure PyTorch implementation.
No pytorch-forecasting or neuralforecast dependencies.

Works on Python 3.13 with just:
    pip install torch numpy pandas scikit-learn

Architecture follows Lim et al. (2021) "Temporal Fusion Transformers
for Interpretable Multi-horizon Time Series Forecasting"
"""
import os
import math
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from features import FEATURE_COLS, compute_features
from config import settings
from utils.logger import get_logger

log = get_logger(__name__)

MODEL_PATH  = "models/tft_pure.pt"
SCALER_PATH = "models/tft_scaler.pkl"

# Hyperparameters
SEQ_LEN     = 48    # Input sequence length
HORIZON     = 10    # Forecast horizon
HIDDEN      = 64    # Hidden dimension
N_HEADS     = 4     # Attention heads
DROPOUT     = 0.1
BATCH_SIZE  = 64
EPOCHS      = 60
LR          = 1e-3
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Label generation ──────────────────────────────────────────────────────────
def make_labels(df: pd.DataFrame) -> pd.Series:
    """Percentile-based labels over HORIZON candles."""
    close = df["close"]
    scores = []
    for i in range(len(close)):
        if i + HORIZON >= len(close):
            scores.append(np.nan)
            continue
        fut     = close.iloc[i+1:i+1+HORIZON]
        max_ret = (fut.max() - close.iloc[i]) / close.iloc[i]
        min_ret = (fut.min() - close.iloc[i]) / close.iloc[i]
        scores.append(max_ret + min_ret)
    s = pd.Series(scores, index=df.index)
    buy_t  = s.quantile(0.75)
    sell_t = s.quantile(0.25)
    labels = []
    for v in scores:
        if pd.isna(v):       labels.append(1)   # HOLD
        elif v >= buy_t:     labels.append(2)   # BUY
        elif v <= sell_t:    labels.append(0)   # SELL
        else:                labels.append(1)   # HOLD
    return pd.Series(labels, index=df.index)


# ── Dataset ───────────────────────────────────────────────────────────────────
class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_sequences(features: np.ndarray, labels: np.ndarray):
    X, y = [], []
    for i in range(SEQ_LEN, len(features)):
        X.append(features[i-SEQ_LEN:i])
        y.append(labels[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


# ── TFT Model ─────────────────────────────────────────────────────────────────
class GRN(nn.Module):
    """Gated Residual Network — core TFT building block."""
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.fc1  = nn.Linear(d_model, d_model)
        self.fc2  = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.elu  = nn.ELU()
        self.sig  = nn.Sigmoid()

    def forward(self, x):
        h = self.elu(self.fc1(x))
        h = self.drop(self.fc2(h))
        g = self.sig(self.gate(x))
        return self.norm(x + g * h)


class TFTClassifier(nn.Module):
    """
    Simplified TFT for classification.
    Input:  (batch, seq_len, n_features)
    Output: (batch, 3)  — logits for SELL/HOLD/BUY
    """
    def __init__(self, n_features: int, hidden: int = HIDDEN,
                 n_heads: int = N_HEADS, dropout: float = DROPOUT):
        super().__init__()

        # Variable selection network
        self.vsn = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )

        # GRN after LSTM
        self.grn = GRN(hidden, dropout)

        # Multi-head self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden)

        # Output GRN + classifier
        self.out_grn = GRN(hidden, dropout)
        self.pool    = nn.AdaptiveAvgPool1d(1)
        self.head    = nn.Linear(hidden, 3)

    def forward(self, x):
        # x: (B, T, F)
        h = self.vsn(x)                          # (B, T, H)
        h, _ = self.lstm(h)                      # (B, T, H)
        h = self.grn(h)                          # (B, T, H)
        attn_out, _ = self.attn(h, h, h)        # (B, T, H)
        h = self.attn_norm(h + attn_out)         # residual
        h = self.out_grn(h)                      # (B, T, H)
        h = h.permute(0, 2, 1)                   # (B, H, T)
        h = self.pool(h).squeeze(-1)             # (B, H)
        return self.head(h)                      # (B, 3)


# ── Strategy class ────────────────────────────────────────────────────────────
class TFTStrategy:
    def __init__(self):
        self.model   = None
        self.scaler  = StandardScaler()
        self.fitted  = False
        self._candles_since_train = 0
        self._load_if_exists()

    def _save(self):
        os.makedirs("models", exist_ok=True)
        torch.save(self.model.state_dict(), MODEL_PATH)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(self.scaler, f)
        log.info("TFT model saved.")

    def _load_if_exists(self):
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            try:
                with open(SCALER_PATH, "rb") as f:
                    self.scaler = pickle.load(f)
                n_feat = len(FEATURE_COLS)
                self.model = TFTClassifier(n_feat).to(DEVICE)
                self.model.load_state_dict(
                    torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
                )
                self.model.eval()
                self.fitted = True
                log.info("TFT model loaded from disk.")
            except Exception as e:
                log.warning(f"Could not load TFT: {e}")

    def train(self, df_raw: pd.DataFrame) -> None:
        log.info(f"Training TFT on {len(df_raw)} candles …")
        df = compute_features(df_raw)
        df["label"] = make_labels(df)
        df.dropna(inplace=True)

        feats  = df[FEATURE_COLS].values
        labels = df["label"].values

        feats_scaled = self.scaler.fit_transform(feats)
        X, y = make_sequences(feats_scaled, labels)

        split = int(len(X) * 0.8)
        train_ds = TimeSeriesDataset(X[:split], y[:split])
        val_ds   = TimeSeriesDataset(X[split:], y[split:])

        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE * 2)

        n_feat     = len(FEATURE_COLS)
        self.model = TFTClassifier(n_feat).to(DEVICE)
        optimizer  = torch.optim.Adam(self.model.parameters(), lr=LR)
        scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS
        )
        # Weighted loss — penalise missing BUY/SELL more than HOLD
        # Since labels are balanced by percentile, equal weights work
        # but we boost BUY weight to force the model to predict it
        class_weights = torch.tensor([1.0, 0.5, 1.5], dtype=torch.float32).to(DEVICE)
        criterion  = nn.CrossEntropyLoss(weight=class_weights)

        best_val_acc = 0.0
        best_state   = None

        for epoch in range(EPOCHS):
            # Train
            self.model.train()
            for xb, yb in train_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            # Validate
            self.model.eval()
            correct = total = 0
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    preds   = self.model(xb).argmax(dim=1)
                    correct += (preds == yb).sum().item()
                    total   += len(yb)
            val_acc = correct / total if total > 0 else 0

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state   = {k: v.clone() for k, v in self.model.state_dict().items()}

            if (epoch + 1) % 5 == 0:
                log.info(f"Epoch {epoch+1}/{EPOCHS} | val_acc={val_acc:.3f} | best={best_val_acc:.3f}")

        if best_state:
            self.model.load_state_dict(best_state)

        self.model.eval()
        self.fitted = True
        self._save()
        self._candles_since_train = 0
        log.info(f"TFT training complete. Best val accuracy: {best_val_acc:.3f}")

    def predict(self, df_raw: pd.DataFrame) -> tuple[str, float]:
        if not self.fitted or self.model is None:
            return "HOLD", 0.0
        try:
            df = compute_features(df_raw)
            if len(df) < SEQ_LEN + 5:
                return "HOLD", 0.0

            feats = df[FEATURE_COLS].values
            feats_scaled = self.scaler.transform(feats)
            seq   = feats_scaled[-SEQ_LEN:]
            x     = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            self.model.eval()
            with torch.no_grad():
                logits = self.model(x)
                probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

            cls_id     = int(np.argmax(probs))
            conf       = float(probs[cls_id])
            signal_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
            signal     = signal_map[cls_id]

            log.info(
                f"TFT: {signal} | conf={conf:.2%} "
                f"[sell={probs[0]:.2f} hold={probs[1]:.2f} buy={probs[2]:.2f}]"
            )
            return signal, conf

        except Exception as e:
            log.error(f"TFT predict error: {e}")
            return "HOLD", 0.0

    def on_new_candle(self, df_raw: pd.DataFrame) -> None:
        self._candles_since_train += 1
        if self._candles_since_train >= settings.RETRAIN_EVERY_N:
            self.train(df_raw)
