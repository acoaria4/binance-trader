"""
diagnose_tft.py
Check what signals TFT is actually generating.
python diagnose_tft.py
"""
import numpy as np
import pandas as pd
from exchange import get_data_exchange, fetch_ohlcv
from features import compute_features, FEATURE_COLS
from strategy import TFTStrategy, make_labels, SEQ_LEN
from config   import settings
import torch

exchange = get_data_exchange()
df_raw   = fetch_ohlcv(exchange, limit=2000, timeframe="4h")
df       = compute_features(df_raw)

tft = TFTStrategy()
if not tft.fitted:
    print("No model found — train first with: python train_tft.py")
    exit()

# Run predictions on last 200 candles directly
from sklearn.preprocessing import StandardScaler
feats = df[FEATURE_COLS].values
feats_scaled = tft.scaler.transform(feats)

all_probs = []
for i in range(SEQ_LEN, min(len(feats_scaled), SEQ_LEN + 200)):
    seq = feats_scaled[i-SEQ_LEN:i]
    x   = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
    tft.model.eval()
    with torch.no_grad():
        logits = tft.model(x)
        probs  = torch.softmax(logits, dim=1)[0].numpy()
    all_probs.append(probs)

all_probs = np.array(all_probs)
p_sell = all_probs[:, 0]
p_hold = all_probs[:, 1]
p_buy  = all_probs[:, 2]

print("\n── TFT probability stats ──────────────────────")
print(f"  p_buy  | mean={p_buy.mean():.3f} max={p_buy.max():.3f} min={p_buy.min():.3f}")
print(f"  p_hold | mean={p_hold.mean():.3f} max={p_hold.max():.3f} min={p_hold.min():.3f}")
print(f"  p_sell | mean={p_sell.mean():.3f} max={p_sell.max():.3f} min={p_sell.min():.3f}")

preds = np.argmax(all_probs, axis=1)
labels = {0:"SELL", 1:"HOLD", 2:"BUY"}
unique, counts = np.unique(preds, return_counts=True)
print("\n── Prediction counts ───────────────────────────")
for u, c in zip(unique, counts):
    print(f"  {labels[u]}: {c}")

print("\n── BUY signals above thresholds ────────────────")
for t in [0.33, 0.35, 0.38, 0.40, 0.45, 0.50]:
    print(f"  p_buy > {t:.2f}: {(p_buy > t).sum()} candles")
