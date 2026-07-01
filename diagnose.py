"""
diagnose.py — run this to inspect raw model probabilities
python diagnose.py
"""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, ".")

from exchange import get_data_exchange, fetch_ohlcv
from features import compute_features, FEATURE_COLS
from strategy import MLStrategy, make_labels
from config   import settings

exchange = get_data_exchange()
df_raw   = fetch_ohlcv(exchange, limit=5000)
df       = compute_features(df_raw)
df["label"] = make_labels(df)

split    = int(len(df) * 0.6)
df_train = df.iloc[:split]
df_test  = df.iloc[split:]

strat = MLStrategy()
if strat.model is None:
    strat.train(df_train.drop(columns=["label"]))

# Predict on full test set at once
X_test = df_test[FEATURE_COLS].values
X_sc   = strat.scaler.transform(X_test)
probas = strat.model.predict_proba(X_sc)
preds  = strat.model.predict(X_sc)

n_classes = probas.shape[1]
print(f"\nModel classes: {strat.model.classes_}  (n={n_classes})")

if n_classes == 3:
    p_sell, p_hold, p_buy = probas[:,0], probas[:,1], probas[:,2]
elif n_classes == 2:
    p_sell, p_hold, p_buy = probas[:,0], probas[:,1], np.zeros(len(probas))
else:
    p_sell = p_hold = p_buy = np.zeros(len(probas))

print("\n── Raw probability stats on test set ──────────────────")
print(f"  p_buy  | mean={p_buy.mean():.3f}  max={p_buy.max():.3f}  min={p_buy.min():.3f}")
print(f"  p_hold | mean={p_hold.mean():.3f}  max={p_hold.max():.3f}  min={p_hold.min():.3f}")
print(f"  p_sell | mean={p_sell.mean():.3f}  max={p_sell.max():.3f}  min={p_sell.min():.3f}")

print("\n── Prediction class counts ────────────────────────────")
unique, counts = np.unique(preds, return_counts=True)
label_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
for u, c in zip(unique, counts):
    print(f"  {label_map.get(u, u)}: {c}")

print("\n── BUY signals above various thresholds ───────────────")
for thresh in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
    n = int((p_buy > thresh).sum())
    print(f"  p_buy > {thresh:.2f}: {n} candles")

print("\n── Sample of actual probabilities (first 10 test rows) ─")
for i in range(min(10, len(probas))):
    print(f"  [{i}] sell={p_sell[i]:.3f} hold={p_hold[i]:.3f} buy={p_buy[i]:.3f} → pred={label_map.get(preds[i], preds[i])}")
