import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random_forest import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier as SklearnRF
from features import FEATURE_SETS

verbose = "--verbose" in sys.argv or "-v" in sys.argv
use_sklearn = "--sklearn" in sys.argv

# Input feature set from data/cleaned/*.csv, defined in features.py
feature_set = "some_diffs.csv"
FEATURE_COLS = FEATURE_SETS[feature_set]
df = pd.read_csv(f"data/cleaned/{feature_set}")

"""
***RESULTS***

all_diffs.csv:

Train accuracy: 86.31%
Test accuracy:  63.48%
Gap:            22.83%
Min confidence: 0.5
Max confidence: 0.96
Mean confidence: 69.7%

no_diffs.csv:
Train accuracy: 96.35%
Test accuracy:  62.87%
Gap:            33.48%
Min confidence: 0.5
Max confidence: 0.92
Mean confidence: 62.00%

some_diffs.csv:
Train accuracy: 91.79%
Test accuracy:  63.13%
Gap:            28.66%
Min confidence: 0.5
Max confidence: 0.96
Mean confidence: 66.24%

"""

# Split by date to avoid leakage — train on pre-2022, test on 2022+
train_df = df[df["tourney_date"] < 20220101]
test_df  = df[df["tourney_date"] >= 20220101]

X_train = train_df[FEATURE_COLS].to_numpy()
Y_train = train_df["p1_won"].to_numpy()

X_test  = test_df[FEATURE_COLS].to_numpy()
Y_test  = test_df["p1_won"].to_numpy()

print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")
print(f"Features: {len(FEATURE_COLS)}")

n_trees = 100
max_depth = 20
max_features = 4

if use_sklearn:
    print("\nUsing sklearn RandomForestClassifier...")
    forest = SklearnRF(
        n_estimators=n_trees,
        max_depth=max_depth,
        max_features="sqrt",
        random_state=99,
        verbose=1 if verbose else 0,
        n_jobs=-1,
    )
    Y_train_fit = Y_train
else:
    print("\nUsing custom RandomForestClassifier...")
    forest = RandomForestClassifier(
        num_trees=n_trees,
        num_features=max_features,
        max_depth=max_depth,
        random_state=99,
        verbose=verbose,
    )
    Y_train_fit = Y_train.reshape(-1, 1)

print("Fitting...")
t0 = time.time()
forest.fit(X_train, Y_train_fit)
print(f"Fit done in {time.time() - t0:.3f}s")

t1 = time.time()

if use_sklearn:
    Y_pred = forest.predict(X_test)
    Y_confidence = np.max(forest.predict_proba(X_test), axis=1)
else:
    Y_pred, Y_confidence = forest.predict(X_test)

print(f"Inference done in {time.time() - t1:.3f}s")

if use_sklearn:
    Y_train_pred = forest.predict(X_train)
else:
    Y_train_pred, _ = forest.predict(X_train)

train_accuracy = (Y_train_pred == Y_train).sum() / len(Y_train)
accuracy = (Y_pred == Y_test).sum() / len(Y_test)
print(f"Train accuracy: {train_accuracy:.4f}")
print(f"Test accuracy:  {accuracy:.4f}")
print(f"Gap:            {train_accuracy - accuracy:.4f}")
print("Min confidence:", Y_confidence.min())
print("Max confidence:", Y_confidence.max())
print("Mean confidence:", Y_confidence.mean())

# Base rate: predict p1 wins if they have a lower (better) rank number
if "rank" in FEATURE_COLS:
    rank_baseline = (test_df["rank_diff"] < 0).astype(int).to_numpy()
    baseline_accuracy = (rank_baseline == Y_test).sum() / len(Y_test)
    print(f"Rank baseline: {baseline_accuracy:.4f}")

conf_mat = np.zeros((2, 2), dtype=int)
for true, pred in zip(Y_test, Y_pred):
    conf_mat[int(true), int(pred)] += 1

fig, ax = plt.subplots()
im = ax.imshow(conf_mat, cmap="Blues", origin="upper")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["0", "1"])
ax.set_yticklabels(["0", "1"])
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")

for i in range(conf_mat.shape[0]):
    for j in range(conf_mat.shape[1]):
        ax.text(j, i, conf_mat[i, j].item(), ha="center", va="center", color="black", size=6)

ax.set_title("Confusion Matrix")
ax.grid(False)
plt.tight_layout()
plt.show()

