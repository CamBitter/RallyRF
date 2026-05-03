import sys
import time
import numpy as np
import pandas as pd
from random_forest import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier as SklearnRF

verbose = "--verbose" in sys.argv or "-v" in sys.argv
use_sklearn = "--sklearn" in sys.argv

FEATURE_COLS = [
    "age_diff",
    "height_diff",
    "ace_vs_df_diff",
    "first_in_diff",
    "first_won_diff",
    "second_won_diff",
    "bp_converted_pct_diff",
    "win_pct_diff",
    "games_played_diff",
    "rank_diff"
]

df = pd.read_csv("data/cleaned/atp_match_features_2*.csv")

# Split by date to avoid leakage — train on pre-2022, test on 2022+
train_df = df[df["tourney_date"] < 20220101]
test_df  = df[df["tourney_date"] >= 20220101]

if not use_sklearn:
    train_df = train_df.sample(n=5000, random_state=99)

X_train = train_df[FEATURE_COLS].to_numpy()
Y_train = train_df["p1_won"].to_numpy()

X_test  = test_df[FEATURE_COLS].to_numpy()
Y_test  = test_df["p1_won"].to_numpy()

print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")
print(f"Features: {len(FEATURE_COLS)}")

if use_sklearn:
    print("\nUsing sklearn RandomForestClassifier...")
    forest = SklearnRF(
        n_estimators=100,
        max_depth=10,
        max_features="sqrt",
        random_state=99,
        verbose=1 if verbose else 0,
        n_jobs=-1,
    )
    Y_train_fit = Y_train
else:
    print("\nUsing custom RandomForestClassifier...")
    forest = RandomForestClassifier(
        num_trees=5,
        num_features=6,
        max_depth=5,
        random_state=99,
        verbose=verbose,
    )
    Y_train_fit = Y_train.reshape(-1, 1)

print("Fitting...")
t0 = time.time()
forest.fit(X_train, Y_train_fit)
print(f"Fit done in {time.time() - t0:.3f}s")

t1 = time.time()
Y_pred = forest.predict(X_test) if use_sklearn else forest.forward(X_test)
print(f"Inference done in {time.time() - t1:.3f}s")

accuracy = (Y_pred == Y_test).sum() / len(Y_test)
print(f"Test accuracy: {accuracy:.4f}")

# Base rate: predict p1 wins if they have a lower (better) rank number
rank_baseline = (test_df["rank_diff"] < 0).astype(int).to_numpy()
baseline_accuracy = (rank_baseline == Y_test).sum() / len(Y_test)
print(f"Rank baseline: {baseline_accuracy:.4f}")
