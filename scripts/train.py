import gzip
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.random_forest import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier as SklearnRF
from src.features import FEATURE_SETS
from src.decision_tree import DecisionTree
import pickle

verbose = "--verbose" in sys.argv or "-v" in sys.argv
use_sklearn = "--sklearn" in sys.argv

# Input feature set from data/cleaned/*.csv, defined in features.py
feature_set = "no_diffs.csv"
FEATURE_COLS = FEATURE_SETS[feature_set]
df = pd.read_csv(f"data/cleaned/{feature_set}")

# Split by date to avoid leakage — train on pre-2022, test on 2022+
date_split = 20220101

train_df = df[df["tourney_date"] < date_split]
test_df  = df[df["tourney_date"] >= date_split]

X_train = train_df[FEATURE_COLS].to_numpy()
Y_train = train_df["p1_won"].to_numpy()

X_test  = test_df[FEATURE_COLS].to_numpy()
Y_test  = test_df["p1_won"].to_numpy()

print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")
print(f"Features: {len(FEATURE_COLS)}")

n_trees = 150
max_depth = 20
max_features = 3

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

# Saved a compressed model to stay under Github 100MB limit
with gzip.open(f"models/forest-{feature_set[:-4]}-{n_trees}_trees-{max_depth}_depth-{max_features}_features.pkl.gz", "wb") as f:
    pickle.dump(forest, f)

print("Saved model.")

t1 = time.time()

# Print first 6 trees
Y_pred, Y_confidence = forest.predict(X_test)
for t in range(6):
    tree_feature_names = [FEATURE_COLS[i] for i in forest.tree_features[t]]
    forest.trees[t].print_tree(feature_names=tree_feature_names)

print(f"Inference done in {time.time() - t1:.3f}s")

# Get train and test accuracy
Y_train_pred, _ = forest.predict(X_train)
train_accuracy = (Y_train_pred == Y_train).sum() / len(Y_train)
accuracy = (Y_pred == Y_test).sum() / len(Y_test)
print(f"Train accuracy: {train_accuracy:.4f}")
print(f"Test accuracy:  {accuracy:.4f}")
print(f"Gap:            {train_accuracy - accuracy:.4f}")
print("Min confidence:", Y_confidence.min())
print("Max confidence:", Y_confidence.max())
print("Mean confidence:", Y_confidence.mean())
