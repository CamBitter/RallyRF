import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.features import FEATURE_SETS

# This script identifies the top 10 permutations of all_diffs feature set that 
# provide the highest accuracy for a Random Forest classifier. It randomly samples 
# subsets of features, trains a model, and evaluates accuracy on a test set. 
# Note that it does not perform an exhaustive search, but rather a random search over the feature space.
# Most feature sets have essentially the same accuracy.
# Made by Cam

DATASET = "all_diffs.csv"
ALL_FEATURES = FEATURE_SETS[DATASET]
N_SEARCHES = 200
MIN_FEATURES = 3
RANDOM_SEED = 99

df = pd.read_csv(f"data/cleaned/{DATASET}")
train_df = df[df["tourney_date"] < 20220101]
test_df  = df[df["tourney_date"] >= 20220101]
Y_train = train_df["p1_won"].to_numpy()
Y_test  = test_df["p1_won"].to_numpy()

forest = RandomForestClassifier(n_estimators=50, max_depth=10, max_features="sqrt", random_state=99, n_jobs=-1)

rng = np.random.default_rng(RANDOM_SEED)
results = []

for i in range(N_SEARCHES):
    # sample a random subset of features (at least MIN_FEATURES)
    n = rng.integers(MIN_FEATURES, len(ALL_FEATURES) + 1)
    subset = list(rng.choice(ALL_FEATURES, size=n, replace=False))

    X_train = train_df[subset].to_numpy()
    X_test  = test_df[subset].to_numpy()

    forest.fit(X_train, Y_train)
    accuracy = (forest.predict(X_test) == Y_test).mean()
    results.append((accuracy, subset))

    if (i + 1) % 20 == 0:
        print(f"  {i + 1}/{N_SEARCHES} done...")

results.sort(reverse=True)

print(f"\nTop 10 feature subsets out of {N_SEARCHES} random searches:\n")
for rank, (acc, subset) in enumerate(results[:10], 1):
    print(f"#{rank}  accuracy={acc:.4f}  n={len(subset)}")
    for f in subset:
        print(f"    {f}")
    print()
