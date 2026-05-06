from decision_tree import DecisionTree
import numpy as np
import pandas as pd

def build_features(df):
    feature_cols = [
        "rank_diff",
        "rank_pts_diff",
        "age_diff",
        "height_diff",
        "surface_win_pct_diff",
        "ace_vs_df_diff",
        "first_in_diff",
        "first_won_diff",
        "second_won_diff",
        "bp_saved_pct_diff",
        "bp_converted_pct_diff",
        "win_pct_diff",
        "games_played_diff"
    ]

    X_df = df[feature_cols].fillna(0)
    y = df["p1_won"].astype(float).values

    return X_df, y

class BoostedTreeClassifier:
    def __init__(self, num_trees, learning_rate, max_depth, random_state, verbose=False):
        """Initialize the boosted tree classifier"""

        self.num_trees = num_trees
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, Y):
        """Fit the boosted tree classifier to the training data"""

        np.random.seed(self.random_state)
        self.trees = []
        self.tree_weights = []

        # Initialize residuals to the original labels
        residuals = Y.copy().flatten()

        for i in range(self.num_trees):
            if self.verbose:
                print(f"  fitting tree {i + 1}/{self.num_trees}...")
            print(f"  fitting tree {i + 1}/{self.num_trees}...")
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X, residuals.reshape(-1, 1))

            # Get predictions from the current tree
            predictions = np.array(tree.forward(X), dtype=float)

            # Update residuals: new_residuals = old_residuals - learning_rate * predictions
            residuals -= self.learning_rate * predictions

            self.trees.append(tree)
            self.tree_weights.append(self.learning_rate)
    def predict(self, X):
        """Predict class labels for the input data"""

        # Initialize predictions to zero
        final_predictions = np.zeros(X.shape[0])

        # Sum predictions from all trees, weighted by their learning rate
        for tree, weight in zip(self.trees, self.tree_weights):
            final_predictions += weight * np.array(tree.forward(X), dtype=float)

        # Convert final predictions to binary class labels (0 or 1)
        return (final_predictions > 0.5).astype(int)

if __name__ == "__main__":

    from sklearn.model_selection import train_test_split
    import pandas as pd

    df = pd.read_csv("data/cleaned/atp_match_features_2.csv")

    X, Y = build_features(df)
    print("Features and labels built.")
    Y = Y.astype(float).reshape(-1, 1)

    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.5, random_state=99
    )
    print("Data split")
    boost = BoostedTreeClassifier(
        num_trees=10,
        learning_rate=0.1,
        max_depth=10,
        random_state=99
    )
    print("about to fit the model!")
    print(X_train.dtypes)
    print(Y_train.dtype)
    print(X_train[:2])
    X_train = X_train.values.astype(float)
    X_val = X_val.values.astype(float)
    boost.fit(X_train, Y_train)
    print("Model fitted.")

    Y_pred = boost.predict(X_val)
    accuracy = 0

    for i in range(len(Y_pred)):
        if Y_pred[i] == Y_val[i]:
            accuracy += 1

    accuracy = accuracy / len(Y_pred)
    print(accuracy)


    