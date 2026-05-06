from decision_tree import DecisionTree
import numpy as np
import pandas as pd
from features import FEATURE_SETS

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
        residuals = Y.copy().flatten().astype(float)

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

    feature_set = "some_diffs.csv"
    FEATURE_COLS = FEATURE_SETS[feature_set]
    df = pd.read_csv(f"data/cleaned/{feature_set}")

    # Split by date to avoid leakage — train on pre-2022, test on 2022+
    train_df = df[df["tourney_date"] < 20220101]
    test_df  = df[df["tourney_date"] >= 20220101]

    X_train = train_df[FEATURE_COLS].to_numpy()
    Y_train = train_df["p1_won"].to_numpy()

    X_test  = test_df[FEATURE_COLS].to_numpy()
    Y_test  = test_df["p1_won"].to_numpy()

    print("Features and labels built.")

    boost = BoostedTreeClassifier(
        num_trees=10,
        learning_rate=0.1,
        max_depth=10,
        random_state=99
    )
    print("about to fit the model!")
    boost.fit(X_train, Y_train)
    print("Model fitted.")

    Y_pred = boost.predict(X_test)
    accuracy = 0

    for i in range(len(Y_pred)):
        if Y_pred[i] == Y_test[i]:
            accuracy += 1

    accuracy = accuracy / len(Y_pred)
    print(accuracy)


    