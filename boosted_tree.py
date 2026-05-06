from decision_tree import DecisionTree
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
        predictions = []

        # Sum predictions from all trees, weighted by their learning rate
        for tree, weight in zip(self.trees, self.tree_weights):
            tree_predictions = weight * np.array(tree.forward(X), dtype=float)
            final_predictions += tree_predictions
            predictions.append(tree_predictions)
        # Convert final predictions to binary class labels (0 or 1)

        probs = 1 / (1 + np.exp(-final_predictions))
        confidence = np.abs(probs - 0.5) * 2
        return (probs > 0.5).astype(int), confidence, predictions
    #Gets the summed confidence of the first n trees, going by 10, to show how confidence changes as more trees are added. 
    def getConfidences(self, tree_preds):
        for n in range(0, len(tree_preds)+1, 10):
            cumulative = sum(tree_preds[1:n+1])
            probs_n = 1 / (1 + np.exp(-cumulative))
            conf_n = np.abs(probs_n - 0.5) * 2
            print(f"Trees: {n}, Avg confidence: {conf_n.mean():.4f}")

    


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
        num_trees=100,
        learning_rate=0.25,
        max_depth=5,
        random_state=41
    )
    print("about to fit the model!")
    print(X_train.dtypes)
    print(Y_train.dtype)
    print(X_train[:2])
    X_train = X_train.values.astype(float)
    X_val = X_val.values.astype(float)
    boost.fit(X_train, Y_train)
    print("Model fitted.")

    Y_pred, Y_confidence, tree_preds = boost.predict(X_val)
    accuracy = 0

    for i in range(Y_pred.shape[0]):
        if Y_pred[i] == Y_val[i]:
            accuracy += 1

    accuracy = accuracy / len(Y_pred)
    print("Min confidence:", Y_confidence.min())
    print("Max confidence:", Y_confidence.max())
    print("Mean confidence:", Y_confidence.mean())
    print(accuracy)

    print(pd.DataFrame(tree_preds).head())
    boost.getConfidences(tree_preds)

    conf_mat = np.zeros((2, 2), dtype=int)
    for true, pred in zip(Y_val.flatten(), Y_pred.flatten()):
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

    steps = list(range(0, len(tree_preds) + 1, 10))
    fig, axes = plt.subplots(1, len(steps), figsize=(4 * len(steps), 4))

    for ax, n in zip(axes, steps):
        cumulative = sum(tree_preds[:n])
        probs_n = 1 / (1 + np.exp(-cumulative))
        Y_pred_n = (probs_n > 0.5).astype(int)

        conf_mat = np.zeros((2, 2), dtype=int)
        for true, pred in zip(Y_val.flatten(), Y_pred_n.flatten()):
            conf_mat[int(true), int(pred)] += 1

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

        ax.set_title(f"Tree: {n}")
        ax.grid(False)

    plt.tight_layout()
    plt.show()