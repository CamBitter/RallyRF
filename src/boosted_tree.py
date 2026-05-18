from src.decision_tree import DecisionTree
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.features import FEATURE_SETS

class BoostedTreeClassifier:
    def __init__(self, num_trees, learning_rate, max_depth, random_state, verbose=False):
        """Initialize the boosted tree classifier"""

        self.num_trees = num_trees
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.verbose = verbose
        self.base_prediction = None

    def fit(self, X, Y):
        """Fit the boosted tree classifier to the training data"""

        np.random.seed(self.random_state)
        self.trees = []
        self.tree_weights = []

        # Initialize residuals to the original labels
        self.base_prediction = Y.copy().flatten().astype(float).mean()
        residuals = Y.copy().flatten().astype(float) - self.base_prediction

        for i in range(self.num_trees):
            if self.verbose:
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
        final_predictions = np.full(X.shape[0], self.base_prediction)
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

    feature_set = "all_diffs.csv"
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
        learning_rate=0.25,
        max_depth=5,
        random_state=41,
        verbose=True
    )
    print("about to fit the model!")
    boost.fit(X_train, Y_train)
    print("Model fitted.")

    Y_pred, Y_confidence, tree_preds = boost.predict(X_test)
    accuracy = 0

    for i in range(Y_pred.shape[0]):
        if Y_pred[i] == Y_test[i]:
            accuracy += 1

    accuracy = accuracy / len(Y_pred)
    print("Accuracy:", accuracy)
    print("Min confidence:", Y_confidence.min())
    print("Max confidence:", Y_confidence.max())
    print("Mean confidence:", Y_confidence.mean())

    print(pd.DataFrame(tree_preds).head())
    boost.getConfidences(tree_preds)

    conf_mat = np.zeros((2, 2), dtype=int)
    for true, pred in zip(Y_test.flatten(), Y_pred.flatten()):
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
        for true, pred in zip(Y_test.flatten(), Y_pred_n.flatten()):
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