from decision_tree import DecisionTree
import numpy as np 
import pandas as pd

class RandomForestClassifier:
    def __init__(self, num_trees, num_features, max_depth, random_state):
        """Initialize the random forest classifier"""

        self.num_trees = num_trees
        self.num_features = num_features
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X, Y):
        """Fit the random forest classifier to the training data"""

        np.random.seed(self.random_state)
        self.trees = []
        self.tree_features = []

        total_features = X.shape[1]

        for _ in range(self.num_trees):
            # Bootstrap sampling with replacement 
            row_indices = np.random.choice(len(X), size=len(X), replace=True)
            # Feature bagging: each tree gets a random subset of columns
            feature_indices = np.random.choice(total_features, size=self.num_features, replace=False)

            X_sample = X[row_indices][:, feature_indices]
            Y_sample = Y[row_indices]

            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_sample, Y_sample)

            self.trees.append(tree)
            self.tree_features.append(feature_indices)

    def forward(self, X):
        """Predict class labels by majority vote across all trees"""

        # Get predictions from each tree for its respective feature subset
        preds_per_tree = np.array([
            tree.forward(X[:, features])
            for tree, features in zip(self.trees, self.tree_features)
        ]).astype(int)
        preds_per_sample = preds_per_tree.T

        y_pred = [np.argmax(np.bincount(votes)) for votes in preds_per_sample]
        return np.array(y_pred)



if __name__ == "__main__":
    # Demo random forest on penguin dataset

    from sklearn.model_selection import train_test_split
    import pandas as pd

    url = "https://raw.githubusercontent.com/PhilChodrow/ml-notes/main/data/palmer-penguins/palmer-penguins.csv"
    df = pd.read_csv(url)

    feature_cols = [
        "Culmen Length (mm)",
        "Culmen Depth (mm)",
        "Flipper Length (mm)",
        "Body Mass (g)",
    ]
    df = df.dropna(subset=feature_cols + ["Species"])

    X = df[feature_cols].to_numpy()
    Y = np.vstack(df["Species"].to_numpy())

    Y_id_to_label = np.unique(Y)
    Y_label_to_id = {str(label): idx for idx, label in enumerate(Y_id_to_label)}
    Y = np.vstack(np.array([Y_label_to_id[label.item()] for label in Y]))

    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.5, random_state=99
    )

    forest = RandomForestClassifier(
        num_trees=50,
        num_features=2,
        max_depth=10,
        random_state=99
    )
    forest.fit(X_train, Y_train)

    Y_pred = forest.forward(X_val)
    accuracy = 0

    for i in range(len(Y_pred)):
        if Y_pred[i] == Y_val[i]:
            accuracy += 1

    accuracy = accuracy / len(Y_pred)
    print(accuracy)

