import numpy as np

class Node:
    def __init__(
        self,
        feature=None,
        threshold=None,
        value=None,
        left=None,
        right=None,
        info_gain=None,
    ):
        """"""

        self.feature = feature
        self.threshold = threshold
        self.value = value
        self.left = left
        self.right = right
        self.info_gain = info_gain


class DecisionTree:
    def __init__(self, min_samples=2, max_depth=2):
        """"""

        self.min_samples = min_samples
        self.max_depth = max_depth
        self.root = None

    def forward(self, X):
        """Predicts a dataset"""
        y_pred = [self.predict_item(self.root, x) for x in X]
        return y_pred

    def predict_item(self, node, x):
        """Predicts single point by recursively moving down the decision tree"""

        if node.value != None:
            # Leaf node
            return node.value

        x_feature = x[node.feature]

        if x_feature <= float(node.threshold):
            return self.predict_item(node.left, x)
        else:
            return self.predict_item(node.right, x)

    def fit(self, X, Y):
        """Starts recursive building of decision tree, fits dataset to tree"""

        dataset = np.concat([X, Y], axis=1)
        self.root = self.build_tree(dataset, curr_depth=0)

    def build_tree(self, dataset, curr_depth):
        """Recursively builds trees around information gain maximizing splits"""

        X = dataset[:, :-1]
        Y = dataset[:, -1]

        num_samples, num_features = np.shape(X)

        if num_samples >= self.min_samples and curr_depth <= self.max_depth:
            best_split = self.get_best_split(dataset, num_features)

            # Recurse on best split
            if best_split["info_gain"] > 0:
                left = self.build_tree(best_split["left_dataset"], curr_depth + 1)
                right = self.build_tree(best_split["right_dataset"], curr_depth + 1)

                return Node(
                    feature=best_split["feature"],
                    threshold=best_split["threshold"],
                    left=left,
                    right=right,
                    info_gain=best_split["info_gain"],
                )

        # Leaf node
        leaf_value = self.calculate_leaf_value(Y)
        return Node(value=leaf_value)

    def print_tree(self, tree=None, prefix="", is_left=True):
        """Print out a fitted decision tree"""
        if tree is None:
            tree = self.root

        if tree.value is not None:
            print(prefix + "└── " + f"[{tree.value}]")
        else:
            connector = "├── " if is_left else "└── "
            print(
                prefix
                + connector
                + f"X{tree.feature} <= {float(tree.threshold):.3f}  (gain: {float(tree.info_gain):.4f})"
            )
            child_prefix = prefix + ("│   " if is_left else "    ")
            self.print_tree(tree.left, child_prefix, is_left=True)
            self.print_tree(tree.right, child_prefix, is_left=False)

    def get_best_split(self, dataset, num_features):
        """Returns the best split feature and threshold using a vectorized cumsum sweep over sorted feature values."""

        X = dataset[:, :-1]
        Y = dataset[:, -1]
        n = len(Y)

        # Parent gini for binary labels
        parent_gini = self.binary_gini(Y)

        best_split = {"info_gain": 0}
        best_weighted_gini = np.inf

        for feature in range(num_features):

            # Create sorted ordering by ascending feature values
            order = np.argsort(X[:, feature])

            # Use numpy indexing to compute sorted x and y arrays for the current feature
            x_sorted = X[order, feature]
            y_sorted = Y[order]

            # cum_ones[i] = count of class 1 in positions 0...i
            cumulative_ones = np.cumsum(y_sorted)
            total_ones = cumulative_ones[-1]

            # For split between positions i and i+1: left = 0..i, right = i+1..n-1
            n_left  = np.arange(1, n)
            n_right = n - n_left

            # Remove last item since you can't split after the last position
            ones_left_of_split  = cumulative_ones[:-1]

            # One right of split is just total ones minus ones left of split
            ones_right_of_split = total_ones - ones_left_of_split

            # Compute  Gini for all split positions at once — each index i represents splitting after
            # sorted position i, with ones_left_of_split/ones_right_of_split giving class counts on each side
            weighted_gini = self.binary_gini_vectorized(ones_left_of_split, ones_right_of_split, n_left, n_right, n)

            # Remove splits with identical adjacent features so that we don't split duplicate values apart
            valid = x_sorted[:-1] != x_sorted[1:]
            weighted_gini = np.where(valid, weighted_gini, np.inf)

            # Find split position that minimizes weighted gini
            pos = np.argmin(weighted_gini)
            if weighted_gini[pos] < best_weighted_gini:
                best_weighted_gini = weighted_gini[pos]
                best_split["feature"] = feature
                best_split["threshold"] = x_sorted[pos]
                best_split["info_gain"] = parent_gini - best_weighted_gini

        # Split dataset if a best split found
        if best_split["info_gain"] > 0:
            left, right = self.split(dataset, best_split["feature"], best_split["threshold"])
            best_split["left_dataset"] = left
            best_split["right_dataset"] = right

        return best_split

    def split(self, dataset, feature, value):
        """Splits the dataset at a given feature by a value"""

        condition = dataset[:, feature] <= value
        left = dataset[condition]
        right = dataset[~condition]

        return left, right

    def calculate_leaf_value(self, Y):
        """Outputs the most common class seen in a leaf node dataset"""

        values, counts = np.unique(Y, return_counts=True)
        most_common = values[np.argmax(counts)]

        return most_common

    def gini(self, y):
        """Calculates Gini Impurity of an array of class labels. Lower impurity indicates higher quality classification"""

        num_classes = np.unique(y)
        sum = 0
        for class_ in num_classes:
            class_probability = len(y[y == class_]) / len(y)
            sum += class_probability**2

        gini = 1 - sum
        return gini
    
    def binary_gini(self, y):
        """
            Calculates Gini Impurity of an array of binary class labels. Optimized for binary classification
            Increases gini processing time by over 700%
        """

        prob_1 = y.sum() / len(y) 
        prob_0 = 1 - prob_1
        gini = 1 - (prob_0**2 + prob_1**2)
    
        return gini
    
    def binary_gini_vectorized(self, ones_left, ones_right, n_left, n_right, n):
        """Computes weighted Gini for all split positions at once given precomputed left/right class counts."""

        p_left  = ones_left  / n_left
        p_right = ones_right / n_right

        gini_left  = 2 * p_left  * (1 - p_left)
        gini_right = 2 * p_right * (1 - p_right)

        return (n_left * gini_left + n_right * gini_right) / n

    def information_gain(self, parent, left, right):
        """Calculates weighted information gain by splitting parent into left and right"""

        # Calculate weight proportional to parent
        weight_left = len(left) / len(parent)
        weight_right = len(right) / len(parent)

        # Calculate weighted information gain by gini impurity
        info_gain = self.binary_gini(parent) - (
            (weight_left * self.binary_gini(left)) + (weight_right * self.binary_gini(right))
        )

        return info_gain


if __name__ == "__main__":
    # Demo decision tree on penguin dataset

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

    tree = DecisionTree(min_samples=2, max_depth=10)
    tree.fit(X_train, Y_train)

    Y_pred = tree.forward(X_val)
    accuracy = 0

    for i in range(len(Y_pred)):
        if Y_pred[i] == Y_val[i]:
            accuracy += 1

    accuracy = accuracy / len(Y_pred)
    print(accuracy)

    # tree.print_tree()