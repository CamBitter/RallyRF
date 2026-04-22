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

        if x_feature <= node.threshold:
            return self.predict_item(node.left, x)
        else:
            return self.predict_item(node.right, x)

    def fit(self, X, Y):
        """Starts recursive building of decision tree"""

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
        """Returns the best split feature and value given a dataset based on Gini information gain."""

        best_split = {}
        max_info_gain = float("-inf")
        X = dataset[:, :-1]

        for feature in range(num_features):
            feature_values = np.unique(X[:, feature])

            for threshold in feature_values:
                left, right = self.split(dataset, feature, threshold)

                if len(left) > 0 and len(right) > 0:
                    y = dataset[:, -1]
                    left_y = left[:, -1]
                    right_y = right[:, -1]

                    info_gain = self.information_gain(y, left_y, right_y)

                    if info_gain > max_info_gain:
                        max_info_gain = info_gain
                        best_split["feature"] = feature
                        best_split["threshold"] = threshold
                        best_split["left_dataset"] = left
                        best_split["right_dataset"] = right
                        best_split["info_gain"] = info_gain

        return best_split

    def split(self, dataset, feature, value):
        """Splits the dataset at a given feature by a value"""
        
        left = np.array([row for row in dataset if row[feature] <= value])
        right = np.array([row for row in dataset if row[feature] > value])

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

    def information_gain(self, parent, left, right):
        """Calculates weighted information gain by splitting parent into left and right"""

        # Calculate weight proportional to parent
        weight_left = len(left) / len(parent)
        weight_right = len(right) / len(parent)

        # Calculate weighted information gain by gini impurity
        info_gain = self.gini(parent) - (
            (weight_left * self.gini(left)) + (weight_right * self.gini(right))
        )

        return info_gain


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    import pandas as pd

    url = "https://raw.githubusercontent.com/PhilChodrow/ml-notes/main/data/palmer-penguins/palmer-penguins.csv"
    df = pd.read_csv(url)

    X = df[["Culmen Length (mm)", "Culmen Depth (mm)"]].to_numpy()
    Y = np.vstack(df["Species"].to_numpy())

    Y_id_to_label = np.unique(Y)
    Y_label_to_id = {str(label): idx for idx, label in enumerate(Y_id_to_label)}
    Y = np.vstack(np.array([Y_label_to_id[label.item()] for label in Y]))

    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.2, random_state=99
    )

    tree = DecisionTree()
    tree.fit(X_train, Y_train)

    Y_pred = tree.forward(X_val)
    accuracy = 0

    for i in range(len(Y_pred)):
        if Y_pred[i] == Y_val[i]:
            accuracy += 1

    accuracy = accuracy / len(Y_pred)
    print(accuracy)

    tree.print_tree()
