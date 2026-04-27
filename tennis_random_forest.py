from random_forest import RandomForestClassifier
from decision_tree import DecisionTree
import numpy as np 
import pandas as pd

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from data import getDF

    df = getDF()

    ## build features/labels

    ## make train/test split

    ## forest = RandomForestClassifier(num_trees=, num_features=, max_depth=, random_state=)
    ## forest.fit(X_train, y_train)

    ## accuracy check