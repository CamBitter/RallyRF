import numpy  as np
import pandas as pd

class DecisionTreeClassifier:

    def __init__(self):
        pass


    def gini(self, y):
        '''
        Calculates Gini Impurity of an array of class labels
        Lower impurity indicates higher quality classification
        '''

        num_classes = np.unique(y)
        sum = 0
        for class_ in num_classes:
            class_probability = len(y[y == class_]) / len(y)
            sum += class_probability ** 2

        gini = 1 - sum
        return gini


    def information_gain(self, parent, left, right):
        '''Calculates weighted information gain by splitting parent into left and right'''

        # Calculate weight proportional to parent
        weight_left = len(left) / len(parent)
        weight_right = len(right) / len(parent)

        # Calculate weighted information gain by gini impurity
        info_gain = self.gini(parent) - ((weight_left * self.gini(left)) + (weight_right * self.gini(right)))
        
        return info_gain