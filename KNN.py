import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# build class of KNN
class KNN(object):
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors
        
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
    def predict(self, X_test):
        y_pred = []
        for i in range(len(X_test)):
            diffMatrix = np.tile(X_test[i], (len(self.X_train), 1)) - self.X_train
            distances = np.sum(diffMatrix ** 2, axis = 1) ** 0.5
            sortedDistancesIndex = distances.argsort()
            classifier = {}
            for n in range(self.n_neighbors):
                vote = self.y_train[sortedDistancesIndex[n]]
                classifier[vote] = classifier.get(vote, 0) + 1
                
            max = 0
            prediction = 0
            for k, v in classifier.items():
                if v > max:
                    prediction = k
                    max = v
            
            y_pred.append(prediction)
            
        return(y_pred)
