import pandas as pd
import numpy as np
import math


class NaiveBayes:
    def __init__(self, X):
        self.num_features, self.num_examples = X.shape[1], X.shape[0]
        # binary classifier, only two classes
        self.num_classes = 2
        self.classes_mean = {}
        self.classes_std = {}
        self.classes_prior = {}

    def fit(self, X, y):

        for c in range(self.num_classes):
            # get rows of dataset with class c
            X_c = X[y == c]

            # dictionary of list of mean
            self.classes_mean[c] = X_c.mean()
            self.classes_std[c] = X_c.std()
            self.classes_prior[c] = len(X_c.index) / self.num_examples
        return

    # predict a vector of test examples and return a list of predicted labels
    def predict(self, X_test):
        prob = {0: 0, 1: 0}
        predicted_label = []
        for index, row in X_test.iterrows():
            for c in range(self.num_classes):

                log_likelihood = np.sum([
                    self.gaussian_formula(row[j], self.classes_mean[c][j], self.classes_std[c][j]) for j in
                    range(self.num_features)])
                prob[c] = log_likelihood * self.classes_prior[c]

            predicted_label.append(max(prob, key=prob.get))
        return predicted_label

    # apply Maximum a Posteriori (MAP) to solve conditional probability P(y_i | x_i)
    # apply log
    def gaussian_formula(self, x, mean, std):
        return np.log(1 / (std * np.sqrt(2 * np.pi))) - np.power((x - mean), 2) / (2 * std * std)
