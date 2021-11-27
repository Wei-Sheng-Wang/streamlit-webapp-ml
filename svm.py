import numpy as np
import streamlit as st

from gradient_descent import GradientDescent

class SVM:
    def __init__(self, X):
        self.d, self.n = X.shape[1], X.shape[0]
        self.num_classes = 2
        self.weights = []
        self.mean_train = X.apply(lambda x: x.mean())
        self.std_train = X.apply(lambda x: x.std())

    def fit(self, X, y, classifier, regularization):
        X = self.standardize(X)
        X['diagnosis'] = y.apply(lambda label: label - 1 if label == 0 else 1)
        gd = GradientDescent()
        self.weights = gd.minimization(X=X, y=y, d=self.d, regularization=regularization, classifier=classifier)


    def predict(self, X_test):
        X_test = self.standardize(X_test)
        y_pred = X_test.apply(lambda x: np.sign(np.dot(self.weights, x)) + 1 if np.sign(np.dot(self.weights, x)) == -1 else 1, axis=1)

        return y_pred
    # can only standardize/normalize using mean and std of training data
    def standardize(self, X):
        return (X - self.mean_train) / self.std_train



