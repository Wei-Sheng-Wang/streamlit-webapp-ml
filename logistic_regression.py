import numpy as np
from gradient_descent import GradientDescent

import streamlit as st


class LogisticRegression:
    def __init__(self, X):
        self.d, self.n = X.shape[1], X.shape[0]
        # binary classifier, two classes, set to -1 and 1
        self.num_classes = 2
        self.weights = []
        self.mean_train = X.apply(lambda x: x.mean())
        self.std_train = X.apply(lambda x: x.std())


    def fit(self, X, y, regularization,  classifier):

        X = self.standardize(X)
        gd = GradientDescent()
        self.weights = gd.minimization(X=X, y=y, d=self.d, regularization=regularization, classifier=classifier)
        st.write(self.weights)

    def predict(self, X_test):
        X_test = self.standardize(X_test)
        # sigmoid function
        y_pred = X_test.apply(lambda x_i: 1 if (1 / (1 + (np.exp(- np.dot(self.weights, x_i))))) >= 0.5 else 0, axis=1)
        return y_pred

    # can only standardize/normalize using mean and std of training data
    def standardize(self, X):
        return (X - self.mean_train) / self.std_train


