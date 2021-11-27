import numpy as np
import math
import pandas as pd
import streamlit as st
import random as random

# Gradient descent for logistic regression log loss function
class GradientDescent:
    def __init__(self):
        self.eps = 0.4

    # wt is a list of weights
    # x is
    def minimization(self, X, y, d, regularization, classifier):
        # d is the number of features
        # self.setEps(regularization)
        y = y.apply(lambda x: x - 1 if x == 0 else 1)
        X['diagnosis'] = y
        w_t1 = [0.001 if i % 2 == 0 else -0.001 for i in range(d)]
        gradient = self.matrixDerivative(X, w_t1, regularization, classifier)

        l2_norm = np.linalg.norm(gradient)
        i = 0.01
        learning_rate = 0.0001 / np.sqrt(i)
        while l2_norm > self.eps:
            w_t1 = np.subtract(w_t1, np.multiply(learning_rate, gradient))
            gradient = self.matrixDerivative(X, w_t1, regularization, classifier)
            i += 1
            l2_norm = np.linalg.norm(gradient)

            print(l2_norm)



        return w_t1


    def differentiate(self, x, w, classifier):
        if classifier == 'Logistic Regression':
            return self.differentiate_log_loss(x, w)
        elif 'SVM' in classifier:
            return  self.differentiate_svm(x, w)


    def differentiate_log_loss(self, x, w):
        y = x['diagnosis'].iloc[0]

        x = x.iloc[:,:-1].iloc[0]
        dot = np.dot(w, x)
        pred_label = -y * dot
        if pred_label >= 700:
            return - y * x

        exponent = np.exp(pred_label)
        numerator = -y * x * exponent
        denominator = 1 + exponent
        return np.true_divide(numerator, denominator)



    def matrixDerivative(self, X, w, regularization, classifier):
        if regularization == 'None':
            return self.differentiate(X.sample(), w, classifier)

        # L1 norm of vector is L2 norm
        elif regularization == 'LASSO':
            vectorized_huber_loss = np.vectorize(self.huber_loss)
            derivative_L1_regularization = vectorized_huber_loss(w)
            return np.add(self.differentiate(X.sample(), w, classifier), 2 * derivative_L1_regularization)

        elif regularization == 'Ridge':
            return np.add(self.differentiate(X.sample(), w, classifier), 2 * np.array(w))


    def huber_loss(self, w_j):
        return w_j if abs(w_j) <= 0.6 else 0.6 * np.sign(w_j)



    def differentiate_svm(self, x, w):
        y = x['diagnosis'].iloc[0]
        x = x.iloc[:,:-1].iloc[0]
        dot_product = np.dot(w, x)
        # print(list(here))
        return np.zeros(len(x)) if y * dot_product > 1 else - y * x


    def logistic_objective_function(self, X, w, regularization):
        regularization = self.regularization_term(regularization, w)
        return np.add(X.apply(lambda x: np.log(1 + np.exp(-x['diagnosis'] * np.dot(w, x))), axis=1), regularization)

    # def svm_objective_function(self, X, w, regularization):
    #     regularization = self.regularization_term(regularization, w)
    #     return np.add(X.apply(lambda x: np.max(0, 1-), axis=1), regularization)
    #


    def regularization_term(self, regularization, w):
        if regularization == 'None':
            return np.zeros(len(w))
        elif regularization == 'LASSO':
            return 2 * np.abs(w)
        elif regularization == 'Ridge':
            return np.linalg.norm(w)


