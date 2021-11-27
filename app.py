import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay,  RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import precision_score, recall_score, accuracy_score
from classifier_implementation import NaiveBayes
import matplotlib.pyplot as plt

from logistic_regression import LogisticRegression
from svm import SVM

# preprocess data
@st.cache(persist=True)
def load_data(data):
    df = pd.read_csv(data)

    df['diagnosis'] = LabelEncoder().fit_transform(df['diagnosis'])
    df = df.drop("Unnamed: 32", axis=1)
    df = df.drop('id', axis=1)
    return df


@st.cache(persist=True)
def split_data(df):
    y = df.diagnosis
    x = df.drop(columns=['diagnosis'])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    return x_train, x_test, y_train, y_test

# preprocess data


def plot_metrics(metrics_list, y_test, y_predicted, class_names, classifier_name):

    if 'Confusion Matrix' in metrics_list:
        st.subheader('Confusion Matrix')
        ConfusionMatrixDisplay.from_predictions(y_true=y_test, y_pred=y_predicted, display_labels=class_names)
        st.pyplot(plt)

    if 'ROC Curve' in metrics_list:
        st.subheader('ROC Curve')
        RocCurveDisplay.from_predictions(y_true=y_test, y_pred=y_predicted, name=classifier_name)
        st.pyplot(plt)

    if 'Precision-Recall Curve' in metrics_list:
        st.subheader('Precision-Recall Curve')
        PrecisionRecallDisplay.from_predictions(y_true=y_test, y_pred=y_predicted, name=classifier_name)
        st.pyplot(plt)


# choose classifier options
def choose_classifier(classifier, X, y, X_test, y_test, class_names):

    if classifier == 'Support Vector Machine (SVM)':
        # st.sidebar.subheader('Model Hyperparameter')
        # C = st.sidebar.number_input('λ (Regularization Parameter)', 0.01, 10.0, step=0.01, key='λ')
        # kernel = st.sidebar.radio('kernel', ('rbf', 'linear'), key='kernel')
        # gamma = st.sidebar.radio('Gamma (Kernel Coefficient)', ('scale', 'auto'), key='gamma')

        metrics = st.sidebar.multiselect('What metrics to plot?', ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
        if st.sidebar.button('Classify', key='classify'):
            st.subheader('Support Vector Machine (SVM)')
            svm_classifier = SVM(X)
            svm_classifier.fit(X, y, classifier, regularization='Ridge')
            predicted_labels = svm_classifier.predict(X_test)

            display_metrics(predicted_labels, y_test, class_names)
            plot_metrics(metrics, y_test, predicted_labels, class_names, 'Support Vector Machine (SVM)')



    if classifier == 'Naive Bayes':
        metrics = st.sidebar.multiselect('What metrics to plot?', ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
        if st.sidebar.button('Classify', key='classify'):

            st.subheader('Naive Bayes Results')
            NBclassifier = NaiveBayes(X)
            NBclassifier.fit(X, y)
            predicted_labels = NBclassifier.predict(X_test)

            display_metrics(predicted_labels, y_test, class_names)
            plot_metrics(metrics, y_test, predicted_labels, class_names, 'Gaussian Naive Bayes')


    elif classifier == 'Logistic Regression':
        regularization = st.sidebar.selectbox('Regularization', ('None', 'LASSO', 'Ridge'), key='regularization')

        metrics = st.sidebar.multiselect('What metrics to plot?',
                                         ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button('Classify', key='classify'):
            st.subheader('Logistic Regression Results')

            logistic_regression = LogisticRegression(X)
            logistic_regression.fit(X, y, regularization, classifier)
            y_pred = logistic_regression.predict(X_test)

            display_metrics(y_pred, y_test, class_names)
            plot_metrics(metrics, y_test, y_pred, class_names, 'Logistic Regression')




def display_metrics(predicted_labels, y_test, class_names):

    # st.write(X_test)

    accuracy = accuracy_score(y_test, predicted_labels)
    precision = precision_score(y_test, predicted_labels, labels=class_names)
    recall = recall_score(y_test, predicted_labels, labels=class_names)
    st.write('Accuracy: ', np.round(accuracy, 4))
    st.write('Precision: ', np.round(precision, 4))
    st.write('Recall: ', np.round(recall, 4))

