from app import *


def main():
    st.title('Binary Classification Web App')
    st.sidebar.title('Binary Classification Web App')
    st.markdown('Do you have breast cancer? ')
    st.sidebar.markdown('Do you have breast cancer? ')

    df = load_data('./data.csv')
    if st.sidebar.checkbox('Show raw data', False):
        st.subheader('Breast Cancer Data Set')
        st.write(df)

    x_train, x_test, y_train, y_test = split_data(df)
    class_names = ['malignant', 'benign']
    st.sidebar.subheader('Choose Classifier')
    classifier = st.sidebar.selectbox('Classifier', ('Support Vector Machine (SVM)', 'Logistic Regression',
                                                     'Naive Bayes'))
    choose_classifier(classifier, x_train, y_train, x_test, y_test, class_names)
    # train_naive_bayes(X, y)




if __name__ == '__main__':
    main()