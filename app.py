# cmd to run web app in anaconda prompt : streamlit run app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve

from sklearn.metrics import precision_score, recall_score

def main():
    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Are ur mushrooms edible or not ? ")
    st.sidebar.markdown("Are ur mushrooms edible or not ? ")

    @st.cache(persist=True)
    def load_data():
        data = pd.read_csv('mushrooms.csv')
        data.rename(columns={'class': 'type'}, inplace=True)
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data

    @st.cache(persist=True)
    def split(df):
        y = df.type
        x = df.drop(columns=['type'])
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=0)
        return x_train, x_test, y_train, y_test

    # def plot_metrics(metrics_list):
    #
    #     if 'Confusion Matrix' in metrics_list:
    #         st.subheader("Confusion Matrix")
    #         plot_confusion_matrix(model, x_test, y_test, display_labels = class_names)
    #         st.pyplot()
    #
    #     if 'ROC Curve' in metrics_list:
    #         st.subheader("ROC Curve")
    #         plot_roc_curve(model, x_test, y_test)
    #         st.pyplot()
    #
    #     if 'Precision Recall Curve' in metrics_list:
    #         st.subheader("Precison Recall Curve")
    #         plot_precision_call(model, x_test, y_test)
    #         st.pyplot()

    df = load_data()

    x_train, x_test, y_train, y_test = split(df)
    class_names = ['edible','poisonous']
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier",("SVM","Log Reg","Random Forest"))

    if classifier=='SVM':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Reg param)", 0.01, 10.0, step=0.01, key='C')
        kernel = st.sidebar.radio("Kernel", ("rbf","linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma(Kernel coeff)",("Scale","auto"),key='gamma')

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("SVM results")

            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train,y_train)
            acc = model.score(x_test, y_test)
            y_pred = model.predict(x_test)

            st.write("Acc : ",acc.round(2))
            st.write("Precision : ",precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("recall : ", recall_score(y_test, y_pred, labels=class_names).round(2))

    if classifier=='Log Reg':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Reg param)", 0.01, 10.0, step=0.01, key='C_lr')
        max_iter = st.sidebar.slider("Max iterations ", 100, 500, key='iter')

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Log Reg results")

            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(x_train,y_train)
            acc = model.score(x_test, y_test)
            y_pred = model.predict(x_test)

            st.write("Acc : ",acc.round(2))
            st.write("Precision : ",precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("recall : ", recall_score(y_test, y_pred, labels=class_names).round(2))


    if classifier=='Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimator = st.sidebar.number_input("No of trees in forest are ", 100, 5000, step=10, key='n_estimator')
        max_depth = st.sidebar.number_input("Max depth of tree ", 1, 20, step=1, key='max_depth')
        bootstrap = st.sidebar.radio("Bootstrap samples", ("True","False"), key='bootstrap')

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Random Forest results")

            model = RandomForestClassifier(n_estimators=n_estimator, max_depth=max_depth, bootstrap=bootstrap)
            model.fit(x_train,y_train)
            acc = model.score(x_test, y_test)
            y_pred = model.predict(x_test)

            st.write("Acc : ",acc.round(2))
            st.write("Precision : ",precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("recall : ", recall_score(y_test, y_pred, labels=class_names).round(2))

    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushrooms data set ")
        st.write(df)
if __name__ == '__main__':
    main()
print("Happily Done !")
