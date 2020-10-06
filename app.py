import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

def main():
    
    st.title('Binary Classification Web App')
    st.sidebar.title('Binary Classification Web App')
    st.markdown("Are your mushrooms edible or poisonous? 🍄 ")
    st.sidebar.markdown("Are your mushrooms edible or poisonous? 🍄 ")
    
    @st.cache(persist = True)
    def load_data(encode = False):
        data = pd.read_csv('D:\ML\portfolio\ML_Web_App/mushrooms.csv')
        if(encode):
            label = LabelEncoder()
            for col in data.columns:
                data[col] = label.fit_transform(data[col])
            return data
        else:
            return data
    
    @st.cache(persist = True)
    def split(df):
        y = df.type
        x = df.drop(columns = ['type'])
        xtrain, xtest , ytrain, ytest = train_test_split(x,y, test_size = .3, random_state = 0)
        return xtrain, xtest , ytrain, ytest
    
    
    def plot_metrics(metrics_list):
        if 'Confusion matrix' in metrics_list:
            st.subheader('Confusion Matrix')
            plot_confusion_matrix(model, xtest,ytest, display_labels = class_names)
            st.pyplot()
            
        if 'ROC Curve' in metrics_list:
            st.subheader('ROC Curve')
            plot_roc_curve(model, xtest,ytest)
            st.pyplot()
            
        if 'Percision-Recall Curve' in metrics_list:
            st.subheader('Percision-Recall Curve')
            plot_precision_recall_curve(model, xtest,ytest)
            st.pyplot()
    
    df = load_data(encode = True)
    xtrain, xtest , ytrain, ytest = split(df)
    class_names = ['edible', 'poisonous']
    st.sidebar.subheader('Choose Classifier')
    classifier = st.sidebar.selectbox('Classifier', ['Support Vector Machine (SVM)',
                                                     'Logistic Regression',
                                                    'Random Forest'])
    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader('Model Hyperparameters')
        C = st.sidebar.number_input("C (Regularization parameter ), Range = .01 to 10 ",
                                    .01, 10.0,step = .01)
        kernel = st.sidebar.radio("Kernel", ('rbf', 'linear'))
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient) ", ("scale", "auto"))
    
        metrics = st.sidebar.multiselect("What metrics do you want to plot?", 
                                         ['Confusion matrix', 'ROC Curve',
                                          'Percision-Recall Curve'])
        #if st.sidebar.button('Classify'):
        st.subheader('Support Vector Machine (SVM) Results')
        model = SVC(C = C , kernel = kernel, gamma = gamma)
        model.fit(xtrain , ytrain)
        accuracy = model.score(xtest, ytest)
        ypred = model.predict(xtest)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Percision : ", precision_score(ytest, ypred, labels = class_names).round(2))
        st.write("Recall : ", recall_score(ytest, ypred, labels = class_names).round(2))
        plot_metrics(metrics)
        
    
    if classifier == 'Logistic Regression':
        st.sidebar.subheader('Model Hyperparameters')
        C = st.sidebar.number_input("C (Regularization parameter ), Range = .01 to 10 ",
                                    .01, 10.0,step = .01)
        max_iter = st.sidebar.slider("Maximum number of iterations", 100,500)
    
        metrics = st.sidebar.multiselect("What metrics do you want to plot?", 
                                         ['Confusion matrix', 'ROC Curve',
                                          'Percision-Recall Curve'])
        #if st.sidebar.button('Classify'):
        st.subheader('Logistic Regression Results')
        model = LogisticRegression(C = C , max_iter = max_iter)
        model.fit(xtrain , ytrain)
        accuracy = model.score(xtest, ytest)
        ypred = model.predict(xtest)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Percision : ", precision_score(ytest, ypred, labels = class_names).round(2))
        st.write("Recall : ", recall_score(ytest, ypred, labels = class_names).round(2))
        plot_metrics(metrics)
        
    if classifier == 'Random Forest':
        st.sidebar.subheader('Model Hyperparameters')
        n_estimators = st.sidebar.number_input("Number of the trees", 100, 5000, step = 10)
        max_depth = st.sidebar.number_input("Maximum depth of the tree", 1, 20, step = 1)
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ['True', 'False'])
        
        metrics = st.sidebar.multiselect("What metrics do you want to plot?", 
                                         ['Confusion matrix', 'ROC Curve',
                                          'Percision-Recall Curve'])
        #if st.sidebar.button('Classify'):
        st.subheader('Random Forest Results')
        model = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, 
                                       bootstrap = bootstrap)
        model.fit(xtrain , ytrain)
        accuracy = model.score(xtest, ytest)
        ypred = model.predict(xtest)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Percision : ", precision_score(ytest, ypred, labels = class_names).round(2))
        st.write("Recall : ", recall_score(ytest, ypred, labels = class_names).round(2))
        plot_metrics(metrics)
    
    
    
    if st.sidebar.checkbox("Show raw data",False):
        df = load_data()
        st.subheader("Mushroom Data Set for Classification")
        st.write(df)
    
    if st.sidebar.checkbox("Label Encode data",False):
        df = load_data(encode = True)
        st.subheader("Encoded Mushroom Data Set for Classification")
        st.write(df)
if __name__ == '__main__':
    main()


