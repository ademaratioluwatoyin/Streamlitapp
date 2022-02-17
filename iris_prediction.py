# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 14:01:10 2022

@author: Toyin
"""

# importing useful libraries
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

# streamlit
import streamlit as st

# loading data
iris = datasets.load_iris()
data = iris.data
target = iris.target

# modelling
model = LogisticRegression(penalty = 'elasticnet', C = 1.0, solver = 'saga', l1_ratio = 0.5).fit(data, target)

#prediction
def prediction(input_data):
    input_data = np.array(input_data).reshape(1, -1)
    predicted_value = model.predict(input_data)
    if predicted_value == 0:
        return('It is Setosa')
    elif predicted_value == 1:
        return('It is Versicolor')
    else:
        return('It is Virginica')
    
def scatterplot(x,y):
    df = sns.load_dataset('iris')
    plt.figure(figsize = (5, 3))
    g = sns.relplot(x = x, y = y, data = df, kind = 'scatter', hue = 'species')
    g.set(title = x + ' and ' + y + ' by species',
          xlabel = x,
          ylabel = y)
    st.pyplot(g)
    
    
def main():
    st.header('Scatter plot')
    x = st.selectbox('Attribute 1', ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    y = st.selectbox('Attribute 2', ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    
    if st.button('Display'):
        scatterplot(x, y)
    
    st.title('Iris Prediction')
    petal_length = st.number_input('Petal Length')
    petal_width = st.number_input('Petal Width')
    sepal_length = st.number_input('Sepal Length')
    sepal_width = st.number_input('Sepal Width')
    
    our_prediction = ''
    if st.button('predict'):
        our_prediction = prediction([sepal_length, sepal_width, petal_length, petal_width])
    
    st.success(our_prediction)
    

if __name__ == '__main__':
    main()
    


