# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 11:24:40 2021

@author: Raghunath B
"""

#Project: Predict the no. of applications based on the placement rate
#Simple Linear Regression

#Importing Libraries and dataset
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('D:/Analytics/projects/Lecture programs/1. Linear Regression/stud_reg.csv')
data

x = data.iloc[:,:-1].values
y = data.iloc[:, 1].values

#Splitting the dataset
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

#Fitting Linear regression to the training set
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train,y_train)

#Predicting the Test set results
y_pred = reg.predict(x_test)

#Co-efiicient and Intecept
print(reg.coef_)
print(reg.intercept_)

#Accuracy of the model
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

#Visualizing the predicted result
plt.plot(x_test, y_test, '--', color='blue')
plt.plot(x_test, y_pred, ':', color='red')

#Final predictive modelling equation
#APPLICANTS = (-192.1 * PLACE_RATE) + 15016.89
