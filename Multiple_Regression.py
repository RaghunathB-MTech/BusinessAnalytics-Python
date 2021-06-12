# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 12:51:00 2021

@author: Raghunath B
"""

# Multiple Linear Regression

#importing the libraries
import pandas as pd

#Load the dataset
data = pd.read_csv('D:/Analytics/projects/Lecture programs/2. Multiple regression/stud_reg_2.csv')
data

x = data.iloc[:,[0,1]].values
y = data.iloc[:,[2]].values

#Splitting the dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=1)

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

#Final predictive modelling equation
#APPLICANTS = (-127.74 * PLACE_RATE) + (0.6169 * NO_GRAD_STUD) + 3442.62


