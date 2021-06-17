# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 20:18:06 2021

@author: Raghunath B
"""

#Project: Predict the profit from a startup based on the costs and location
# Multiple Regression with Backward Elimination

#Importing the dataset
import pandas as pd

#Loading the dataset
data = pd.read_csv('D:/Analytics/projects/Lecture programs/3. Multiple reg with backward elimination/50_Startups.csv')
data
data.info()

#Check for missing values
data.isnull().sum()

#Handling categorical variable(State)
data['State'].unique()
s_dummies = pd.get_dummies(data['State'],drop_first=True)

data = pd.concat([data,s_dummies],axis=1)
data.drop(['State'], axis=1, inplace=True)

#Splitting the dataset
x = data.iloc[:,[0,1,2,4,5]].values
y = data.iloc[:,[3]].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#Fitting Linear regression to the training set
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train,y_train)

#Predicting the Test set results
y_pred = reg.predict(x_test)

#Accuracy of the model
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

#Backward elimination
import numpy as np
import statsmodels.api as sm
x = np.append(arr = np.ones((len(x), 1)).astype(int), values = x, axis = 1)
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
SL = 0.05
x_opt = x[:, [0, 1, 2, 3, 4, 5]]
x_Modeled = backwardElimination(x_opt, SL)
#Only R&D cost is the significant variable

#Splitting the dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_Modeled, y, test_size=0.2, random_state=0)

#Fitting Linear regression to the training set
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train,y_train)

#Predicting the Test set results
y_pred = reg.predict(x_test)

#Accuracy of the model
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

#Calculating co-efficient and intercept
print(reg.coef_)
print(reg.intercept_)

#Final predictive modelling equation
#Profit = (0.8516 * R&D Spend) + 48416.29



























