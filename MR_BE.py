# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 20:18:06 2021

@author: Raghunath B
"""

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
x = np.append(np.ones([50,1]), values=x, axis=1)
x_opt = x[:,[0,1,2,3,4,5]]

import statsmodels.api as sm
reg_OLS = sm.OLS(endog=y, exog=x_opt).fit()
reg_OLS.summary()

#Remove 5th variable(Newyork) since it has very high p-value & p>=0.05
x_opt = x[:,[0,1,2,3,4]]

reg_OLS = sm.OLS(endog=y, exog=x_opt).fit()
reg_OLS.summary()

#Remove 4th variable(Florida) since it has very high p-value & p>=0.05
x_opt = x[:,[0,1,2,3]]

reg_OLS = sm.OLS(endog=y, exog=x_opt).fit()
reg_OLS.summary()

#Remove 2nd variable(Admin cost) since it has very high p-value & p>=0.05
x_opt = x[:,[0,1,3]]

reg_OLS = sm.OLS(endog=y, exog=x_opt).fit()
reg_OLS.summary()

#Remove 3rd variable(Marketing cost) since it has very high p-value & p>=0.05
x_opt = x[:,[0,1]]

reg_OLS = sm.OLS(endog=y, exog=x_opt).fit()
reg_OLS.summary()
#Only R&D cost is the significant variable

#Splitting the dataset
x = data.iloc[:,:-5].values
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

#Calculating co-efficient and intercept
print(reg.coef_)
print(reg.intercept_)

#Final predictive modelling equation
#Profit = (0.8516 * R&D Spend) + 48416.29



























