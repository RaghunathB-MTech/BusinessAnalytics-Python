# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 08:53:42 2021

@author: Raghunath B
"""

#Project: Predict wheteher a person survived or not survived the titanic crash based on various factors
#Logistic regression

#Loading the dataset
import pandas as pd
data = pd.read_csv('D:/Analytics/projects/Lecture programs/4. Logistic regression/titanic.csv')
len(data.index)

#Checking datatype
data.info()
data['age'] = pd.to_numeric(data['age'],errors='coerce')
data['fare'] = pd.to_numeric(data['fare'],errors='coerce')
data.info()

#Handling missing values
data.isnull().sum()

#Drop the rows having missing values in fare column since it has very less missing values
data.dropna(subset=['fare'],inplace=True)

#Plot the age column to check for outliers before doing imputation on missing rows
import seaborn as sns 
sns.boxplot(x='age', data=data)

#We can go for mean imputation since there are less number of outliers
data['age'].fillna(data['age'].mean(),inplace=True)
data.isnull().sum()

#Handling categorical variables
data['pclass'].unique()
pclass_dummy = pd.get_dummies(data['pclass'],drop_first=True)

sex_dummy = pd.get_dummies(data['sex'],drop_first=True)

data['embarked'].unique() #? in this column can be treated as a seperate category(New category Imputation)
embarked_dummy = pd.get_dummies(data['embarked'],drop_first=True)

data = pd.concat([data,pclass_dummy,sex_dummy,embarked_dummy],axis=1)
data.drop(['Passenger_id','name','sex','ticket','embarked','pclass'],axis=1,inplace=True)

#Seperating the variables
x = data.drop(['survived'],axis=1).values
y = data['survived'].values

#Splitting the dataset
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.25, random_state=1)

#Fitting the model to the training data
from sklearn.linear_model import LogisticRegression
reg = LogisticRegression()
reg.fit(x_train,y_train)

#Predict for test data
y_pred = reg.predict(x_test)

#Accuracy of model using confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)
#Accuracy = (165+95)\(165+95+30+33) = 80.495%

#Backward elimination - To obtain list of significant independent variables
#Add a column with all values as 1 to the x 
import numpy as np
x1 = np.append(arr=np.ones((1291,1),dtype=int),values=x,axis=1)

x_opt = x1[:,[0,1,2,3,4,5,6,7,8,9,10]]

#Fit the model by creating a regressor_OLS object of new class OLS of statsmodels library.
import statsmodels.api as sm

reg_OLS = sm.OLS(endog=y,exog=x_opt).fit()
reg_OLS.summary()

#Remove x3 since it has max. p-value & p>=0.05
x_opt = x1[:,[0,1,2,4,5,6,7,8,9,10]]

reg_OLS = sm.OLS(endog=y,exog=x_opt).fit()
reg_OLS.summary()

x_opt = x1[:,[0,1,2,4,5,6,7,9,10]]

reg_OLS = sm.OLS(endog=y,exog=x_opt).fit()
reg_OLS.summary()

x_opt = x1[:,[0,1,2,5,6,7,9,10]]

reg_OLS = sm.OLS(endog=y,exog=x_opt).fit()
reg_OLS.summary()

x_opt = x1[:,[0,1,2,5,6,7,10]]

reg_OLS = sm.OLS(endog=y,exog=x_opt).fit()
reg_OLS.summary()
#All varaibles have p<=0.05.
#Set of significant variables - age,sibsp,pclass,sex,embarked.

#Splitting the dataset
from sklearn.model_selection import train_test_split 
x_BE_train,x_BE_test,y_BE_train,y_BE_test = train_test_split(x_opt,y, test_size=0.25, random_state=1)

#Fitting the model to the training data
from sklearn.linear_model import LogisticRegression
reg_BE = LogisticRegression()
reg_BE.fit(x_BE_train,y_BE_train)

#Predict for test data
y_BE_pred = reg_BE.predict(x_BE_test)

#Accuracy of model using confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_BE_test, y_BE_pred)
#Accuracy = (168+96)/(168+96+27+32) = 81.73%

#Co-eff & Intercept
reg_BE.coef_

reg_BE.intercept_

#Final predictive model equation is
#Survived
#exp( 2.58 -0.027*age -0.31*sibsp -1.02*pclass(2) -1.88*pclass(3) -2.36*sex(male) -0.56*embarked(S))
# /
#1 + exp( 2.58 -0.027*age -0.31*sibsp -1.02*pclass(2) -1.88*pclass(3) -2.36*sex(male) -0.56*embarked(S))
