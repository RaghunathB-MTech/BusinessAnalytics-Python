# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 07:27:59 2021

@author: Raghunath B
"""

#Analyze the Attrition rate of Employees

#Load the dataset
import pandas as pd
df = pd.read_csv('D:/Analytics/projects/Live project/Attrition rate/HR - Attrition Data - Data Set.csv')

#DATA PREPARATION--------------------------------------
#Check for data type
df.info()

#Check for missing values
df.isnull().sum()
#No missing values found

#Handling Categorical variables
#Attrition
df['Attrition'].unique()
Attrition_dummy = pd.get_dummies(df['Attrition'],drop_first=True)
Attrition_dummy.rename(columns={'Yes':'Attr'},inplace=True)

#BusinessTravel
df['BusinessTravel'].unique()
BusinessTravel_dummy = pd.get_dummies(df['BusinessTravel'],drop_first=True)

#Department
df['Department'].unique()
Department_dummy = pd.get_dummies(df['Department'],drop_first=True)

#EducationField
df['EducationField'].unique()
EducationField_dummy = pd.get_dummies(df['EducationField'],drop_first=True)

#Gender
df['Gender'].unique()
Gender_dummy = pd.get_dummies(df['Gender'],drop_first=True)

#JobRole
df['JobRole'].unique()
JobRole_dummy = pd.get_dummies(df['JobRole'],drop_first=True)

#MaritalStatus
df['MaritalStatus'].unique()
MaritalStatus_dummy = pd.get_dummies(df['MaritalStatus'],drop_first=True)

#OverTime
df['OverTime'].unique()
OverTime_dummy = pd.get_dummies(df['OverTime'],drop_first=True)

df = pd.concat([df,Attrition_dummy,BusinessTravel_dummy,Department_dummy,EducationField_dummy,Gender_dummy,JobRole_dummy,MaritalStatus_dummy,OverTime_dummy],axis=1)
df.drop(['Attrition','BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','OverTime'],axis=1,inplace=True)

#DATA ANALYSIS--------------------------------------------
#Backward Elimination - To uncover the factors that lead to employee attrition.

#Seperating the dependent and independent variable
x = df.drop(['Attr'],axis=1).values
y = df['Attr'].values

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
x_opt = x

x_Modeled = backwardElimination(x_opt, SL)
#Significant variables(Significance Level = 5%) - Age, DistanceFromHome, EnvironmentSatisfaction,
#JobInvolvement, JobSatisfaction, NumCompaniesWorked, RelationshipSatisfaction,
#TotalWorkingYears, WorkLifeBalance, YearsInCurrentRole, YearsSinceLastPromotion,
#BusinessTravel(Travel_Frequently,Travel_Rarely), EducationField(Life Sciences,Medical,Other),
#Gender(Male), JobRole(Laboratory Technician,Sales Representative),
#MaritalStatus(Single), OverTime(Yes)

















