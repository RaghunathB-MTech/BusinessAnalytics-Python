# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 18:00:10 2021

@author: Raghunath B
"""

#Project: Predict whether a customer will buy a car based on his/her age, gender and Income
#Decision Tree

#Import libraries
import pandas as pd

#Load the dataset
df = pd.read_csv('D:/Analytics/projects/Lecture programs/5. Decision tree/Purchase_History.csv')

#Handling categorical variables
Gender_dummy = pd.get_dummies(df['Gender'],drop_first=True)
df = pd.concat([df,Gender_dummy],axis=1)
df.drop('Gender',axis=1,inplace=True)

#Split the dataset
x = df.drop(['Purchased','User ID'],axis=1)
y = df['Purchased']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

#Fit the classifier to the training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy',max_depth = 3, min_samples_leaf=5)
classifier.fit(x_train,y_train)

#Predict results for test set
y_pred = classifier.predict(x_test)

#Accuracy of model
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))
#Accuracy = (64+30)/(64+4+2+30) = 94%

#Decision tree visualization
from sklearn import tree
tree.plot_tree(classifier)
#Contents in the tree are blurred

import matplotlib.pyplot as plt

fig = plt.subplots(nrows=1,ncols=1,figsize=(4,4),dpi=300)
cn=['0','1']
tree.plot_tree(classifier,class_names=cn,filled=True)

#To save the tree
fig.savefig('DT.png')











