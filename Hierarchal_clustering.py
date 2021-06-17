# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 15:51:41 2021

@author: Raghunath B
"""

#Project: Customer segmentation
#Hierarchial clustering

#Load the dataset
import pandas as pd
df = pd.read_csv('D:/Analytics/projects/Lecture programs/8. Hierarchial clustering/Mall_Customers.csv')

x = df.iloc[:,[3,4]].values

#Dendogram - To find the optimum no. of clusters
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(x,method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
#Optimum no. of clusters(k)=5 since a horizontal line cuts 5 vertical lines without meeting any intersections

#Fit hierarchial clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc = hc.fit_predict(x)

# Visualising the clusters
plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s = 100, c = 'cyan', label = 'Cluster 3')
plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s = 100, c = 'green', label = 'Cluster 4')
plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()

#Cluster 1: High Income but Low SS --> People who seek offers
#Cluster 2: Medium Income and Spending Score --> Regular customers
#Cluster 3: High Income and High SS --> Elite buyers
#CLuster 4: Low Income and High SS --> Lavish spenders
#Cluster 5: Low Income and SS --> Poor buyers







