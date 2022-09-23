#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


data= pd.read_csv(r'C:\Users\masan\Downloads\Mall_Customers.csv')
x= data.iloc[:,[3,4]].values


# In[9]:


data


# In[4]:


x


# In[6]:


data.dtypes


# In[7]:


data.isna().sum()


# HIERACHIAL CLUSTERING

# In[8]:


#using the dendoram to find the optinal number of cluster

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x, method='ward'))
plt.title('dendogram')
plt.xlabel('customers')
plt.ylabel('Euclidean clusters')
plt.show()


# In[10]:


#select optimum condition

from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_hc=hc.fit_predict(x)


# In[11]:


y_hc


# In[17]:


plt.scatter(x[y_hc==0,0],x[y_hc ==0,1], s=100, c='red',label='cluster 1')
plt.scatter(x[y_hc==1,0],x[y_hc ==1,1], s=100, c='blue',label='cluster 2')
plt.scatter(x[y_hc==2,0],x[y_hc ==2,1], s=100, c='green',label='cluster 3')
plt.scatter(x[y_hc==3,0],x[y_hc ==3,1], s=100, c='yellow',label='cluster 4')
plt.scatter(x[y_hc==4,0],x[y_hc ==4,1], s=100, c='magenta',label='cluster 5')
plt.legend()
plt.xlabel('annual income')
plt.ylabel('spending score(1-100)')
plt.show()


# In[ ]:




