#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

data_df= pd.read_csv(r'C:\Users\masan\Downloads\iris.csv')


# In[3]:


data_df.head()


# In[5]:


iris_df=data_df.iloc[:,1:5]
iris_df.head()


# In[6]:


iris_df.describe()


# In[8]:


#normalize

from sklearn.preprocessing import Normalizer
norm=Normalizer()
iris_df=norm.fit_transform(iris_df)


# In[12]:


from scipy.cluster.hierarchy import dendrogram,linkage
import matplotlib.pyplot as plt
linked=linkage(iris_df,method='ward')
plt.figure(figsize=(15,8))
dendrogram(linked,orientation='top',distance_sort='ascending',show_leaf_counts=True)
plt.hlines(0.5,0,5000,color='r',linestyle='dashed')
plt.show()


# In[14]:


from sklearn.cluster import AgglomerativeClustering
cluster=AgglomerativeClustering(n_clusters=3, affinity='euclidean',linkage='ward')
labels=cluster.fit_predict(iris_df)
labels


# In[18]:


df=pd.DataFrame({'labels':labels,'Species':data_df['Species']})
df


# In[20]:


pd.crosstab(df['labels'],df['Species'])


# In[ ]:




