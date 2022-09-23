#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np

data_df= pd.read_csv(r'C:\Users\masan\Downloads\iris.csv')


# In[3]:


data_df


# In[10]:


iris_df=data_df.iloc[:,1:5]


# In[11]:


iris_df


# In[12]:


iris_df.shape


# In[15]:


iris_df.isna().sum()


# In[17]:


iris_df.describe(include='all')


# In[18]:


import seaborn as sns 
sns.pairplot(iris_df)


# In[22]:


#kmeans

from sklearn.cluster import KMeans
model_kmeans=KMeans(n_clusters=3)
model_kmeans.fit(iris_df)
labels=model_kmeans.predict(iris_df)
print(labels)


# In[25]:


#any 2 featurs
import matplotlib.pyplot as plt
sl=iris_df['Sepal.Length']
pl=iris_df['Petal.Length']
plt.scatter(sl,pl,c=labels)


# In[28]:


samples=[[5.2,2.2,2.0,0.4]]
new_label= model_kmeans.predict(samples)
print(new_label)


# In[29]:


#elbow method

model_kmeans.inertia_


# In[31]:


#good modeling inertia should be low

ks=range(1,6)
inertia=[]
for k in ks:
    model=KMeans(n_clusters=k)
    model.fit(iris_df)
    inertia.append(model.inertia_)


# In[33]:


plt.plot(ks,inertia,'-o')
plt.xlabel('no of clusters')
plt.ylabel('inertia')
plt.show()


# In[36]:


#Evaluate the model

data_df['Species']


# In[41]:


data_df['Species'].value_counts()


# In[57]:


species=data_df['Species']
df=pd.DataFrame({'label':label_sc,'Species':species})


# In[45]:


df


# In[46]:


pd.crosstab(df['label'],df['Species'])


# In[49]:


#standard scaler

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
from sklearn.pipeline import make_pipeline
pipe=make_pipeline(scaler,model_kmeans)
pipe.fit(iris_df)
label_sc=pipe.predict(iris_df)
print(label_sc)


# In[61]:


#check whether good model or not

species=data_df['Species']
df=pd.DataFrame({'label':labels,'Species':species})


# In[62]:


pd.crosstab(df['label'],df['Species'])


# In[63]:


#NormalizATION
from sklearn.preprocessing import Normalizer
norm=Normalizer()

pipe1=make_pipeline(norm,model_kmeans)
pipe1.fit(iris_df)
label_norm=pipe1.predict(iris_df)
print(label_norm)


# In[64]:


species=data_df['Species']
df=pd.DataFrame({'label':label_norm,'Species':species})


# In[65]:


df


# In[66]:


pd.crosstab(df['label'],df['Species'])


# In[ ]:




