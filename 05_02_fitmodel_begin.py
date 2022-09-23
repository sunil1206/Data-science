#!/usr/bin/env python
# coding: utf-8

# ## Python statistics essential training - 05_02_fitmodel

# Standard imports

# In[1]:


import math


# In[2]:


import numpy as np
import pandas as pd


# In[3]:


import matplotlib
import matplotlib.pyplot as pp


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf


# Loading gapminder data for year 1985 (Live Aid!) and setting up plot as in chapter 3

# In[5]:


gapminder = pd.read_csv('gapminder.csv')


# In[6]:


gdata = gapminder.query('year == 1985')


# In[ ]:


size = 1e-6 * gdata.population

colors = gdata.region.map({'Africa': 'skyblue', 'Europe': 'gold', 'America': 'palegreen', 'Asia': 'coral'})

def plotdata():
    gdata.plot.scatter('age5_surviving','babies_per_woman',
                       c=colors,s=size,linewidths=0.5,edgecolor='k',alpha=0.5)

