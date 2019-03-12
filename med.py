#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


data = pd.read_csv('med.csv')


# In[4]:


from sklearn.linear_model import LogisticRegression


# In[5]:


import statsmodels.api as mysm


# In[6]:


x = data[["Dosage_Med_1","Dosage_Med_2","Dosage_Med_3"]]


# In[7]:


y = data.Prob


# In[8]:


x["Intercept"] = 1


# In[9]:


mymodel = mysm.Logit(y,x)


# In[10]:


myresult = mymodel.fit()


# In[11]:


summary = myresult.summary()


# In[12]:


pred = myresult.predict(x)


# In[13]:


newdata = pd.DataFrame(pred)


# In[14]:


newdata.to_csv('med_1.csv')


# In[15]:


print(summary)

# In[16]:


print(x)





