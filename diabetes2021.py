#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv("pima-data.csv")


# In[3]:


data.shape


# In[4]:


data.head(7)


# In[5]:


data.isnull().values.any()


# In[6]:


data.isnull().sum()


# In[7]:


data.describe()


# In[8]:


data.info()


# In[9]:


data.corr()


# In[10]:


crr = data.corr()


# In[11]:


import seaborn as sns
crr
sns.heatmap(crr,xticklabels =crr.columns,yticklabels = crr.columns)


# In[12]:


import seaborn as sns
import matplotlib.pyplot as plt
#get correlations of each features in dataset
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,10))
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[13]:


data.plot(kind = 'box', figsize = (20,10)) 
plt.show()


# In[33]:


da = data.hist(bins=15,figsize=(8,10))


# In[15]:


diabetes_map = {True: 1, False: 0}


# In[16]:


data['diabetes'] = data['diabetes'].map(diabetes_map)


# In[17]:


data.head(10)


# In[18]:


diabetes_true_count = len(data.loc[data['diabetes'] == True])
diabetes_false_count = len(data.loc[data['diabetes'] == False])


# In[20]:


(diabetes_true_count)


# In[21]:


(diabetes_false_count)


# In[35]:


print("total number of rows : {0}".format(len(data)))
print("number of rows missing glucose_conc: {0}".format(len(data.loc[data['glucose_conc'] == 0])))
print("number of rows missing diastolic_bp: {0}".format(len(data.loc[data['diastolic_bp'] == 0])))
print("number of rows missing insulin: {0}".format(len(data.loc[data['insulin'] == 0])))
print("number of rows missing bmi: {0}".format(len(data.loc[data['bmi'] == 0])))
print("number of rows missing diab_pred: {0}".format(len(data.loc[data['diab_pred'] == 0])))
print("number of rows missing age: {0}".format(len(data.loc[data['age'] == 0])))
print("number of rows missing skin: {0}".format(len(data.loc[data['skin'] == 0])))


# In[22]:


from sklearn.model_selection import train_test_split
feature_columns = ['num_preg', 'glucose_conc', 'diastolic_bp', 'insulin', 'bmi', 'diab_pred', 'age', 'skin']
predicted_class = ['diabetes']


# In[23]:


X = data[feature_columns].values
y = data[predicted_class].values


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=10)


# In[25]:


len(X_train)


# In[26]:


len(y_test)


# In[42]:


from sklearn.linear_model import LinearRegression
F = LinearRegression()


# In[43]:


F.fit(X_train,y_train)


# In[44]:


F.predict(X_test)


# In[39]:


y_test


# In[41]:


F.score(X_test,y_test)


# In[ ]:





# In[ ]:




