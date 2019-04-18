#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import matplotlib.pyplot as plt


# In[11]:


df = pd.read_csv("C:/Users/SUMEET/Downloads/diabetes.csv")


# In[12]:


df.head()


# In[13]:


x = df[['Pregnancies',
'Glucose',
'BloodPressure',
'SkinThickness',
'Insulin',
'BMI',
'DiabetesPedigreeFunction',
'Age']]


# In[14]:


x.head()


# In[15]:


y = df[['Outcome']]


# In[16]:


y.head()


# In[17]:


from sklearn.model_selection import train_test_split


# In[18]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)


# In[19]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[20]:


plt.scatter(x[['Pregnancies']],y,color = 'red',marker='*')
plt.scatter(x[['Glucose']],y,color = 'green',marker='*')
plt.scatter(x[['BloodPressure']],y,color = 'blue',marker='*')
plt.scatter(x[['Insulin']],y,color = 'yellow',marker='*')
plt.scatter(x[['BMI']],y,color = 'black',marker='*')


# In[21]:


from sklearn.svm import SVC


# In[22]:


model = SVC(kernel='linear',C=1)


# In[23]:


model.fit(x_train,y_train)


# In[24]:


model.score(x_test,y_test)


# In[25]:


model.predict([[6,148,72,35,0,33.6,0.627,50]])


# In[26]:


import pickle


# In[27]:


with open('model_file','wb') as f:
    pickle.dump(model,f)


# In[28]:


with open('model_file','rb') as f:
    mp = pickle.load(f)


# In[30]:


mp.predict([[6,148,72,35,0,33.6,0.627,50]])


# In[ ]:




