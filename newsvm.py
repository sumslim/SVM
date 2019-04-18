#!/usr/bin/env python
# coding: utf-8

# In[142]:


import pandas as pd
import matplotlib.pyplot as plt


# In[143]:


df = pd.read_csv("C:/Users/SUMEET/Downloads/diabetes.csv")


# In[144]:


df.head()


# In[145]:


x = df[['Pregnancies',
'Glucose',
'BloodPressure',
'SkinThickness',
'Insulin',
'BMI',
'DiabetesPedigreeFunction',
'Age']]


# In[146]:


x.head()


# In[147]:


y = df[['Outcome']]


# In[148]:


y.head()


# In[149]:


from sklearn.model_selection import train_test_split


# In[150]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)


# In[151]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[152]:


plt.scatter(x[['Pregnancies']],y,color = 'red',marker='*')
plt.scatter(x[['Glucose']],y,color = 'green',marker='*')
plt.scatter(x[['BloodPressure']],y,color = 'blue',marker='*')
plt.scatter(x[['Insulin']],y,color = 'yellow',marker='*')
plt.scatter(x[['BMI']],y,color = 'black',marker='*')


# In[153]:


x_test.head()


# In[154]:


y_test.head()


# In[155]:


from sklearn import svm


# In[156]:


model = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)


# In[157]:


model.fit(x_train,y_train)


# In[158]:


model.predict([[1,85,66,29,0,26.6,0.351,31]])


# In[159]:


model.predict([[6,148,72,35,0,33.6,0.627,50]])


# In[160]:


y_predicted = model.fit_predict(x_test)
y_predicted


# In[161]:


m=0
for i in range(len(y_predicted)):
    if y_predicted[i]==1:
        m=m+1


# In[162]:


score = (len(y_predicted)-m)/len(y_predicted)
score


# In[163]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_predicted)
cm


# In[164]:


import seaborn as sn
sn.heatmap(cm,annot=True)


# In[165]:


import pickle


# In[166]:


with open('model_file','wb') as f:
    pickle.dump(model,f)


# In[167]:


with open('model_file','rb') as f:
    mp = pickle.load(f)


# In[168]:


mp.predict([[6,148,72,35,0,33.6,0.627,50]])


# In[ ]:




