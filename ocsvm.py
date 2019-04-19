#!/usr/bin/env python
# coding: utf-8

# In[157]:


import pandas as pd
import matplotlib.pyplot as plt


# In[158]:


df = pd.read_csv("C:/Users/SUMEET/Downloads/diabetes.csv")


# In[159]:


df.head()


# In[160]:


x = df[['Pregnancies',
'Glucose',
'BloodPressure',
'SkinThickness',
'Insulin',
'BMI',
'DiabetesPedigreeFunction',
'Age']]


# In[161]:


x.head()


# In[162]:


y = df[['Outcome']]


# In[163]:


y.head()


# In[164]:


from sklearn.model_selection import train_test_split


# In[165]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)


# In[166]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[167]:


plt.scatter(x[['Pregnancies']],y,color = 'red',marker='*')
plt.scatter(x[['Glucose']],y,color = 'green',marker='*')
plt.scatter(x[['BloodPressure']],y,color = 'blue',marker='*')
plt.scatter(x[['Insulin']],y,color = 'yellow',marker='*')
plt.scatter(x[['BMI']],y,color = 'black',marker='*')


# In[168]:


x_test.head()


# In[169]:


y_test.head()


# In[170]:


n=0
for i in y_test.Outcome:
    if i==1:
        n=n+1


# In[171]:


l=0
for i in y_test.Outcome:
    if i==0:
        l=l+1


# In[172]:


l/(n+l)


# In[173]:


from sklearn import svm


# In[174]:


model = svm.OneClassSVM(nu=0.1, kernel='rbf', gamma=100)


# In[175]:


model.fit(x_train,y_train)


# In[176]:


model.predict([[1,85,66,29,0,26.6,0.351,31]])


# In[177]:


model.predict([[6,148,72,35,0,33.6,0.627,50]])


# In[178]:


y_predicted = model.fit_predict(x_test)
y_predicted


# In[179]:


m=0
for i in range(len(y_predicted)):
    if y_predicted[i]==1:
        m=m+1


# In[180]:


score = (len(y_predicted)-m)/len(y_predicted)
score


# In[181]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_predicted)
cm


# In[182]:


import seaborn as sn
sn.heatmap(cm,annot=True)


# In[183]:


import pickle


# In[184]:


with open('model_file','wb') as f:
    pickle.dump(model,f)


# In[185]:


with open('model_file','rb') as f:
    mp = pickle.load(f)


# In[186]:


mp.predict([[6,148,72,35,0,33.6,0.627,50]])


# In[ ]:




