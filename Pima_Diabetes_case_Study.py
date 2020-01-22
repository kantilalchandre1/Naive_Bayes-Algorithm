#!/usr/bin/env python
# coding: utf-8

# In[41]:


import numpy as np
import pandas as pd
from sklearn import datasets


# In[42]:


dataset=pd.read_csv('diabetes.csv')


# In[43]:


dataset


# In[44]:


import seaborn as sns


# In[45]:


sns.pairplot(dataset,diag_kind='kde',hue='Outcome')


# In[46]:


dataset.describe().transpose()


# In[47]:


#check target feature for balanced data
dataset['Outcome'].value_counts()


# In[ ]:


#Check zeros values in columns
print('Zero values in Glucose:',len(dataset[dataset['Glucose']==0]))
print('Zero values in Pregnancies:',len(dataset[dataset['Pregnancies']==0]))
print('Zero values in BloodPressure	:',len(dataset[dataset['BloodPressure']==0]))
print('Zero values in SkinThickness:',len(dataset[dataset['SkinThickness']==0]))
print('Zero values in Insulin:',len(dataset[dataset['Insulin']==0]))
print('Zero values in BMI:',len(dataset[dataset['BMI']==0]))
print('Zero values in DiabetesPedigreeFunction:',len(dataset[dataset['DiabetesPedigreeFunction']==0]))
print('Zero values in Age:',len(dataset[dataset['Age']==0]))


# In[60]:


# splitting dataset into x and y
X=dataset.iloc[:,0:8]
y=dataset.iloc[:,8]


# In[69]:


# train and test spillting 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=0)


# In[70]:


#fill zeros with mean values
from sklearn.preprocessing import Imputer
fill_values=Imputer(missing_values=0,strategy='mean',axis=0)


# In[71]:


X_train=fill_values.fit_transform(X_train)
X_test=fill_values.fit_transform(X_test)


# In[73]:


# Building the model by using Naive bayes Algorithm
from sklearn.naive_bayes import GaussianNB
model=GaussianNB()


# In[74]:


# model fitting
model.fit(X_train,y_train)


# In[75]:


#Prediction on test Data
y_pred=model.predict(X_test)


# In[87]:


# Checking accuracy 
from sklearn.metrics import confusion_matrix,classification_report
cm=confusion_matrix(y_test,y_pred)


# In[88]:


print(cm)


# In[89]:


# by using classification report study precision,recall,f1-score
print(classification_report(y_test,y_pred))


# In[105]:


# checking accuray by using Cross validation
from sklearn.model_selection import cross_val_score
cl=cross_val_score(model,X,y,cv=5)


# In[106]:


cll=np.average(cl)


# In[107]:


cll


# In[ ]:




