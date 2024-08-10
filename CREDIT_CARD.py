#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# # LOAD DATA

# In[2]:


df = pd.read_csv('Downloads/fraudTrain.csv')
df


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


dt = pd.read_csv('Downloads/fraudTest.csv')
dt


# In[7]:


dt.head()


# In[8]:


dt.shape


# In[9]:


dt.info()


# # DATA PREPROCESSING

# In[10]:


data = pd.concat([df,dt])


# In[11]:


data


# In[12]:


data.shape


# In[13]:


data.info()


# In[14]:


data.describe()


# In[15]:


data.isnull().sum()


# In[16]:


data.dropna()


# In[17]:


data.drop_duplicates()


# In[18]:


def clean_data(clean):
     clean.drop(["Unnamed: 0",'cc_num','first', 'last', 'street', 'city', 'state', 'zip', 'dob', 'trans_num','trans_date_trans_time'],axis=1, inplace=True)
     clean.dropna()
     return clean


# In[19]:


clean_data(data)


# In[20]:


from sklearn.preprocessing import LabelEncoder


# In[21]:


encoder=LabelEncoder()
def encode(data):
    data['merchant']=encoder.fit_transform(data['merchant'])
    data["category"] = encoder.fit_transform(data["category"])
    data["gender"] = encoder.fit_transform(data["gender"])
    data["job"] = encoder.fit_transform(data["job"])
    return data


# In[22]:


encode(data)


# In[23]:


data.corr()


# # EXPLORATORY DATA ANALYSIS

# In[24]:


data.hist(figsize=(20,10),bins=50)


# In[25]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[26]:


sns.heatmap(data.corr(),annot=True)


# In[27]:


corr = data.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)


# # TRAINING AND TESTING DATA

# In[28]:


from sklearn.model_selection import train_test_split


# In[29]:


x=data.drop(columns=['is_fraud'])
y=data['is_fraud']


# In[30]:


x


# In[31]:


y


# In[32]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# # LOGISTIC REGRESSION

# In[33]:


from sklearn.linear_model import LogisticRegression


# In[34]:


model = LogisticRegression()
model.fit(x_train,y_train)


# In[35]:


pred = model.predict(x_test)


# In[36]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[37]:


print("Accuracy:",accuracy_score(y_test,pred))
print("*"*50)
print(confusion_matrix(y_test,pred))
print("*"*50)
print(classification_report(y_test,pred))


# # DECISION TREE REGRESSOR

# In[39]:


from sklearn.tree import DecisionTreeRegressor


# In[40]:


model = DecisionTreeRegressor()
model.fit(x_train,y_train)


# In[41]:


pred = model.predict(x_test)


# In[42]:


print("Accuracy:",accuracy_score(y_test,pred))
print("*"*50)
print(confusion_matrix(y_test,pred))
print("*"*50)
print(classification_report(y_test,pred))

