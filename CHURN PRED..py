#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
# # LOAD DATA

# In[2]:


data = pd.read_csv('Downloads/Churn_Modelling.csv')


# In[3]:


data


# In[4]:


data.head()


# # DATA PREPROCESSING

# In[5]:


data.info()


# In[6]:


data.describe()


# In[7]:


data.isnull().sum()


# In[8]:


data.dropna()


# In[9]:


data.drop_duplicates()


# In[10]:


data['Gender'].value_counts()


# # EXPLORATORY DATA ANALYSIS(EDA)

# In[11]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[12]:


sns.countplot(x='Gender', data=data)
plt.show()


# In[13]:


sns.pairplot(data)


# In[14]:


data.corr()


# In[15]:


sns.heatmap(data.corr(),annot=True)


# In[16]:


data.drop(['Surname','Geography'],axis=1)


# In[17]:


data['Gender'] = data['Gender'].astype('category')
data['Gender'] = pd.get_dummies(data['Gender'],drop_first=True)
pd.get_dummies(data['Gender'],drop_first=False)


# # TRAINING AND TESTING DATA

# In[18]:


from sklearn.preprocessing import LabelEncoder


# In[19]:


categorical_features = data.select_dtypes(include = ['object']).columns.tolist()
le = LabelEncoder()
for col in categorical_features:
    data[col] = le.fit_transform(data[col])


# In[20]:


X = data.drop('Exited',axis = 1)
y = data['Exited']


# In[21]:


X


# In[22]:


y


# In[23]:


from sklearn.preprocessing import StandardScaler


# In[24]:


from sklearn.model_selection import train_test_split


# In[25]:


X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3,random_state=20)


# In[26]:


Scaler = StandardScaler()
X_train = Scaler.fit_transform(X_train)
X_test = Scaler.transform(X_test)


# # LOGISTIC REGRESSION

# In[27]:


from sklearn.linear_model import LogisticRegression


# In[28]:


model = LogisticRegression()
model.fit(X_train,y_train)


# In[29]:


pred = model.predict(X_test)


# In[30]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[31]:


print("Accuracy:",accuracy_score(y_test,pred))
print("*"*50)
print(confusion_matrix(y_test,pred))
print("*"*50)
print(classification_report(y_test,pred))


# In[32]:


result = pd.DataFrame({'Actual': y_test, 'Predicted': pred})
result


# # RANDOM FOREST

# In[33]:


from sklearn.ensemble import RandomForestClassifier


# In[34]:


model = RandomForestClassifier()
model.fit(X_train,y_train)


# In[35]:


pred = model.predict(X_test)


# In[36]:


print("Accuracy:",accuracy_score(y_test,pred))
print("*"*50)
print(confusion_matrix(y_test,pred))
print("*"*50)
print(classification_report(y_test,pred))


# In[37]:


result = pd.DataFrame({'Actual': y_test, 'Predicted': pred})
result


# # DECISION TREE

# In[38]:


from sklearn.tree import DecisionTreeClassifier


# In[39]:


model = DecisionTreeClassifier()
model.fit(X_train,y_train)


# In[40]:


pred = model.predict(X_test)


# In[41]:


print("Accuracy:",accuracy_score(y_test,pred))
print("*"*50)
print(confusion_matrix(y_test,pred))
print("*"*50)
print(classification_report(y_test,pred))


# # SVC

# In[42]:


from sklearn.svm import SVC


# In[43]:


model = SVC()
model.fit(X_train,y_train)


# In[44]:


pred = model.predict(X_test)


# In[45]:


print("Accuracy:",accuracy_score(y_test,pred))
print("*"*50)
print(confusion_matrix(y_test,pred))
print("*"*50)
print(classification_report(y_test,pred))


# # KNN

# In[46]:


from sklearn.neighbors import KNeighborsClassifier


# In[47]:


model = KNeighborsClassifier()
model.fit(X_train,y_train)


# In[48]:


pred = model.predict(X_test)


# In[49]:


print("Accuracy:",accuracy_score(y_test,pred))
print("*"*50)
print(confusion_matrix(y_test,pred))
print("*"*50)
print(classification_report(y_test,pred))


# # ACCURACY PLOTTING

# In[50]:


acc=[81.66,86.33,79.26,85.7,82]
name=['log_reg','random','disi','SVC','KNN']
batch_size=[16,32,64,128,135]


# In[51]:


fig = plt.figure(figsize = (10, 5))
plt.rc('font', size=20)
plt.ylim((0,100))
plt.bar(name, acc,color=['blue','green','red','yellow','black'],width = 0.8,edgecolor='black')
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison")
plt.show()


# In[52]:


plt.plot(batch_size,acc,'b-o',label='Accuracy over batch size for 1000 iterations');


# # CALIBRATION PLOT

# In[ ]:


import scikitplot as skplt


# In[54]:


#CALIBRATION CURVE FOR ALL MODELS
lr_probas = LogisticRegression().fit(X_train, y_train).predict_proba(X_test)
rf_probas = RandomForestClassifier().fit(X_train, y_train).predict_proba(X_test)
dt_probas = DecisionTreeClassifier().fit(X_train, y_train).predict_proba(X_test)
#svc_probas = SVC().fit(X_train,y_train).predict_proba(X_test)
Knn_probas = KNeighborsClassifier().fit(X_train,y_train).predict_proba(X_test)


# In[55]:


probas_list = [rf_probas, dt_probas,lr_probas,Knn_probas]
clf_names = ['Random Forest', 'Decision Tree','Logistic Regression', 'Knn']


# In[56]:


skplt.metrics.plot_calibration_curve(y_test,
                                     probas_list,
                                     clf_names, n_bins=15,
                                     figsize=(12,6)
                                     );


# In[ ]:




