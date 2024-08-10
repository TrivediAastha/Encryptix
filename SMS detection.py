#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
import numpy as np


# # LOAD DATA

# In[33]:


data = pd.read_csv('Downloads/spam.csv',encoding='latin1')
data


# # DATA PREPROCESSING

# In[34]:


data.info()


# In[35]:


data.describe()


# In[36]:


data.isnull().sum()


# In[37]:


data = data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
data


# In[38]:


import string,re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns


# In[39]:


data['v1'].value_counts()


# In[40]:


plt.pie(data['v1'].value_counts(), labels = ['ham', 'spam'], autopct = '%0.2f')
plt.show()


# In[41]:


from nltk.stem import WordNetLemmatizer
stemmer = WordNetLemmatizer()
def preprocessing(txt):
    doc = []
    for sen in range(0,len(txt)):
        document = re.sub(r'\W', ' ',str(txt[sen]))
        document = re.sub(r'\s+[^a-zA-Z]\s+',' ',document)
        document = re.sub(r'\s+[^a-zA-Z]\s+',' ',document, flags=re.I)
        document = document.lower()
        document = document.split()
        document = [stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)
        
        doc.append(document)
        
        
    return doc


# In[42]:


x,y = data['v2'],data['v1']
data


# In[43]:


documents = preprocessing(x)
documents


# # TF-IDF

# In[44]:


from nltk.corpus import stopwords


# In[45]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_df=0.90,min_df=2,max_features=2500,stop_words=stopwords.words('english'))
x = vectorizer.fit_transform(documents).toarray()


# # TRAINING AND TESTING

# In[46]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.3,random_state=38)


# In[47]:


X_train.shape,X_test.shape,Y_train.shape,Y_test.shape


# # RANDOM FOREST

# In[48]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100,random_state=0)
model.fit(X_train,Y_train)
pred = model.predict(X_test)


# In[49]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print("accruracy:",accuracy_score(Y_test,pred))
print(classification_report(Y_test,pred))
print(accuracy_score(Y_test, pred))


# # DECISION TREE

# In[50]:


from sklearn.tree import DecisionTreeClassifier


# In[51]:


model = DecisionTreeClassifier(criterion='gini',random_state=0)
model.fit(X_train,Y_train)
pred = model.predict(X_test)


# In[52]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print("accruracy:",accuracy_score(Y_test,pred))
print(classification_report(Y_test,pred))
print(accuracy_score(Y_test, pred))


# # LOGISTIC REGRESSION

# In[53]:


from sklearn.linear_model import LogisticRegression


# In[54]:


model = LogisticRegression(C=1.0,random_state=0)
model.fit(X_train,Y_train)
pred = model.predict(X_test)


# In[55]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print("accruracy:",accuracy_score(Y_test,pred))
print(classification_report(Y_test,pred))
print(accuracy_score(Y_test, pred))


# # SVC

# In[56]:


from sklearn.svm import SVC


# In[57]:


model = SVC(C=1.0,random_state=0)
model.fit(X_train,Y_train)
pred = model.predict(X_test)


# In[58]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print("accruracy:",accuracy_score(Y_test,pred))
print(classification_report(Y_test,pred))
print(accuracy_score(Y_test, pred))


# # KNN

# In[59]:


from sklearn.neighbors import KNeighborsClassifier


# In[60]:


model = KNeighborsClassifier(n_neighbors=5,n_jobs=None)
model.fit(X_train,Y_train)
pred = model.predict(X_test)


# In[61]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print("accruracy:",accuracy_score(Y_test,pred))
print(classification_report(Y_test,pred))
print(accuracy_score(Y_test, pred))


# # ACCURACY PLOTTING

# In[64]:


acc=[98.02,96.59,96.29,97.90,91.26]
name=['random','disi','log_reg','SVC','KNN']


# In[65]:


#PLOTTING ACCURACY COMPARISON
fig = plt.figure(figsize = (10, 5))
plt.rc('font', size=20)
plt.ylim((0,100))
plt.bar(name, acc,color=['blue','green','red','yellow','black'],width = 0.8,edgecolor='black')
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison")
plt.show()


# In[ ]:




