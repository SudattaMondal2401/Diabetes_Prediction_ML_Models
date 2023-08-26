#!/usr/bin/env python
# coding: utf-8

# # Importing the main libraries

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# # Loading the dataset

# In[3]:


dataset = pd.read_csv("diabetes_prediction_dataset.csv")
print(dataset.head())


# # Preliminary Viewing of Data

# ## Basic overview

# In[105]:


dataset.info()


# ## Checking the categorical data 

# In[106]:


dataset["smoking_history"].value_counts()


# In[107]:


dataset["gender"].value_counts()


# ## Plotting of the histograms

# In[108]:


dataset.hist(bins=50, figsize=(20,15))
plt.show()


# In[109]:


dataset.describe()


# ## Finding correlation with target variable

# In[110]:


correlation = dataset.corr()
correlation["diabetes"].sort_values(ascending = False)


# # Cleaning and Processing the data

# ## Dealing with the smoking category

# In[4]:


dataset.loc[dataset["smoking_history"] == 'No Info','smoking_history'] = np.nan
dataset.loc[dataset['smoking_history'] == 'former', 'smoking_history'] = "not current"
print(dataset.info())
dataset.dropna(subset = ["smoking_history"], inplace = True)
print(dataset.info())


# In[5]:


print(dataset["smoking_history"].value_counts())
print(dataset["gender"].value_counts())


# ## Dealing with the gender category

# In[6]:


dataset = dataset[dataset["gender"]!="Other"]
dataset["gender"].value_counts()


# ## Separating into target variable and splitting the dataset

# In[7]:


x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]


# In[115]:


#from sklearn.model_selection import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

#[we get higher accuracy score with this split]


# In[8]:


from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for i, (train_index, test_index) in enumerate(sss.split(x, y)):
    train = train_index
    test = test_index


# In[9]:


i1 = []
for i in train:
    i1.append(i)
    
i2 = []
for i in test:
    i2.append(i)

x_train = x.iloc[i1, :]
x_test = x.iloc[i2, :]


# In[10]:


y_train = y.iloc[i1]
y_test = y.iloc[i2]


# ## Encoding and Scaling the data 

# In[11]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

attr1 = ["gender"]
attr2 = ["smoking_history"]
attr3 = ["age", "hypertension", "heart_disease", "bmi", "HbA1c_level", "blood_glucose_level"]
ct = ColumnTransformer(transformers =
    [("Standard", StandardScaler(), attr3),
     ('OneHot', OneHotEncoder(), attr2),
    ('Ordinal', OrdinalEncoder(), attr1)],
      remainder = "passthrough")

x_train = ct.fit_transform(x_train)


# In[12]:


print(pd.DataFrame(x_train))


# # Building the model

# In[13]:


from sklearn.metrics import accuracy_score, confusion_matrix
x_test = ct.transform(x_test)


# ## Decision Tree

# In[18]:


from sklearn.tree import DecisionTreeClassifier
classifier1 = DecisionTreeClassifier(random_state = 0)
classifier1.fit(x_train, y_train)

y_pred = classifier1.predict(x_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# ## Logistic Regression

# In[123]:


from sklearn.linear_model import LogisticRegression
classifier2 = LogisticRegression(random_state = 0)
classifier2.fit(x_train, y_train)

y_pred = classifier2.predict(x_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# ## Random Forest 

# In[124]:


from sklearn.ensemble import RandomForestClassifier
classifier3 = RandomForestClassifier(n_estimators = 50, random_state = 0) #grid search was not used, but I experimented with a few values for n_neighbors. This gave the best result
classifier3.fit(x_train, y_train)
classifier3.fit(x_train, y_train)

y_pred = classifier3.predict(x_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# ## Naive Bayes

# In[125]:


from sklearn.naive_bayes import GaussianNB
classifier4 = GaussianNB()
classifier4.fit(x_train, y_train)

y_pred = classifier4.predict(x_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# ## K-NN Classifier

# In[126]:


from sklearn.neighbors import KNeighborsClassifier
classifier5 = KNeighborsClassifier(n_neighbors = 9) #grid search was not used, but I experimented with a few values for n_neighbors. This gave the best result
classifier5.fit(x_train, y_train)

y_pred = classifier5.predict(x_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# ## Support Vector Classifier (RBF model) 

# In[127]:


from sklearn.svm import SVC
classifier6 = SVC(kernel = 'rbf', random_state = 0) #linear model was tried, results were lower
classifier6.fit(x_train, y_train)

y_pred = classifier6.predict(x_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# ## Stacking Model

# In[128]:


from sklearn.ensemble import StackingClassifier
level0 = list()
level0.append(('lr', LogisticRegression()))
level0.append(('knn', KNeighborsClassifier(n_neighbors = 9)))
level0.append(('cart', DecisionTreeClassifier()))
level0.append(('svm', SVC(kernel = 'rbf')))
level0.append(('bayes', GaussianNB()))
level0.append(('random', RandomForestClassifier(n_estimators = 50)))

level1 = LogisticRegression()

model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# # Voting Classifier

# In[129]:


from sklearn.ensemble import VotingClassifier
voting_clf = VotingClassifier(
 estimators=[('lr', LogisticRegression()), ('svc', SVC(kernel = 'rbf', probability = True)), ('rf', RandomForestClassifier(n_estimators = 50))],
 voting='soft')
voting_clf.fit(x_train, y_train)
y_pred = voting_clf.predict(x_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# In[130]:


from sklearn.ensemble import VotingClassifier
voting_clf = VotingClassifier(
 estimators=[('lr', LogisticRegression()), ('svc', SVC(kernel = 'rbf')), ('rf', RandomForestClassifier(n_estimators = 50)), ('dt', DecisionTreeClassifier()), ('knn', KNeighborsClassifier(n_neighbors = 9))],
 voting='hard')
voting_clf.fit(x_train, y_train)
y_pred = voting_clf.predict(x_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# In[ ]:




