#!/usr/bin/env python
# coding: utf-8

# In[49]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score,mean_squared_error
from sklearn.preprocessing import LabelEncoder


# In[50]:


df = pd.read_csv('salaries.csv')
df.head(10)


# In[51]:


label_encoder = LabelEncoder()
for col in ['company', 'job', 'degree']:
    df[col] = label_encoder.fit_transform(df[col])


# In[52]:


X = df.drop('salary_more_then_100k', axis=1)
Y = df['salary_more_then_100k']


# In[53]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[54]:


dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, Y_train)


# In[55]:


dt_predictions = dt_classifier.predict(X_test)
dt_predictions


# In[56]:


dt_accuracy = accuracy_score(Y_test, dt_predictions)
print("Decision Tree Classifier Accuracy:", dt_accuracy)


# In[57]:


log_reg = LogisticRegression()
log_reg.fit(X_train, Y_train)


# In[58]:


log_reg_predictions = log_reg.predict(X_test)
log_reg_predictions


# In[59]:


X = df.drop('salary_more_then_100k', axis=1)
Y = df['salary_more_then_100k']


# In[60]:



linear_reg = LinearRegression()
linear_reg.fit(X_train, Y_train)


# In[62]:


for model in [DecisionTreeClassifier(), LogisticRegression(), LinearRegression()]:
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    evaluate_model(predictions, model_name=model.__class__.__name__)


# In[63]:


models = ['Decision Tree', 'Logistic Regression', 'Linear Regression']
accuracies = [dt_accuracy, log_reg_accuracy, 1 - linear_reg_mse]  # Assuming accuracy scores have been calculated
plt.bar(models, accuracies)
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()


# In[ ]:




