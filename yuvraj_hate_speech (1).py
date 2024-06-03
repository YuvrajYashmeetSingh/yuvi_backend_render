#!/usr/bin/env python
# coding: utf-8

# In[7]:


#Importing the packages
import pandas as pd
import numpy as np
from sklearn. feature_extraction. text import CountVectorizer
from sklearn. model_selection import train_test_split
from sklearn. tree import DecisionTreeClassifier


# In[8]:


data = pd. read_csv("labeled_data.csv")
#To preview the data
print(data. head())


# In[3]:


import re
#nltk. download('stopwords')
import nltk
from nltk.corpus import stopwords


# In[4]:


nltk.download('stopwords')  # Download stopwords corpus if not already downloaded
stopword = set(stopwords.words('english'))
stemmer = nltk.SnowballStemmer("english")


# In[9]:


data["labels"] = data["class"]. map({0: "Hate Speech", 1: "Offensive Speech", 2: "No Hate and Offensive Speech"})
data = data[["tweet", "labels"]]
print(data. head())


# In[11]:


import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Initialize stopwords and stemmer
nltk.download('stopwords')
stopwords_set = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")

# Define the clean function
def clean(text):
    text = str(text).lower()
    text = re.sub('[.?]', '', text)
    text = re.sub('https?://\S+|www.\S+', '', text)
    text = re.sub('<.?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w\d\w', '', text)
    text = [word for word in text.split(' ') if word not in stopwords_set]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text

# Assuming 'data' is your DataFrame
data["tweet"] = data["tweet"].apply(clean)


# In[12]:


x = np. array(data["tweet"])
y = np. array(data["labels"])
cv = CountVectorizer()
X = cv. fit_transform(x)
# Splitting the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[13]:


#Model building
model = DecisionTreeClassifier()
#Training the model
model. fit(X_train,y_train)


# In[14]:


#Testing the model
y_pred = model. predict (X_test)
y_pred


# In[15]:


#Accuracy Score of our model
from sklearn. metrics import accuracy_score
print (accuracy_score (y_test,y_pred))


# In[17]:


#Predicting the outcome
inp = "You are too bad and I dont like your attitude"
inp = cv.transform([inp]).toarray()
print(model.predict(inp))


# In[18]:


inp = "It is really awesome"
inp = cv. transform([inp]). toarray()
print(model. predict(inp))


# In[19]:


inp = "fuck you"
inp = cv. transform([inp]). toarray()
print(model. predict(inp))


# In[20]:


inp = "your are idiot man"
inp = cv. transform([inp]). toarray()
print(model. predict(inp))


# In[21]:


inp = "your are killing me"
inp = cv. transform([inp]). toarray()
print(model. predict(inp))


# In[22]:


inp = "go to hell man"
inp = cv. transform([inp]). toarray()
print(model. predict(inp))


# In[23]:


inp = "your looking fucking awsawesome "
inp = cv. transform([inp]). toarray()
print(model. predict(inp))


# In[24]:


inp = "good morning"
inp = cv. transform([inp]). toarray()
print(model. predict(inp))

import pickle 
pickle.dump(model, open('yuvraj_hate_speech(1).pkl', 'wb'))





