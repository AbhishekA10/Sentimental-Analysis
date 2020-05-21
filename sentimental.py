#Import libraries
import numpy as np
import pandas as pd

#Import dataset
data=pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t')

import nltk
from nltk.corpus import stopwords
import string

#Cleaning data
def clean(text):
    text=text.lower().split()
    punc=[char for char in text if char not in string.punctuation]
    Review=[word for word in punc if word not in stopwords.words('english')]
    Review=' '.join(Review)
    return Review

data['Review']=data['Review'].apply(clean)

#Splitting the training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data['Review'], data['Liked'], test_size = 0.30, random_state = 101)

#Creating Bag of Words model
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer()
X_train=vectorizer.fit_transform(X_train).toarray()

#Training using Logistic Regression
from sklearn.linear_model import LogisticRegression
lm=LogisticRegression()
lm.fit(X_train,y_train)

#Predicting output
X_test=vectorizer.transform(X_test).toarray()
p=lm.predict(X_test)

#Creating Confusion matrix and Classification report
from sklearn.metrics import confusion_matrix,classification_report
mat=confusion_matrix(y_test,p)
rep=classification_report(y_test,p)