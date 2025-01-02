#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm


# In[24]:


df = pd.read_csv(r"C:\Users\user\Downloads\emails.csv")


# In[25]:


df


# In[26]:


df.shape


# In[27]:


df.info()


# In[28]:


df.describe()


# # Data Cleaning

# In[29]:


df.head()


# In[30]:


df.isnull().sum()


# In[31]:


df.duplicated().sum()


# In[32]:


df = df.drop_duplicates()


# In[33]:


df.duplicated().sum()


# # Data Analysis

# In[34]:


get_ipython().system(' pip install nltk')


# In[35]:


import nltk


# In[36]:


nltk.download('punkt')


# In[37]:


df['text'].apply(len)


# In[38]:


df['no_characters']=df['text'].apply(len)


# In[39]:


df


# In[44]:


df['text'].apply(lambda x:nltk.word_tokenize(x))


# In[45]:


df['text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[46]:


df['no_words']=df['text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[47]:


df


# In[48]:


df['text'].apply(lambda x:nltk.sent_tokenize(x))


# In[93]:


df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[94]:


df['no_sentences']=df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[95]:


df


# In[98]:


df[['no_characters', 'no_words', 'no_sentences']].describe()


# In[100]:


df[df['spam']==0][['no_characters', 'no_words', 'no_sentences']].describe()


# In[101]:


df[df['spam']==0][['no_characters', 'no_words', 'no_sentences']].describe()


# In[107]:


plt.figure(figsize=(18,6))
sns.histplot(df[df['spam']==0]['no_characters'])
sns.histplot(df[df['spam']==1]['no_characters'], color='red')



# In[108]:


plt.figure(figsize=(18,6))
sns.histplot(df[df['spam']==0]['no_words'])
sns.histplot(df[df['spam']==1]['no_words'], color='red')


# In[109]:


plt.figure(figsize=(18,6))
sns.histplot(df[df['spam']==0]['no_sentences'])
sns.histplot(df[df['spam']==1]['no_sentences'], color='red')


# In[111]:


sns.pairplot(df, hue='spam')


# In[112]:


pd.to_numeric


# In[116]:


numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True)


# # Data/Text Preprocesing
# 

# In[4]:


# 1.Lower

def transform_text1(text1):
    text1 = text1.lower()
    return text1


# In[6]:


transform_text1("HI HOW are YOU")


# In[19]:


# 2.Tokenization
def transform_text2(text2):
    text2 = nltk.word_tokenize(text2)
    return text2


# In[20]:


transform_text2("HI how Are You")


# In[17]:


# 3.removing special character:
def transform_text3(text3):
    y = []
    for i in text3:
        if i.isalnum(): #or not i:
            y.append(i)
    return y


# In[12]:


transform_text3('HI how Are % @ 20 45* You')


# In[51]:


def transform_text3a(text3a):
    text3a = text3a.lower()
    text3a = nltk.word_tokenize(text3a)
    y_a=[]
    for i in text3a:
        if i.isalnum():
            y_a.append(i)
    return y_a


# In[52]:


transform_text3a('HI how Are % @ 20 45* You')


# # Train_test_split

# In[56]:


X = df['text']
Y = df['spam']
X_train,Y_train,X_test,Y_train = train_test_split(X,Y,test_size=0.2,random_state=1)
"""train_test_Split function:It used to divide the dataset into training and testing datasets.
   test_size: The proportion of the dataset to include in the testing split. 
   Here, test_size=0.2 means that 20% of the data will be used for testing, and the remaining 80% will be used for training.
   X_train: The features of the training set.
   X_test: The features of the testing set.
   Y_train: The target variable of the training set.
   Y_test: The target variable of the testing set.
"""


# In[57]:


cv = CountVectorizer()
#CountVectorizer = used to convert collection of text documents into matrix token counts
#initialize the CountVectorizer() class by cv variable.
""" where each row represents a document and each column represents a unique word (token) in the corpus."""
features=cv.fit_transform(X_train)
"""This line fits the CountVectorizer object to the training data (X_train) and transforms it into a matrix of token counts.
    The fit_transform() method first fits the vectorizer to the training data, learning the vocabulary of the corpus and 
    building the token count matrix."""


# In[58]:


X_train.describe()


# In[59]:


model = svm.SVC()
"""This line initializes an SVM model object named model using the SVC class from scikit-learn. 
   SVC stands for Support Vector Classifier, which is a type of SVM model used for classification tasks.
   By default, SVC uses the radial basis function (RBF) kernel for non-linear classification."""
model.fit(features,Y_train)


# In[60]:


print(f"Size of splitted data")
print(f"X_train {X_train.shape}")
print(f"Y_train {Y_train.shape}")
print(f"X_test {X_test.shape}")
print(f"Y_test {Y_test.shape}")


# In[61]:


Y_test.value_counts()


# # Confusion Matrix and naive bayes

# In[59]:


from sklearn.feature_extraction.text import TfidfVectorizer
#Here we import the TfidfVectorizer from sklearn.feature_extraction.textb Wterm frequencyhich is used to convert the textual data into TFIDF vector.
# TFIDF:term frequency inverse document frequency.
from sklearn.naive_bayes import GaussianNB
#GaussianNB from sklearn.naive_bayes for the Gaussian Naive Bayes classifier.

# Assuming X_train and X_test are Series objects containing textual data
# Convert textual data into numerical format using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
#initialize the TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
"""This line fits the tfidf_vectorizer to the training data X_train and transforms it into a matrix of TF-IDF features. 
    The fit_transform() method first fits the vectorizer to the training data, learning the vocabulary and computing the IDF 
    (Inverse Document Frequency) values. Then, it transforms the training data into a matrix of TF-IDF features based on the
    learned vocabulary and IDF values."""
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Initialize and fit Gaussian Naive Bayes model
nb = GaussianNB()
nb.fit(X_train_tfidf.toarray(), Y_train)
"""This line fits the Naive Bayes classifier nb to the TF-IDF features of the training data X_train_tfidf and the corresponding
   labels Y_train. Since Naive Bayes expects input features to be in a dense matrix format
   toarray() method is used to convert the sparse TF-IDF matrix into a dense matrix."""
y_pred_nb = nb.predict(X_test_tfidf.toarray())
"""This line predicts the labels for the test data X_test_tfidf using the trained Naive Bayes classifier nb.
   The predict() method computes the predicted labels based on the TF-IDF features of the test data."""


# In[61]:


ConfusionMatrixDisplay.from_predictions(Y_test,y_pred_nb)
plt.title('Naive bayes')
plt.show()
print(f" Accuracy is {accuracy_score(Y_test,y_pred_nb)}")
print(classification_report(Y_test,y_pred_nb))


# In[ ]:





# In[ ]:




