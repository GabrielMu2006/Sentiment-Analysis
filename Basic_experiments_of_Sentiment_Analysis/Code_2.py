import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
import nltk
from nltk.stem import PorterStemmer
import sklearn
import matplotlib.pyplot as plt
import sklearn.feature_extraction
import sklearn.linear_model
import sklearn.metrics
import sklearn.naive_bayes
import sklearn.svm
from tqdm import tqdm_notebook

data=pd.read_csv("Data/IMDB Dataset.csv")

#Data Cleaning
def remove_html(text):
    bs = BeautifulSoup(text,"html.parser")
    return ' ' + bs.get_text() + ' '

def keep_only_letters(text):
    text=re.sub(r'[^a-zA-Z\s]','',text)
    return text

def convert_to_lowercase(text):
    return text.lower()

def clean_reviews(text):
    text=remove_html(text)
    text=keep_only_letters(text)
    text=convert_to_lowercase(text)
    return text

data['review'] = data['review'].apply(lambda review: clean_reviews(review))

#Stop Words Removal
english_stop_words=nltk.corpus.stopwords.words('english')
def remove_stop_words(text):
    for stopword in english_stop_words:
        stopword=' '+stopword+' '
        text=text.replace(stopword,' ')
    return text

data['review']=data['review'].apply(remove_stop_words)

#Stemming
def text_stemming(text):
    if not isinstance(text, str) or text.strip() == '':
        return ''
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in text.split()]
    stemmed_text = ' '.join(stemmed_tokens)
    return stemmed_text

data['review']=data['review'].apply(text_stemming)

imdb_train=data[:40000]
imdb_test=data[40000:]

train_labels=[1 if sentiment=='positive' else 0 for sentiment in imdb_train['sentiment']]
test_labels=[1 if sentiment=='positive' else 0 for sentiment in imdb_test['sentiment']]

vectorizer=sklearn.feature_extraction.text.TfidfVectorizer(use_idf=True,ngram_range=(1,2))
tfidf_features_train=vectorizer.fit_transform(imdb_train['review'])
tfidf_features_test=vectorizer.transform(imdb_test['review'])
print(tfidf_features_train.shape,tfidf_features_test.shape)

clf=sklearn.svm.LinearSVC()
clf.fit(tfidf_features_train,train_labels)

predictions=clf.predict(tfidf_features_test)
print(sklearn.metrics.classification_report(test_labels,predictions,target_names=['Negative','Positive']))
print(sklearn.metrics.confusion_matrix(test_labels,predictions,labels=[0,1]))