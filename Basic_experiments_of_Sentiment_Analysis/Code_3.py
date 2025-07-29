import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import tensorflow as tf
#from tensorflow.keras.layers import Activation


data=pd.read_csv("Data/IMDB Dataset.csv")

def remove_html(text):
    bs = BeautifulSoup(text,"html.parser")
    return ' ' + bs.get_text() + ' '

def keep_only_letters(text):
    text=re.sub(r'[^a-zA-Z\s]',' ',text)
    return text

def convert_to_lowercase(text):
    return text.lower()

def clean_reviews(text):
    text=remove_html(text)
    text=keep_only_letters(text)
    text=convert_to_lowercase(text)
    return text

data['review'] = data['review'].apply(lambda review: clean_reviews(review))



imdb_train=data[:40000]
imdb_test=data[40000:]

from collections import Counter

counter=Counter([words for reviews in imdb_train['review'] for words in reviews.split()])
df=pd.DataFrame()
df['key']=counter.keys()
df['value']=counter.values()
df.sort_values(by='value',ascending=False,inplace=True)

#print(df.shape[0])
#print(df[:10000].value.sum()/df.value.sum())
top_10k_words=list(df[:10000].key.values)

def get_encoded_input(review):
    words=review.split()
    if len(words) > 500:
        words = words[:500]
    encoding=[]
    for word in words:
        try:
            index=top_10k_words.index(word)
        except:
            index=10000
        encoding.append(index)
    while len(encoding) < 500:
        encoding.append(10001)
    return encoding

trianing_data=np.array([get_encoded_input(review) for review in imdb_train['review']])
testing_data=np.array([get_encoded_input(review) for review in imdb_test['review']])
#print(trianing_data.shape,testing_data.shape)

data['review_word_length']=[len(review.split()) for review in data['review']]
#data['review_word_length'].plot(kind='hist',bins=30)
#plt.title('Word length distribution')
#plt.show()

train_labels=[1 if sentiment=='positive' else 0 for sentiment in imdb_train['sentiment']]
test_labels=[1 if sentiment=='positive' else 0 for sentiment in imdb_test['sentiment']]
train_labels=np.array(train_labels)
test_labels=np.array(test_labels)

'''
#MLP
input_data=tf.keras.layers.Input(shape=(500,))

data=tf.keras.layers.Embedding(input_dim=10002,output_dim=32,input_length=500)(input_data)

data=tf.keras.layers.Flatten()(data)

data=tf.keras.layers.Dense(16)(data)
data=tf.keras.layers.Activation('relu')(data)
data=tf.keras.layers.Dropout(0.5)(data)

data=tf.keras.layers.Dense(8)(data)
data=tf.keras.layers.Activation('relu')(data)
data=tf.keras.layers.Dropout(0.5)(data)

data=tf.keras.layers.Dense(4)(data)
data=tf.keras.layers.Activation('relu')(data)
data=tf.keras.layers.Dropout(0.5)(data)

data=tf.keras.layers.Dense(1)(data)
output_data=tf.keras.layers.Activation('sigmoid')(data)

model=tf.keras.models.Model(inputs=input_data,outputs=output_data)

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
#model.summary()

model.fit(trianing_data,train_labels,epochs=10,batch_size=256,validation_data=(testing_data,test_labels))
'''

'''
#RNN
input_data=tf.keras.layers.Input(shape=(500,))

data=tf.keras.layers.Embedding(input_dim=10002,output_dim=32,input_length=500)(input_data)

data=tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(50))(data)

data=tf.keras.layers.Dense(1)(data)
output_data=tf.keras.layers.Activation('sigmoid')(data)

model=tf.keras.models.Model(inputs=input_data,outputs=output_data)

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

model.fit(trianing_data,train_labels,epochs=10,batch_size=256,validation_data=(testing_data,test_labels))
'''

'''
#LSTM
input_data=tf.keras.layers.Input(shape=(500,))

data=tf.keras.layers.Embedding(input_dim=10002,output_dim=32,input_length=500)(input_data)

data=tf.keras.layers.Embedding(input_dim=10002,output_dim=32,input_length=500)(input_data)

data=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50))(data)
data=tf.keras.layers.Dense(1)(data)
output_data=tf.keras.layers.Activation('sigmoid')(data)

model=tf.keras.models.Model(inputs=input_data,outputs=output_data)

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

model.fit(trianing_data,train_labels,epochs=10,batch_size=256,validation_data=(testing_data,test_labels))
'''

#1D CNN
input_data=tf.keras.layers.Input(shape=(500,))

data=tf.keras.layers.Embedding(input_dim=10002,output_dim=32,input_length=500)(input_data)

data=tf.keras.layers.Conv1D(50,kernel_size=3,activation='relu')(data)
data=tf.keras.layers.MaxPool1D(pool_size=2)(data)

data=tf.keras.layers.Conv1D(40,kernel_size=3,activation='relu')(data)
data=tf.keras.layers.MaxPool1D(pool_size=2)(data)

data=tf.keras.layers.Conv1D(30,kernel_size=3,activation='relu')(data)
data=tf.keras.layers.MaxPool1D(pool_size=2)(data)

data=tf.keras.layers.Conv1D(30,kernel_size=3,activation='relu')(data)
data=tf.keras.layers.MaxPool1D(pool_size=2)(data)

data=tf.keras.layers.Flatten()(data)

data=tf.keras.layers.Dense(20)(data)
data=tf.keras.layers.Dropout(0.5)(data)

data=tf.keras.layers.Dense(1)(data)
output_data=tf.keras.layers.Activation('sigmoid')(data)

model=tf.keras.models.Model(inputs=input_data,outputs=output_data)

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

model.fit(trianing_data,train_labels,epochs=10,batch_size=256,validation_data=(testing_data,test_labels))