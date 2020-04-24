import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
import re
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard
from time import time

from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('spam.csv', encoding="latin1")

# Keeping only the neccessary columns
data = data[['v1','v2']]

data['v2'] = data['v2'].apply(lambda x: x.lower())
data['v2'] = data['v2'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

for idx, row in data.iterrows():
    row[0] = row[0].replace('rt', ' ')

max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['v2'].values)
X = tokenizer.texts_to_sequences(data['v2'].values)
X = pad_sequences(X)

embed_dim = 128
lstm_out = 196
def createmodel():
    model = Sequential()
    model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(3,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    return model
# print(model.summary())

# Model for spam.csv
def createmodel_spam():
    model = Sequential()
    model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    return model

labelencoder = LabelEncoder()
integer_encoded = labelencoder.fit_transform(data['v1'])
integer_decoded = labelencoder.inverse_transform(integer_encoded)
y = to_categorical(integer_encoded)

X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.33, random_state = 42)

# Part 2 the applying of GridSearchCV
"""
model = KerasClassifier(build_fn=createmodel,verbose=0)
batch_size = [10, 20]
epochs = [1, 2]
param_grid = dict(batch_size=batch_size, epochs=epochs)

from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X_train, Y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
"""

# Original batch_size and epochs
# batch_size = 32
# epochs = 1
# New batch size and epochs based on GridSearchCV
batch_size = 10
epochs = 1

# creating and fitting model for spam.csv, saving as 'spam_model.h5'
# model = createmodel_spam()
# tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
# model.fit(X_train, Y_train, epochs = epochs, batch_size=batch_size, verbose = 2)
# model.save('spam_model.h5')

from keras.models import load_model
# loading and visualizing and evaluating model
# model = load_model('spam_model.h5')
# score,acc = model.evaluate(X_test,Y_test,verbose=2,batch_size=batch_size)
# print(score)
# print(acc)
# print(model.metrics_names)


# Loading evaluating and testing model for Sentiment.csv
model = load_model('model.h5')

score,acc = model.evaluate(X_test,Y_test,verbose=2,batch_size=batch_size)
print(score)
print(acc)
print(model.metrics_names)

# Prediction of text sentiment printing the softmaxes of each category.
# tweet = ['A lot of good things are happening. We are respected again throughout the world, and thats a great thing.@realDonaldTrump']
tweet = ['Love. Connection. Experience. Thats great your great awesome stuff']
X = tokenizer.texts_to_sequences(tweet)
X = pad_sequences(X, maxlen=28)

print(tweet)
predict = model.predict(X)
print("Negative: ", end="")
print(predict[0][0])
print("Neutral: ", end="")
print(predict[0][1])
print("Positive: ", end="")
print(predict[0][2])