from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.layers import Embedding
from keras.layers import Flatten
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer


twenty_train = fetch_20newsgroups(subset='train', shuffle=True, categories=['alt.atheism',
                                                                            'sci.space',
                                                                            'rec.autos',
                                                                            'rec.motorcycles'])
tfidf_Vect = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)

df = pd.read_csv('imdb_master.csv', encoding='latin-1')
sentences = df['review'].values
y = df['label'].values

max_review_len = max([len(s.split()) for s in sentences])

num_words = 1000

print("tokenizing...")
#tokenizing data
tokenizer = Tokenizer(num_words=max_review_len)
tokenizer.fit_on_texts(sentences)
#getting the vocabulary of data
print("getting vocabulary of data...")
sentences = tokenizer.texts_to_matrix(sentences)

print("encoding target column...")
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

print("spliting training and test...")
X_train, X_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)

vocab_size = len(tokenizer.word_index)+1
print("MRL:", max_review_len)
print("VS:", vocab_size)

model = Sequential()
model.add(Embedding(vocab_size, 8, input_length=max_review_len))
model.add(Flatten())
### Made the input_dim of the first layer to equal 2000 to match the word number assigned for the tokenizer ###
#model.add(layers.Dense(300, input_dim=max_review_len, activation='relu'))
model.add(layers.Dense(300, activation='relu'))
### Changed the neuron number to 10000 and the activation to softmax ###
model.add(layers.Dense(10000, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['acc'])
history=model.fit(X_train,y_train, epochs=5, verbose=True, validation_data=(X_test,y_test), batch_size=256)


# Plotting accuracy and loss
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.title('Training Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

plt.plot(history.history['loss'])
plt.title('Training Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

plt.plot(history.history['val_accuracy'])
plt.title('Validation Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

plt.plot(history.history['val_loss'])
plt.title('Validation Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
