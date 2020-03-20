import pandas
from keras.models import Sequential
from keras.layers.core import Dense, Activation

# load dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


dataset = pd.read_csv("diabetes.csv", header=None).values

X_train, X_test, Y_train, Y_test = train_test_split(dataset[:,0:8], dataset[:,8],
                                                    test_size=0.25, random_state=87)
np.random.seed(155)

# first model, source code
first_nn = Sequential() # create model
first_nn.add(Dense(20, input_dim=8, activation='relu')) # hidden layer
first_nn.add(Dense(1, activation='sigmoid')) # output layer
first_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
first_nn_fitted = first_nn.fit(X_train, Y_train, epochs=100,
                                     initial_epoch=0)
print(first_nn.summary())


# second model, added a layer
second_nn = Sequential() # create model
second_nn.add(Dense(20, input_dim=8, activation='relu')) # hidden layer
second_nn.add(Dense(15, activation="relu"))
second_nn.add(Dense(1, activation='sigmoid')) # output layer
second_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
second_nn_fitted = second_nn.fit(X_train, Y_train, epochs=100,
                                     initial_epoch=0)
print(second_nn.summary())


# third model, added another layer
third_nn = Sequential() # create model
third_nn.add(Dense(20, input_dim=8, activation='relu')) # hidden layer
third_nn.add(Dense(15, activation="relu"))
third_nn.add(Dense(20, activation="relu"))
third_nn.add(Dense(1, activation='sigmoid')) # output layer
third_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
third_nn_fitted = third_nn.fit(X_train, Y_train, epochs=100,
                                     initial_epoch=0)
print(third_nn.summary())


# Comparing accuarcy of the 3 models
print("First Model evaluation:")
print(first_nn.evaluate(X_test, Y_test))

print("Second Model evaluation:")
print(second_nn.evaluate(X_test, Y_test))

print("Third Model evaluation:")
print(third_nn.evaluate(X_test, Y_test))

print("No noticeable improvement, third model losses accuracy.")

input()

# Working with the Breast Cancer detection dataset now
# Getting evaluation before normalizing dataset
data = pd.read_csv("breastcancer.csv")

# Changing the target 'diagnosis' to binary so it is numerical
diagnosis_dict = {"M": 0, "B": 1}
diagnosis_col = []
for d in data["diagnosis"]:
    diagnosis_col.append(diagnosis_dict[d])

data["diagnosis"] = diagnosis_col


df = data.values
X = df[:,2:32]
Y = df[:,1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.25, random_state=87)

model = Sequential() # create model
model.add(Dense(20, input_dim=30, activation='relu')) # hidden layer
model.add(Dense(1, activation='sigmoid')) # output layer
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
fit_model = model.fit(X_train, Y_train, epochs=100,
                                     initial_epoch=0)


# Evaluating model now on a normalized dataset
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(data)

df = data.values
X = df[:,2:32]
Y = df[:,1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.25, random_state=87)

model_n = Sequential() # create model
model_n.add(Dense(20, input_dim=30, activation='relu')) # hidden layer
model_n.add(Dense(1, activation='sigmoid')) # output layer
model_n.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
fit_model = model_n.fit(X_train, Y_train, epochs=100,
                                     initial_epoch=0)


print("Before normalization:", model.evaluate(X_test, Y_test))
print("After normalization:", model_n.evaluate(X_test, Y_test))

print("Hard to see much change in the accuracy. It varies for each run. For a more complex and extensive dataset "
      ", normalization could make a big difference.")