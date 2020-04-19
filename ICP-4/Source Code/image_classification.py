# Simple CNN model for CIFAR-10
import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import tensorflow as tf
#K.set_image_dim_ordering('th')

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# load data
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
# normalize inputs from 0-255 to 0.0-1.0
X_train = train_images.astype('float32')
X_test = test_images.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

X_train = np.moveaxis(X_train, -1, 1)
X_test = np.moveaxis(X_test, -1, 1)

# one hot encode outputs
y_train = np_utils.to_categorical(train_labels)
y_test = np_utils.to_categorical(test_labels)
num_classes = y_test.shape[1]
# Create the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

# Compile model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#model.summary()
# Fit the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)
# Final evaluation of the model

model.save('./model' + '.h5')


# Plot history of model training
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

"""
# Load saved model
save_model = tf.keras.models.load_model('./model.h5')
#save_model.summary()

scores = save_model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# show 5 images first with their ground truths, then the models prediction of them
for i in range(5):
    pred = save_model.predict_classes(X_test[[i], :])[0]

    plt.imshow(test_images[i,:,:],cmap='gray')
    plt.title('Ground Truth : {}'.format(test_labels[i]))
    plt.show()

    plt.imshow(test_images[i,:,:],cmap='gray')
    plt.title('Prediction : {}'.format(pred))
    plt.show() 
"""