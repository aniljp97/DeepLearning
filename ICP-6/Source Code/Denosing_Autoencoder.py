from keras.layers import Input, Dense
from keras.models import Model

# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(784,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# "decoded" is the lossy reconstruction of the input
decoded = Dense(784, activation='sigmoid')(encoded)
# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)
# this model maps an input to its encoded representation
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
from keras.datasets import fashion_mnist
import numpy as np
(x_train, _), (x_test, _) = fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

#introducing noise
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

history = autoencoder.fit(x_train_noisy, x_train,
                epochs=10,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test_noisy, x_test_noisy))

#### my code ####

# visualize the input, noisy inputand reconstructed representation(denosed output)of the Denosing_Autoencoder using
# Matplotlib
from matplotlib import pyplot as plt

for i in range(1, 4):
    # input
    a = plt.subplot(3, 3, i)
    plt.imshow(x_test[i-1].reshape(28,28))
    a.get_xaxis().set_visible(False)
    a.get_yaxis().set_visible(False)

    # noisy input
    a = plt.subplot(3, 3, i + 3)
    plt.imshow(x_test_noisy[i-1].reshape(28,28))
    a.get_xaxis().set_visible(False)
    a.get_yaxis().set_visible(False)

    # reconstructed
    a = plt.subplot(3, 3, i + 6)
    predictions = autoencoder.predict(x_test_noisy)
    plt.imshow(predictions[i-1].reshape(28,28))
    a.get_xaxis().set_visible(False)
    a.get_yaxis().set_visible(False)

plt.show()

# plot loss and accuracy using the history object

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

# Visualize the compressed version of the input data in the middle layer
encoded_imgs = encoder.predict(x_test)
plt.imshow(encoded_imgs[0].reshape(32, 1))
plt.show()