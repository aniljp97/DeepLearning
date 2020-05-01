from keras.layers import Input, Dense
from keras.models import Model

# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(784,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)

# added one more hidden layer to autoencoder
encoded = Dense(encoding_dim, activation='relu')(encoded)

# "decoded" is the lossy reconstruction of the input
decoded = Dense(784, activation='sigmoid')(encoded)
# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)
# this model maps an input to its encoded representation
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
from keras.datasets import mnist, fashion_mnist
import numpy as np
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.astype(   'float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

autoencoder.fit(x_train, x_train,
                epochs=5,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

#### my code ####

# visualize the input and reconstructed representation of the autoencoder using Matplotlib of 3 images
from matplotlib import pyplot as plt

for i in range(1,4):
    plt.title("1st row = Input, 2nd row = Output")
    # input
    a = plt.subplot(2, 3, i)
    plt.imshow(x_test[i-1].reshape(28,28))
    a.get_xaxis().set_visible(False)
    a.get_yaxis().set_visible(False)

    # reconstructed
    a = plt.subplot(2, 3, i + 3)
    predictions = autoencoder.predict(x_test)
    plt.imshow(predictions[i-1].reshape(28,28))
    a.get_xaxis().set_visible(False)
    a.get_yaxis().set_visible(False)

plt.show()

