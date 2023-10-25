# Dataset of Mnist
import tensorflow as tf
import numpy as np

# Modeling an autoencoder
import keras
from keras import layers

# Plotting
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

train_images = train_images.reshape(train_images.shape[0], np.prod(train_images.shape[1:]))
test_images = test_images.reshape(test_images.shape[0], np.prod(test_images.shape[1:]))

# print(train_images.shape)
# print(test_images.shape)

encoding_dim = 32

input_layer = keras.Input(shape = (train_images.shape[1],))
encoded = layers.Dense(encoding_dim, activation='relu')(input_layer)
decoded = layers.Dense(train_images.shape[1], activation='sigmoid')(encoded)

autoencoder = keras.Model(input_layer, decoded)

encoder = keras.Model(input_layer, encoded)
encoded_input = keras.Input(shape=(encoding_dim,))

decoder_layer = autoencoder.layers[-1]
decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.summary()

# Training the autoencoder
autoencoder.fit(train_images, train_images, epochs=50, batch_size=256, shuffle=True, validation_data=(test_images, test_images))

encoded_imgs = encoder.predict(test_images)
decoded_imgs = decoder.predict(encoded_imgs)

n = 5  # How many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(test_images[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

