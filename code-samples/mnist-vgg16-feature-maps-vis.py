import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import img_to_array, array_to_img
import random

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = tf.image.grayscale_to_rgb(tf.convert_to_tensor(train_images)[..., None])
train_images = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48, 48))) for im in train_images])

base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(48, 48, 3))
base_model.summary()

for layer in base_model.layers:
    layer.trainable = False

layer_outputs = [layer.output for layer in base_model.layers]

feature_map_model = tf.keras.models.Model(inputs=base_model.input, outputs=layer_outputs)
feature_map_model.summary()

input_image = train_images[random.randint(0,10)]
input = tf.keras.utils.img_to_array(input_image)
input = input.reshape((1,) + input.shape)
input /= 255.0

feature_maps = feature_map_model.predict(input)

for feature_map in feature_maps:
    plt.imshow(feature_map[0, :, :, 0], cmap="gray")
    plt.show()
