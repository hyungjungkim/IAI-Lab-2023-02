import tensorflow as tf
from tensorflow import keras
import random
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0


model = keras.Sequential()
model.add(keras.layers.SimpleRNN(8, input_shape=(None, 28)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(10))
model.summary()

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer="sgd",
    metrics=["accuracy"],
)

model.fit(
    x_train, y_train, validation_data=(x_test, y_test), batch_size=64, epochs=1
)

index = random.randint(0, 9)
sample, sample_label = x_test[index], y_test[index]

result = tf.argmax(model.predict(tf.expand_dims(sample, 0)), axis=1)
print(result.numpy())

plt.imshow(sample, cmap='gray')
plt.show()
