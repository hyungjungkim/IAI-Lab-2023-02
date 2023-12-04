import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

image_size = 28*28
train_images = train_images.reshape(train_images.shape[0], image_size)
test_images = test_images.reshape(test_images.shape[0], image_size)

num_classes = 10
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes)

activation='sigmoid'

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(image_size,)))
model.add(tf.keras.layers.Dense(units=8, activation=activation))
model.add(tf.keras.layers.Dense(units=8, activation=activation))
model.add(tf.keras.layers.Dense(units=8, activation=activation))
model.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=tf.keras.losses.CategoricalCrossentropy(), 
              metrics=['acc'])

epoch = 30
batch_size = 128
history = model.fit(train_images, train_labels, 
                    batch_size=batch_size, epochs=epoch, 
                    verbose=False, validation_split=.2)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy - layers 8 8 8 30')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='best')
plt.show()