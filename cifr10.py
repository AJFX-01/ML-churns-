import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, Input
import matplotlib.pyplot as plt
import numpy as np


(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
y_train = y_train.reshape(-1,)

classes = ['airplane','automobile','bird','bird','cat','deer','dog','frog','horse','ship','truck']

# def plot_sample(x, y, index):
#     plt.figure(figsize= (15, 2))
#     plt.imshow(x[index])
#     plt.xlabel(classes[y[index]])
#
# plot_sample(x_train, y_train, 0)
# plot_sample(x_train, y_train, 1)

x_train = x_train / 255
x_text = x_test / 255
# #Using the ANN model Artificial Neural Network
# ann = models.Sequential([
#         layers.Flatten(input_shape=(32,32,3)),
#         layers.Dense(3000, activation='relu'),
#         layers.Dense(1000, activation='relu'),
#         layers.Dense(10, activation='sigmoid')
# ])
# cnn.compile(optimizer='SGD',
#             loss='sparse_categorical_crossentropy',
#             metrics=['accuracy'])
#
# cnn.fit(x_train, y_train, epochs=5)
# Using the CNN model
cnn = models.Sequential([
        # CNN layers
        # first_layers
        layers.Conv2D(filters=32, kernel_size=(4, 4), activation='relu'),
        layers.MaxPooling2D(2, 2),

        # Second Layers
        layers.Conv2D(filters=64, kernel_size=(4, 4), activation='relu'),
        layers.MaxPooling2D(2, 2),

        # (Dense network)
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')

])
# Functional API
# def my_model():
#     inputs = keras.Input(shape=(32, 32, 3))
#     x = layers.Conv2D(32, 3)(inputs)
#     x = layers.BatchNormalization()(x)
#     x = keras.activations.relu(x)
#     x = layers.MaxPooling2D()(x)
#     x = layers.Conv2D(64, 5, padding='same')(x)
#     x = layers.BatchNormalization()(x)
#     x = keras.activations.relu(x)
#     x = layers.Conv2D(128, 3)(x)
#     x = layers.BatchNormalization()(x)
#     x = keras.activations.relu(x)
#     x = layers.Flatten()(x)
#     x = layers.Dense(64, activation='relu')(x)
#     outputs = layers.Dense(10)(x)
#     model = keras.Model(inputs=inputs, outputs=outputs)
#     return model
#
# model = my_model()
#
# model.compile(optimizer='adam',
#             loss='sparse_categorical_crossentropy',
#             metrics=['accuracy'])

cnn.fit(x_train, y_train, batch_size=64, epochs=10, verbose=2)

experi = cnn.evaluate(x_text, y_test)
print(experi)