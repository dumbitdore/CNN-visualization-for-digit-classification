"""
CNN model definition for MNIST.
"""

from tensorflow.keras import models, layers, Input

def build_model():
    cnn_model = models.Sequential()

    cnn_model.add(Input(shape=(28, 28, 1)))

    # convolutional layer, relu for nonlinearity
    cnn_model.add(layers.Conv2D(32, (3, 3), activation='relu'))

    cnn_model.add(layers.MaxPooling2D((2, 2)))
    cnn_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    cnn_model.add(layers.MaxPooling2D((2, 2)))
    cnn_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    cnn_model.add(layers.Flatten())
    cnn_model.add(layers.Dense(64, activation='relu'))
    cnn_model.add(layers.Dense(10, activation='softmax'))

    cnn_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',  # for multicategorical classification problems with integer labels
        metrics=['accuracy'])

    return cnn_model