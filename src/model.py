"""
CNN model definition for MNIST.
"""

from tensorflow.keras import Model, layers, Input

def build_model():
    # input
    inputs = Input(shape=(28, 28, 1), name="input")

    # convolutional clock 1
    conv1 = layers.Conv2D(32, (3, 3), activation="relu", name="conv1")(inputs)
    pool1 = layers.MaxPooling2D((2, 2), name="pool1")(conv1)

    # Convolutional Block 2
    conv2 = layers.Conv2D(64, (3, 3), activation="relu", name="conv2")(pool1)
    pool2 = layers.MaxPooling2D((2, 2), name="pool2")(conv2)

    # Convolutional Block 3
    conv3 = layers.Conv2D(64, (3, 3), activation="relu", name="conv3")(pool2)

    # Dense Part
    flatten_layer = layers.Flatten(name="flatten")(conv3)
    dense1 = layers.Dense(64, activation="relu", name="dense1")(flatten_layer)
    outputs = layers.Dense(10, activation="softmax", name="output")(dense1)

    # build model
    model = Model(inputs=inputs, outputs=outputs, name="mnist_cnn")

    # compile model
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model