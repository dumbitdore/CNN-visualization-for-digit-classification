"""
Test the CNN on MNIST dataset and show confusion matrix.
"""

import numpy as np
import seaborn
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import matplotlib.pyplot as pyplot

def run_test(model):
    # load MNIST (test data only)
    _, (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    # reshape images as 4-dimensional array
    test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

    # calculate predictions for test dataset
    predictions = model.predict(test_images)

    # convert predictions into class labels
    predicted_labels = np.argmax(predictions, axis=1)

    # create confusion matrix
    matrix = confusion_matrix(test_labels, predicted_labels)

    # visualize confusion matrix
    pyplot.figure(figsize=(5, 4))
    seaborn.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    pyplot.xlabel('Predicted')
    pyplot.ylabel('True value')
    pyplot.show()