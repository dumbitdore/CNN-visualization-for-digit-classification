"""
Training and loading helpers for MNIST CNN.
"""

import tensorflow as tf
import os
import numpy as np
import random

from src.config import SEED
from src.model import build_model

# Ensure reproducibility
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

def train_and_save(epochs=5, batch_size=128, save_path='models/cnn_model_full.keras'):
    """
    trains the model, saves it to the given path and returns the history
    """

    # load mnist dataset (training images only)
    (train_images, train_labels), _ = tf.keras.datasets.mnist.load_data()

    # reshape into 4-dimensional array
    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255

    # get compiled model
    model = build_model()

    # training model with given epochs
    history = model.fit(
        train_images,
        train_labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2)  # 20% of the training images are used for validation during the training

    # create dir if not exists and save the whole model (not only weights but all parameters)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    return history

def load_model(model_path="models/cnn_model_full.keras"):
    model = build_model()

    # load model if file exist
    if os.path.exists(model_path):
        model.load_weights(model_path)
    else:
        raise FileNotFoundError(f"Model not found at {model_path}")
    return model