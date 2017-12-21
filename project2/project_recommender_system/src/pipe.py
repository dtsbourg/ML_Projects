"""
CS-433 : Machine Learning
Project 2 -- Recommender Systems

Team :
* Dylan Bourgeois
* Antoine Mougeot
* Philippe Verbist

---

pipe.py : interface for heavy lifting.

Defines the main computational operations : fitting the model and using it to predict.
"""

import keras
from keras import layers
from keras import models
from keras import optimizers
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np

from utils import load_embedding, load_full_embedding

def fit(model, train_x, train_y, test_x, test_y, embedding=False, epochs=10, batch_size=512):
    """
    Run the fitting procedure on the training data.

    Args:
        model (Network): wrapper for the Keras model. see `models.py`
        train_x, train_y (list): training data
        test_x, test_y (list): test data
        embedding (bool): Are we using the embeddings ?
        epochs (int): Number of epochs
        batch_size (int): Batch size
    Returns:
        history (Keras.History): hold the training history (see https://keras.io/callbacks/#history)
    """
    training_data = [train_x.Item, train_x.User]
    test_data = [test_x.Item, test_x.User]

    if embedding is True:
        training_data = load_full_embedding(training_data)
        test_data     = load_full_embedding(test_data)

    callbacks = [EarlyStopping('val_loss', patience=6),
                 ModelCheckpoint('../res/model/'+model.description_str(), save_best_only=True),
                 ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=4, min_lr=0.0001)]

    history = model.model.fit(
        training_data, train_y,
        validation_data=(test_data, test_y),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks)

    return history

def predict(categorical_predictions, voting='weighted'):
    """
    Run the prediction procedure on the training data.

    Args:
        categorical_predictions ([float]): the predictions from the model. For each
                                           there are 5 confidence values (one per class)
        voting (str): The voting scheme used :
                        * 'weighted' = scale the class predictions by their confidence
                        * 'absolute' = choose the class with the highest confidence
    Returns:
        ratings ([float]): The predicted rating
    """
    if voting=='absolute':
        return [np.argmax(p)+1 for p in categorical_predictions]
    elif voting=='weighted':
        weighted = lambda p: np.dot(range(1,6), p)
        return [weighted(p) for p in categorical_predictions]
    else:
        raise NotImplementedError
