"""
CS-433 : Machine Learning
Project 2 -- Recommender Systems

Team :
* Dylan Bourgeois
* Antoine Mougeot
* Philippe Verbist

---

run.py : interface for heavy lifting.

Defines the main computational operations : fitting the model and using it to predict.
"""

import keras
from keras import layers
from keras import models
from keras import optimizers
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

from utils import load_embedding

import numpy as np

def fit(model, train_x, train_y, test_x, test_y, embedding=False, epochs=10, batch_size=512):
    training_data = [train_x.Item, train_x.User]
    test_data = [test_x.Item, test_x.User]

    if embedding is True:
        train_data += load_embedding(path='../data/embeddings/items_spectral_64.npy', idx=train_x.Item)
        train_data += load_embedding(path='../data/embeddings/users_spectral_64.npy', idx=train_x.User)
        train_data += load_embedding(path='../data/embeddings/items_lle_64.npy',      idx=train_x.Item)
        train_data += load_embedding(path='../data/embeddings/users_lle_64.npy',      idx=train_x.User)
        train_data += load_embedding(path='../data/embeddings/items_fa_64.npy',       idx=train_x.Item)
        train_data += load_embedding(path='../data/embeddings/users_fa_64.npy',       idx=train_x.User)
        train_data += load_embedding(path='../data/embeddings/items_nmf_64.npy',      idx=train_x.Item)
        train_data += load_embedding(path='../data/embeddings/users_nmf_64.npy',      idx=train_x.User)

        test_data += load_embedding(path='../data/embeddings/items_spectral_64.npy', idx=test_x.Item)
        test_data += load_embedding(path='../data/embeddings/users_spectral_64.npy', idx=test_x.User)
        test_data += load_embedding(path='../data/embeddings/items_lle_64.npy',      idx=test_x.Item)
        test_data += load_embedding(path='../data/embeddings/users_lle_64.npy',      idx=test_x.User)
        test_data += load_embedding(path='../data/embeddings/items_fa_64.npy',       idx=test_x.Item)
        test_data += load_embedding(path='../data/embeddings/users_fa_64.npy',       idx=test_x.User)
        test_data += load_embedding(path='../data/embeddings/items_nmf_64.npy',      idx=test_x.Item)
        test_data += load_embedding(path='../data/embeddings/users_nmf_64.npy',      idx=test_x.User)

    callbacks = [EarlyStopping('val_loss', patience=5),
                 ModelCheckpoint('../res/model/'+model.description_str(), save_best_only=True)]

    history = model.model.fit(
        training_data, train_y,
        validation_data=(test_data, test_y),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks)

    model.descr = model.description_str(suffix= str(max(history.epoch)+1) + "_epochs_", uid=True)

    return history

def predict(categorical_predictions, voting='weighted'):
    if voting=='absolute':
        return [np.argmax(p)+1 for p in categorical_predictions]
    elif voting=='weighted':
        weighted = lambda p: np.dot(range(1,6), p)
        return [weighted(p) for p in categorical_predictions]
    else:
        raise NotImplementedError
