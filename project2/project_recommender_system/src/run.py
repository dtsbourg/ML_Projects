import keras
from keras import layers
from keras import models
from keras import optimizers

import matplotlib.pyplot as plt
plt.style.use('ggplot')

import numpy as np

def fit(model, train_x, train_y, test_x, test_y, epochs=10, batch_size=512):
    history = model.fit(
        [train_x.Item, train_x.User], train_y,
        validation_data=([test_x.Item, test_x.User], test_y),
        batch_size=batch_size,
        epochs=epochs
    )
    return history

def plot(description, history, show=False):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    if show is False:
        plt.savefig('../res/img/' + description + '.png')

def predict(categorical_predictions, voting='weighted'):
    if voting=='absolute':
        return [np.argmax(p)+1 for p in categorical_predictions]
    elif voting=='weighted':
        weighted = lambda p: np.dot(p,np.argsort(p))/sum(p)+1
        return [weighted(p) for p in categorical_predictions]
