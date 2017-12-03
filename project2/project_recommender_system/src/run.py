import keras
from keras import layers
from keras import models
from keras import optimizers

import matplotlib.pyplot as plt
plt.style.use('ggplot')

def fit(model, train_x, train_y, test_x, test_y, epochs=10, batch_size=512):
    history = model.fit(
        [train_x.Item, train_x.User], train_y,
        validation_data=([test_x.Item, test_x.User], test_y),
        #batch_size=2048,
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
