import keras
from keras import layers
from keras import models
from keras import optimizers
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

import matplotlib.pyplot as plt
plt.style.use('ggplot')

import numpy as np

def fit(model, train_x, train_y, test_x, test_y, user_embedding=False, item_embedding=False, epochs=10, batch_size=512):
    training_data = [train_x.Item, train_x.User]
    test_data = [test_x.Item, test_x.User]

    if item_embedding is True:
        i_emb = np.load('../data/embeddings/items_t_sne.npy')
        train_emb_i = u_emb[train_x.Item - 1]
        training_data += [train_emb_i]

        i_emb_test = np.load('../data/embeddings/items_t_sne_test.npy')
        test_emb_i = i_emb_test[test_x.Item - 1]
        test_data += [test_emb_i]

    if user_embedding is True:
        u_emb = np.load('../data/embeddings/users_t_sne.npy')
        train_emb_u = u_emb[train_x.User - 1]
        training_data += [train_emb_u]

        u_emb_test = np.load('../data/embeddings/users_t_sne_test.npy')
        test_emb_u = u_emb_test[test_x.User - 1]
        test_data += [test_emb_u]

    callbacks = [EarlyStopping('val_loss', patience=5),
                 ModelCheckpoint('../res/model/'+model.description_str(), save_best_only=True)]

    history = model.model.fit(
        training_data, train_y,
        validation_data=(test_data, test_y),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks
    )

    model.descr = model.description_str(suffix= str(max(history.epoch)+1) + "_epochs_", uid=True)

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
