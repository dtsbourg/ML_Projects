"""
CS-433 : Machine Learning
Project 2 -- Recommender Systems

Team :
* Dylan Bourgeois
* Antoine Mougeot
* Philippe Verbist

---

models.py : main model definition module.

Here are defined the main models that we use in this project. They are all
subclassed off of `Network`, but other classes of models are welcome here too.
They all return a compiled Keras model, and require a `model_func` which
explicits their architecture.
"""

import keras
from keras import layers
from keras import models
from keras import optimizers
from keras import regularizers
import keras.backend as K
import datetime
import numpy as np


class Network(object):
    """
    Defines the Network abstract class, exposing the structure
    of the models we will be developping.

    Attributes:
        k_features (int): number of features in the embedding of the input layer.
        n_items, n_users (int): number of items / users in the dataset.
        n_classes (int): 5 if we are training for categorical, 1 for regression
        n_train_samples, n_test_samples (int): number of samples in the data set
        optimizer (str or Keras.Optimizer): Optimizer used in the model (usually 'sgd' or 'adam')
        loss (str or Keras.Loss): Loss function (usually 'mse' or 'categorical_crossentropy')
        EMBED_FEAT (int): Number of features in the provided embeddings
        DROPOUT (float): Dropout rate in the network.
        EMBED_REG (float): Regularisation parameter for the learned embeddings ($\lambda$)
        model (Keras.Model): The actual compiled Keras model, defined in a model_func
        model_type (str): Model name for bookkeeping.
        descr (str): Precise model description for bookkeeping.
    """
    def __init__(self, k_features=64, n_items=10000, n_users=1000, n_train_samples=10, n_test_samples=10, n_classes=5, optimizer='sgd', loss='categorical_crossentropy'):
        super(Network, self).__init__()
        self.k_features = k_features
        self.n_items = n_items
        self.n_users = n_users
        self.n_classes = n_classes
        self.n_train_samples = n_train_samples
        self.n_test_samples  = n_test_samples
        self.optimizer = optimizer
        self.loss = loss
        self.EMBED_FEAT = 64
        self.DROPOUT    = 0.5
        self.EMBED_REG  = 1e-4
        self.model = self.model_func()
        self.model_type = "Basic"
        self.descr = self.description_str()

    def model_func(self):
        """
        Virtual function for the Network class, in which the user must
        create and return the Keras Model that this Network wraps.
        """
        raise NotImplementedError()

    def description_str(self, suffix="", uid=False):
        """
        Helper to generate a unique identifier for saving the model.
        """
        model_type = self.model_type + "_"
        model_size = str(self.n_train_samples) + "_train_" + str(self.n_test_samples) + "_test_" + str(self.k_features) + "_features_"
        if isinstance(self.optimizer, str):
            model_params = self.optimizer + "_" + self.loss + "_"
        else:
            model_params = "Adam" + "_" + self.loss + "_"
        model_categ = ""
        if self.n_classes > 1:
            model_categ = "categorical_"
        model_uid = datetime.datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
        return model_type + model_size + model_params + model_categ + suffix

class ShallowNetwork(Network):
    """Shallow Neural Network"""
    def __init__(self, *args, **kwargs):
        super(ShallowNetwork, self).__init__(*args, **kwargs)
        self.model_type = "Shallow_Final"
        self.descr = self.description_str()

    def model_func(self):
        input_i = layers.Input(shape=[1])
        i = layers.Embedding(self.n_items+1, self.k_features)(input_i)
        i = layers.Flatten()(i)

        input_u = layers.Input(shape=[1])
        u = layers.Embedding(self.n_users+1, self.k_features)(input_u)
        u = layers.Flatten()(u)

        input_i_spectral = layers.Input(shape=[self.EMBED_FEAT])
        input_u_spectral = layers.Input(shape=[self.EMBED_FEAT])

        input_i_lle = layers.Input(shape=[self.EMBED_FEAT])
        input_u_lle = layers.Input(shape=[self.EMBED_FEAT])

        input_i_fa = layers.Input(shape=[self.EMBED_FEAT])
        input_u_fa = layers.Input(shape=[self.EMBED_FEAT])

        input_i_nmf = layers.Input(shape=[self.EMBED_FEAT])
        input_u_nmf = layers.Input(shape=[self.EMBED_FEAT])

        nn = layers.concatenate([i, u,
                                 input_i_spectral, input_u_spectral,
                                 input_i_lle, input_u_lle,
                                 input_i_fa, input_u_fa,
                                 input_i_nmf, input_u_nmf])

        nn = layers.Dense(512, activation='relu')(nn)
        nn = layers.Dropout(self.DROPOUT)(nn)
        nn = layers.normalization.BatchNormalization()(nn)

        nn = layers.Dense(128, activation='relu')(nn)

        output = layers.Dense(self.n_classes, activation='softmax')(nn)

        model = models.Model([input_i, input_u,
                             input_i_spectral, input_u_spectral,
                             input_i_lle, input_u_lle,
                             input_i_fa, input_u_fa,
                             input_i_nmf, input_u_nmf],
                             output)
        model.compile(optimizer=self.optimizer, loss=self.loss)
        return model


class DeepNetwork(Network):
    """
    Deep Neural Network
    This is the model with which we obtained the best results.
    """
    def __init__(self, *args, **kwargs):
        super(DeepNetwork, self).__init__(*args, **kwargs)
        self.model_type = "Deep_Full_Final"
        self.descr = self.description_str()

    def model_func(self):
        input_i = layers.Input(shape=[1])
        i = layers.Embedding(self.n_items + 1, self.k_features, embeddings_regularizer=regularizers.l2(self.EMBED_REG))(input_i)
        i = layers.Flatten()(i)

        input_u = layers.Input(shape=[1])
        u = layers.Embedding(self.n_users + 1, self.k_features, embeddings_regularizer=regularizers.l2(self.EMBED_REG))(input_u)
        u = layers.Flatten()(u)

        input_i_spectral = layers.Input(shape=[self.EMBED_FEAT])
        input_u_spectral = layers.Input(shape=[self.EMBED_FEAT])

        input_i_lle = layers.Input(shape=[self.EMBED_FEAT])
        input_u_lle = layers.Input(shape=[self.EMBED_FEAT])

        input_i_fa = layers.Input(shape=[self.EMBED_FEAT])
        input_u_fa = layers.Input(shape=[self.EMBED_FEAT])

        input_i_nmf = layers.Input(shape=[self.EMBED_FEAT])
        input_u_nmf = layers.Input(shape=[self.EMBED_FEAT])

        nn = layers.concatenate([i, u,
                                 input_i_spectral, input_u_spectral,
                                 input_i_lle, input_u_lle,
                                 input_i_fa, input_u_fa,
                                 input_i_nmf, input_u_nmf])

        nn = layers.Dense(1024, activation='relu')(nn)
        nn = layers.Dropout(self.DROPOUT)(nn)
        nn = layers.normalization.BatchNormalization()(nn)
        nn = layers.Dense(512, activation='relu')(nn)
        nn = layers.Dropout(self.DROPOUT)(nn)
        nn = layers.normalization.BatchNormalization()(nn)
        nn = layers.Dense(256, activation='relu')(nn)
        nn = layers.Dropout(self.DROPOUT)(nn)
        nn = layers.normalization.BatchNormalization()(nn)
        nn = layers.Dense(128, activation='relu')(nn)

        output = layers.Dense(self.n_classes, activation='softmax')(nn)

        model = models.Model([input_i, input_u,
                              input_i_spectral, input_u_spectral,
                              input_i_lle, input_u_lle,
                              input_i_fa, input_u_fa,
                              input_i_nmf, input_u_nmf], output)

        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])
        return model

class DenseNetwork(Network):
    """Dense Neural Network"""
    def __init__(self, *args, **kwargs):
        super(DenseNetwork, self).__init__(*args, **kwargs)
        self.model_type = "Dense_Final"

    def model_func(self):
        input_i = layers.Input(shape=[1])
        i = layers.Embedding(self.n_items + 1, self.k_features, embeddings_regularizer=regularizers.l2(self.EMBED_REG))(input_i)
        i = layers.Flatten()(i)

        input_u = layers.Input(shape=[1])
        u = layers.Embedding(self.n_users + 1, self.k_features, embeddings_regularizer=regularizers.l2(self.EMBED_REG))(input_u)
        u = layers.Flatten()(u)

        input_i_spectral = layers.Input(shape=[self.EMBED_FEAT])
        input_u_spectral = layers.Input(shape=[self.EMBED_FEAT])

        input_i_lle = layers.Input(shape=[self.EMBED_FEAT])
        input_u_lle = layers.Input(shape=[self.EMBED_FEAT])

        input_i_fa = layers.Input(shape=[self.EMBED_FEAT])
        input_u_fa = layers.Input(shape=[self.EMBED_FEAT])

        input_i_nmf = layers.Input(shape=[self.EMBED_FEAT])
        input_u_nmf = layers.Input(shape=[self.EMBED_FEAT])

        nn = layers.concatenate([i, u,
                                 input_i_spectral, input_u_spectral,
                                 input_i_lle, input_u_lle,
                                 input_i_fa, input_u_fa,
                                 input_i_nmf, input_u_nmf])

        nn = layers.Dense(256, activation='relu')(nn)
        nn = layers.Dropout(self.DROPOUT)(nn)
        nn = layers.normalization.BatchNormalization()(nn)

        nn = layers.Dense(256, activation='relu')(nn)
        nn = layers.Dropout(self.DROPOUT)(nn)
        nn = layers.normalization.BatchNormalization()(nn)

        nn = layers.Dense(256, activation='relu')(nn)

        output = layers.Dense(self.n_classes, activation='softmax')(nn)

        model = models.Model([input_i, input_u,
                              input_i_spectral, input_u_spectral,
                              input_i_lle, input_u_lle,
                              input_i_fa, input_u_fa,
                              input_i_nmf, input_u_nmf],
                              output)

        model.compile(optimizer=self.optimizer, loss=self.loss)
        return model
