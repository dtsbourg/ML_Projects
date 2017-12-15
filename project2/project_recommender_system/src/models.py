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
    """docstring for Network."""
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
        self.model = self.model_func()
        self.model_type = "Basic"
        self.descr = self.description_str()
        self.embed_dim = None

    def model_func(self):
        raise NotImplementedError()

    def description_str(self, suffix="", uid=False):
        model_type = self.model_type + "_"
        model_size = str(self.n_train_samples) + "_train_" + str(self.n_test_samples) + "_test_" + str(self.k_features) + "_features_"
        if isinstance(self.optimizer, str):
            model_params = self.optimizer + "_" + self.loss + "_"
        else:
            model_params = "SGD" + "_" + self.loss + "_"
        model_categ = ""
        if self.n_classes > 1:
            model_categ = "categorical_"
        model_uid = ""
        if uid is True:
            model_uid = datetime.datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
        return model_type + model_size + model_params + model_categ + suffix

class ShallowNetwork(Network):
    """docstring for ShallowNetwork."""
    def __init__(self, *args, **kwargs):
        super(ShallowNetwork, self).__init__(*args, **kwargs)
        self.model_type = "Shallow"

    def model_func(self):
        input_i = layers.Input(shape=[1])
        i = layers.Embedding(self.n_items+1, self.k_features)(input_i)
        i = layers.Flatten()(i)
        i = layers.normalization.BatchNormalization()(i)

        input_u = layers.Input(shape=[1])
        u = layers.Embedding(self.n_users+1, self.k_features)(input_u)
        u = layers.Flatten()(u)
        u = layers.normalization.BatchNormalization()(u)

        nn = layers.concatenate([i, u])

        nn = layers.Dense(512, activation='relu')(nn)
        nn = layers.Dropout(0.5)(nn)
        nn = layers.normalization.BatchNormalization()(nn)

        nn = layers.Dense(128, activation='relu')(nn)

        output = layers.Dense(self.n_classes, activation='softmax')(nn)

        model = models.Model([input_i, input_u], output)
        model.compile(optimizer=self.optimizer, loss=self.loss)
        return model

class ShallowNetwork2(Network):
    """docstring for ShallowNetwork2."""
    def __init__(self, *args, **kwargs):
        super(ShallowNetwork, self).__init__(*args, **kwargs)
        self.model_type = "Shallow2"

    def model_func(self):
        input_i = layers.Input(shape=[1])
        i = layers.Embedding(self.n_items+1, self.k_features, embeddings_regularizer=l2(0.0001))(input_i)
        input_u = layers.Input(shape=[1])
        u = layers.Embedding(self.n_users+1, self.k_features, embeddings_regularizer=l2(0.0001))(input_u)

        nn = layers.concatenate([i, u])

        nn = Flatten()(nn)
        nn = Dropout(0.25)(nn)
        nn = Dense(64, activation='relu')(nn)
        nn = Dropout(0.75)(nn)

        output = Dense(1)(nn)

        model = models.Model([input_i, input_u], output)
        model.compile(Adam(0.001), loss='mse')

        return model


class DeepNetwork(Network):
    """docstring for DeepNetwork."""
    def __init__(self, *args, **kwargs):
        super(DeepNetwork, self).__init__(*args, **kwargs)
        self.model_type = "Simple_Deep"

    def model_func(self):
        input_i = layers.Input(shape=[1])
        i = layers.Embedding(self.n_items + 1, self.k_features)(input_i)
        i = layers.Flatten()(i)
        i = layers.normalization.BatchNormalization()(i)

        input_u = layers.Input(shape=[1])
        u = layers.Embedding(self.n_users + 1, self.k_features)(input_u)
        u = layers.Flatten()(u)
        u = layers.normalization.BatchNormalization()(u)

        nn = layers.concatenate([i, u])

        nn = layers.Dense(1024, activation='relu')(nn)
        nn = layers.Dropout(0.5)(nn)
        nn = layers.normalization.BatchNormalization()(nn)
        nn = layers.Dense(512, activation='relu')(nn)
        nn = layers.Dropout(0.5)(nn)
        nn = layers.normalization.BatchNormalization()(nn)
        nn = layers.Dense(256, activation='relu')(nn)
        nn = layers.Dropout(0.5)(nn)
        nn = layers.normalization.BatchNormalization()(nn)
        nn = layers.Dense(128, activation='relu')(nn)
        nn = layers.Dropout(0.5)(nn)
        nn = layers.normalization.BatchNormalization()(nn)
        nn = layers.Dense(64, activation='relu')(nn)

        output = layers.Dense(5, activation='softmax')(nn)

        model = models.Model([input_i, input_u], output)
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])
        return model

class DeepNetworkFeat(Network):
    """docstring for DeepNetworkFeat."""
    def __init__(self, *args, **kwargs):
        super(DeepNetworkFeat, self).__init__(*args, **kwargs)
        self.model_type = "Deep_Full_Feat"

    def model_func(self):
        input_i = layers.Input(shape=[1])
        i = layers.Embedding(self.n_items + 1, self.k_features)(input_i)
        i = layers.Flatten()(i)
        i = layers.normalization.BatchNormalization()(i)

        input_u = layers.Input(shape=[1])
        u = layers.Embedding(self.n_users + 1, self.k_features)(input_u)
        u = layers.Flatten()(u)
        u = layers.normalization.BatchNormalization()(u)

        input_i_spectral = layers.Input(shape=[128])
        im_spectral = layers.normalization.BatchNormalization()(input_i_spectral)
        input_u_spectral = layers.Input(shape=[128])
        um_spectral = layers.normalization.BatchNormalization()(input_u_spectral)

        input_i_lle = layers.Input(shape=[128])
        im_lle = layers.normalization.BatchNormalization()(input_i_lle)
        input_u_lle = layers.Input(shape=[128])
        um_lle = layers.normalization.BatchNormalization()(input_u_lle)

        nn = layers.concatenate([i, u, im_spectral, um_spectral, im_lle, um_lle])

        nn = layers.Dense(1024, activation='relu')(nn)
        nn = layers.Dropout(0.5)(nn)
        nn = layers.normalization.BatchNormalization()(nn)
        nn = layers.Dense(512, activation='relu')(nn)
        nn = layers.Dropout(0.5)(nn)
        nn = layers.normalization.BatchNormalization()(nn)
        nn = layers.Dense(256, activation='relu')(nn)
        nn = layers.Dropout(0.5)(nn)
        nn = layers.normalization.BatchNormalization()(nn)
        nn = layers.Dense(128, activation='relu')(nn)

        output = layers.Dense(5, activation='softmax')(nn)

        model = models.Model([input_i, input_u, input_i_spectral, input_u_spectral, input_i_lle, input_u_lle], output)
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])
        return model



# class DeepNetworkFeatRegCustLoss(Network):
#     """docstring for DeepNetworkFeatCustLoss."""
#     def __init__(self, *args, **kwargs):
#         super(DeepNetworkFeatRegCustLoss, self).__init__(*args, **kwargs)
#         self.model_type = "Deep_Full_Feat_RegCustom_Loss"
#
#     def customLoss(y_true,y_pred):
#         return K.sum(K.log(yTrue) - K.log(yPred))
#
#     def model_func(self):
#         input_i = layers.Input(shape=[1])
#         i = layers.Embedding(self.n_items + 1, self.k_features)(input_i)
#         i = layers.Flatten()(i)
#         i = layers.normalization.BatchNormalization()(i)
#
#         input_u = layers.Input(shape=[1])
#         u = layers.Embedding(self.n_users + 1, self.k_features)(input_u)
#         u = layers.Flatten()(u)
#         u = layers.normalization.BatchNormalization()(u)
#
#         input_i_spectral = layers.Input(shape=[128])
#         im_spectral = layers.normalization.BatchNormalization()(input_i_spectral)
#         input_u_spectral = layers.Input(shape=[128])
#         um_spectral = layers.normalization.BatchNormalization()(input_u_spectral)
#
#         input_i_lle = layers.Input(shape=[128])
#         im_lle = layers.normalization.BatchNormalization()(input_i_lle)
#         input_u_lle = layers.Input(shape=[128])
#         um_lle = layers.normalization.BatchNormalization()(input_u_lle)
#
#         nn = layers.concatenate([i, u, im_spectral, um_spectral, im_lle, um_lle])
#
#         nn = layers.Dense(1024, activation='relu', activity_regularizer=regularizers.l2(0.001))(nn)
#         nn = layers.Dropout(0.5)(nn)
#         nn = layers.normalization.BatchNormalization()(nn)
#         nn = layers.Dense(512, activation='relu', activity_regularizer=regularizers.l2(0.001))(nn)
#         nn = layers.Dropout(0.5)(nn)
#         nn = layers.normalization.BatchNormalization()(nn)
#         nn = layers.Dense(256, activation='relu', activity_regularizer=regularizers.l2(0.001))(nn)
#         nn = layers.Dropout(0.5)(nn)
#         nn = layers.normalization.BatchNormalization()(nn)
#         nn = layers.Dense(128, activation='relu', activity_regularizer=regularizers.l2(0.001))(nn)
#
#         output = layers.Dense(5, activation='softmax')(nn)
#
#         model = models.Model([input_i, input_u, input_i_spectral, input_u_spectral, input_i_lle, input_u_lle], output)
#         model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])
#         return model

class DeepNetworkFeatReg(Network):
    """docstring for DeepNetworkFeat."""
    def __init__(self, *args, **kwargs):
        super(DeepNetworkFeatReg, self).__init__(*args, **kwargs)
        self.model_type = "Deep_Full_Feat_Reg"

    def model_func(self):
        input_i = layers.Input(shape=[1])
        i = layers.Embedding(self.n_items + 1, self.k_features, embeddings_regularizer=regularizers.l2(0.0001))(input_i)
        i = layers.Flatten()(i)
        i = layers.normalization.BatchNormalization()(i)

        input_u = layers.Input(shape=[1])
        u = layers.Embedding(self.n_users + 1, self.k_features, embeddings_regularizer=regularizers.l2(0.0001))(input_u)
        u = layers.Flatten()(u)
        u = layers.normalization.BatchNormalization()(u)

        input_i_spectral = layers.Input(shape=[64])
        im_spectral = layers.normalization.BatchNormalization()(input_i_spectral)
        input_u_spectral = layers.Input(shape=[64])
        um_spectral = layers.normalization.BatchNormalization()(input_u_spectral)

        input_i_lle = layers.Input(shape=[64])
        im_lle = layers.normalization.BatchNormalization()(input_i_lle)
        input_u_lle = layers.Input(shape=[64])
        um_lle = layers.normalization.BatchNormalization()(input_u_lle)

        input_i_fa = layers.Input(shape=[64])
        im_fa = layers.normalization.BatchNormalization()(input_i_fa)
        input_u_fa = layers.Input(shape=[64])
        um_fa = layers.normalization.BatchNormalization()(input_u_fa)

        input_i_nmf = layers.Input(shape=[64])
        im_nmf = layers.normalization.BatchNormalization()(input_i_nmf)
        input_u_nmf = layers.Input(shape=[64])
        um_nmf = layers.normalization.BatchNormalization()(input_u_nmf)

        nn = layers.concatenate([i, u,
                                 im_spectral, um_spectral,
                                 im_lle, um_lle,
                                 im_fa, um_fa,
                                 im_nmf, um_nmf])

        nn = layers.Dense(1024, activation='relu')(nn)
        nn = layers.Dropout(0.5)(nn)
        nn = layers.normalization.BatchNormalization()(nn)
        nn = layers.Dense(512, activation='relu')(nn)
        nn = layers.Dropout(0.5)(nn)
        nn = layers.normalization.BatchNormalization()(nn)
        nn = layers.Dense(256, activation='relu')(nn)
        nn = layers.Dropout(0.5)(nn)
        nn = layers.normalization.BatchNormalization()(nn)
        nn = layers.Dense(128, activation='relu')(nn)

        output = layers.Dense(5, activation='softmax')(nn)

        model = models.Model([input_i, input_u,
                              input_i_spectral, input_u_spectral,
                              input_i_lle, input_u_lle,
                              input_i_fa, input_u_fa,
                              input_i_nmf, input_u_nmf], output)

        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])
        return model

class DenseNetwork(Network):
    """docstring for ShallowNetwork."""
    def __init__(self, *args, **kwargs):
        super(DenseNetwork, self).__init__(*args, **kwargs)
        self.model_type = "Simple_Dense"

    def model_func(self):
        input_i = layers.Input(shape=[1])
        i = layers.Embedding(self.n_items + 1, self.k_features, embeddings_regularizer=regularizers.l2(0.0001))(input_i)
        i = layers.Reshape((self.k_features,))(i)
        i = layers.Dropout(0.5)(i)
        # i = layers.normalization.BatchNormalization()(i)

        input_u = layers.Input(shape=[1])
        u = layers.Embedding(self.n_users + 1, self.k_features, embeddings_regularizer=regularizers.l2(0.0001))(input_u)
        u = layers.Reshape((self.k_features,))(u)
        u = layers.Dropout(0.5)(u)
        # u = layers.normalization.BatchNormalization()(u)

        nn = layers.concatenate([i, u])

        nn = layers.Dense(256, activation='relu')(nn)
        nn = layers.Dropout(0.5)(nn)
        nn = layers.normalization.BatchNormalization()(nn)

        nn = layers.Dense(256, activation='relu')(nn)
        nn = layers.Dropout(0.5)(nn)
        nn = layers.normalization.BatchNormalization()(nn)

        nn = layers.Dense(256, activation='relu')(nn)

        output =  layers.Dense(self.n_classes, activation='softmax')(nn)

        model = models.Model([input_i, input_u], output)

        model.compile(optimizer=self.optimizer, loss=self.loss)
        return model
