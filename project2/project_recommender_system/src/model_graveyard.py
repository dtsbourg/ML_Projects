"""
CS-433 : Machine Learning
Project 2 -- Recommender Systems

Team :
* Dylan Bourgeois
* Antoine Mougeot
* Philippe Verbist

---

model_graveyard.py : A collection of experimental, partial, unused or deprecated models.

This module is meant to hold the models that didn't make the cut.
Enter at your own risk.
"""

import warnings

import keras
from keras import layers
from keras import models
from keras import optimizers
from keras import regularizers
import keras.backend as K
import datetime
import numpy as np

class DeepNetwork1(Network):
    """docstring for DeepNetwork."""
    def __init__(self, *args, **kwargs):
        warnings.warn("This model is deprecated. Are you looking for `DeepNetwork` ?", DeprecationWarning)
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

class ShallowNetwork2(Network):
    """docstring for ShallowNetwork2."""
    def __init__(self, *args, **kwargs):
        warnings.warn("This model is deprecated. Are you looking for `ShallowNetwork` ?", DeprecationWarning)
        super(ShallowNetwork, self).__init__(*args, **kwargs)
        self.model_type = "Shallow2"

    def model_func(self):
        lambda_ = 1e-4
        input_i = layers.Input(shape=[1])
        i = layers.Embedding(self.n_items+1, self.k_features, embeddings_regularizer=l2(lambda_))(input_i)
        input_u = layers.Input(shape=[1])
        u = layers.Embedding(self.n_users+1, self.k_features, embeddings_regularizer=l2(lambda_))(input_u)

        nn = layers.concatenate([i, u])

        nn = Flatten()(nn)
        nn = Dropout(0.25)(nn)
        nn = Dense(64, activation='relu')(nn)
        nn = Dropout(0.75)(nn)

        output = Dense(1)(nn)

        model = models.Model([input_i, input_u], output)
        model.compile(Adam(0.001), loss='mse')

        return model

class DeepNetworkFeat(Network):
    """docstring for DeepNetworkFeat."""
    def __init__(self, *args, **kwargs):
        warnings.warn("This model is deprecated. Are you looking for `DeepNetwork` ?", DeprecationWarning)
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

###########################################
# An attempt to plug into keras backend
# for a custom loss function
##########################################
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
