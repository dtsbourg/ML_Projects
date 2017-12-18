"""
CS-433 : Machine Learning
Project 2 -- Recommender Systems

Team :
* Dylan Bourgeois
* Antoine Mougeot
* Philippe Verbist

---

pipeline.py : Defines some common pipeline used in this project.

A pipeline defines the set of operations to prepare, train, and run
a model. It takes as arguments which of these sequences should be run.

Available pipelines :
* dense_net_pipeline
* shallow_net_pipeline
* deep_net_pipeline
* deep_regularized_feat_net_pipeline
* deep_feat_net_pipeline
* embedding_pipeline
"""

from keras   import optimizers
from sklearn import model_selection
import pandas as pd
import numpy  as np
import datetime

import models
import run
import data
import utils

def dense_net_pipeline(train, predict, p=None):
    if train:
        train_x, train_y, test_x, test_y = data.load_full()
        n_users = train_x.User.max(); n_items = train_x.Item.max()

        m = models.DenseNetwork(n_items=n_items,
                                n_users=n_users,
                                n_train_samples=len(train_x),
                                n_test_samples=len(test_x),
                                optimizer=optimizers.Adam(lr=0.005),
                                k_features=64)

        p = training_pipeline(m, train_x, train_y, test_x, test_y)
    if predict:
        predict_pipeline(p)


def shallow_net_pipeline(train, predict, p=None):
    if train:
        train_x, train_y, test_x, test_y = data.load_full()
        n_users = train_x.User.max(); n_items = train_x.Item.max()

        m = models.ShallowNetwork(n_items=n_items,
                                  n_users=n_users,
                                  n_train_samples=len(train_x),
                                  n_test_samples=len(test_x),
                                  optimizer='adam')
        p = training_pipeline(m, train_x, train_y, test_x, test_y)
    if predict:
        predict_pipeline(p)

def deep_net_pipeline(train, predict, p=None):
    if train:
        train_x, train_y, test_x, test_y = data.load_full()
        n_users = train_x.User.max(); n_items = train_x.Item.max()

        m = models.DeepNetwork( n_items=n_items,
                                n_users=n_users,
                                n_train_samples=len(train_x),
                                n_test_samples=len(test_x),
                                optimizer=optimizers.Adam(lr=0.001),
                                k_features=128)

        p = training_pipeline(m, train_x, train_y, test_x, test_y)
    if predict:
        predict_pipeline(p)

def training_pipeline(model, train_x, train_y, test_x, test_y):
    batch_size = 2048
    s = model

    history = run.fit(model=s,
                      train_x=train_x,
                      train_y=train_y,
                      test_x=test_x,
                      test_y=test_y,
                      embedding=True,
                      epochs=100,
                      batch_size=batch_size)

    utils.plot(s.description_str(), history)
    return utils.save_model(s)

def predict_pipeline(path):
    batch_size = 2048
    m = utils.load_model(path)

    sub      = data.load_submission()
    sub_data = utils.load_full_embedding([sub.Item, sub.User])

    pred = m.predict(sub_data, batch_size=batch_size)

    sub['Prediction'] = run.predict(pred)

    model_uid = datetime.datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
    save_path = '../res/pred/submission_'+model_uid+'.csv'
    sub.to_csv(save_path, columns=['Id', 'Prediction'], index=False)
    return save_path

def embedding_pipeline(suffix=''):
    x, y, _, _ = data.load_full(categorical=False, test_split=0.0)
    ##################################
    # Interaction Matrix
    ##################################
    # path = utils.build_interaction_matrix(users=np.asarray(x.User),
    #                                       items=np.asarray(x.Item),
    #                                       ratings=np.asarray(y),
    #                                       suffix=suffix)
    path = '../data/embeddings/interaction_matrix_full'
    ##################################
    # t-SNE Embedding (deprecated)
    ##################################
    # utils.build_tSNE_embedding(path+'.npy', suffix=suffix)

    ##################################
    # Spectral Embedding
    ##################################
    # utils.build_spectral_embedding(path+'.npy', suffix=suffix)

    ##################################
    # Locally Linear Embedding (LLE)
    ##################################
    # utils.build_lle_embedding(path+'.npy', suffix=suffix)

    ##################################
    # Non-Negative MF (NMF)
    ##################################
    # utils.build_nmf_embedding(path+'.npy', suffix=suffix)

    ##################################
    # Factor Analysis (FA)
    ##################################
    # utils.build_fa_embedding(path+'.npy', suffix=suffix)
