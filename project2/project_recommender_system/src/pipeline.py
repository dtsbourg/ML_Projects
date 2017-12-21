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
* embedding_pipeline
"""

from keras   import optimizers
from sklearn import model_selection
import pandas as pd
import numpy  as np
import datetime

import models
import pipe
import data
import utils

def dense_net_pipeline(train, predict, p=None):
    """
    Defines the pipeline for our dense network architecture.

    Available modes:
        train   : runs the training procedure on the selected model
        predict : runs the prediction with a given model. If no model is \
                  specified it will run with the model generated by train.

    Args:
        train (bool): Is training activated
        predict (bool): Is prediction activated
        p (str): path to model if only predict is activated
    """
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
    """
    Defines the pipeline for our shallow network architecture.

    Available modes:
        train   : runs the training procedure on the selected model
        predict : runs the prediction with a given model. If no model is \
                  specified it will run with the model generated by train.

    Args:
        train (bool): Is training activated
        predict (bool): Is prediction activated
        p (str): path to model if only predict is activated
    """
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
    """
    Defines the pipeline for our deep network architecture.
    This is the network with the best results.

    Available modes:
        train   : runs the training procedure on the selected model
        predict : runs the prediction with a given model. If no model is \
                  specified it will run with the model generated by train.

    Args:
        train (bool): Is training activated
        predict (bool): Is prediction activated
        p (str): path to model if only predict is activated
    """
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
        print("Predicting with model ", p)
        predict_pipeline(p)

def training_pipeline(model, train_x, train_y, test_x, test_y):
    """
    Defines an abstract training pipeline.

    Args:
        model (bool): Is training activated
        train_x, train_y (list): Training set
        test_x , test_y  (list): Test set
    """
    batch_size = 2048
    s = model

    history = pipe.fit(model=s,
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
    """
    Defines an abstract prediction pipeline.

    Args:
        path (str): THe path to the model to predict with.
    """
    batch_size = 2048
    m = utils.load_model(path)

    sub      = data.load_submission()
    sub_data = utils.load_full_embedding([sub.Item, sub.User])

    pred = m.predict(sub_data, batch_size=batch_size)

    sub['Prediction'] = pipe.predict(pred)

    model_uid = datetime.datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
    save_path = '../res/pred/submission_'+model_uid+'.csv'
    sub.to_csv(save_path, columns=['Id', 'Prediction'], index=False)
    return save_path

def embedding_pipeline(suffix=''):
    """
    Defines an embedding pipeline, saving the computed
    values to a `.npy` file.

    Args:
        suffix (str): utility to add a uid to the embedding file.

    Available embeddings:
        * Interaction Matrix
        * t-SNE
        * Spectral Embedding
        * Locally Linear Embedding
        * Non-negative Matrix Factorisation
        * Factor Analysis
    """
    print("Starting embedding pipeline ...")
    x, y, _, _ = data.load_full(categorical=False, test_split=0.0)
    ##################################
    # Interaction Matrix
    ##################################
    # path = utils.build_interaction_matrix(users=np.asarray(x.User),
    #                                       items=np.asarray(x.Item),
    #                                       ratings=np.asarray(y),
    #                                       suffix=suffix)
    path = '../data/embeddings/interaction_matrix_full'
    print("Loaded interaction matrix ...")
    ##################################
    # t-SNE Embedding (deprecated)
    ##################################
    # utils.build_tSNE_embedding(path+'.npy', suffix=suffix)
    # print("Built t-SNE embeddings ...")

    ##################################
    # Spectral Embedding
    ##################################
    utils.build_spectral_embedding(path+'.npy', suffix=suffix)
    print("Built Spectral Embeddings ...")

    ##################################
    # Locally Linear Embedding (LLE)
    ##################################
    utils.build_lle_embedding(path+'.npy', suffix=suffix)
    print("Built LLE Embeddings ...")

    ##################################
    # Non-Negative MF (NMF)
    ##################################
    utils.build_nmf_embedding(path+'.npy', suffix=suffix)
    print("Built NMF Embeddings ...")

    ##################################
    # Factor Analysis (FA)
    ##################################
    utils.build_fa_embedding(path+'.npy', suffix=suffix)
    print("Built FA Embeddings ...")
