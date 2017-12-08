from sklearn import model_selection
import pandas as pd
import numpy as np
import datetime

import models
import run
import data
import utils

def deep_net_pipeline(train, predict, setup):
    if train is True:
        train_x, train_y, test_x, test_y = data.load_subset()
        n_users = train_x.User.max()
        n_items = train_x.Item.max()

        s = models.DeepNetwork(n_items=n_items,
                               n_users=n_users,
                               n_train_samples=len(train_x),
                               n_test_samples=len(test_x),
                               optimizer='adam')
        history = run.fit(model=s,
                          train_x=train_x,
                          train_y=train_y,
                          test_x=test_x,
                          test_y=test_y,
                          epochs=20,
                          batch_size=256)
        run.plot(s.description_str(), history)
        utils.save_model(s)

    if predict is True:
        model_str = 'Simple_Deep_117695_train_58848_test_64_features_adam_categorical_crossentropy_categorical_11_epochs_.h5'
        path = '../res/model/'+model_str
        m = utils.load_model(path)

        sub = data.load_submission()

        pred = m.predict([sub.Item, sub.User], batch_size=256)

        sub['Prediction'] = run.predict(pred)
        model_uid = datetime.datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
        sub.to_csv('../res/pred/submission_test_weighted_deep_full'+model_uid+'.csv', columns=['Id', 'Prediction'], index=False)

def deep_regularized_feat_net_pipeline(train, predict, setup):
    batch_size = 2048
    if train is True:
        #train_x, train_y, test_x, test_y = data.load_subset()
        train_x, train_y, test_x, test_y = data.load_full(test_split=0.5)
        n_users = train_x.User.max()
        n_items = train_x.Item.max()

        s = models.DeepNetworkFeatReg( n_items=n_items,
                                       n_users=n_users,
                                       n_train_samples=len(train_x),
                                       n_test_samples=len(test_x),
                                       optimizer='adam',
                                       k_features=128)

        history = run.fit(model=s,
                          train_x=train_x,
                          train_y=train_y,
                          test_x=test_x,
                          test_y=test_y,
                          embedding=True,
                          epochs=20,
                          batch_size=batch_size)

        run.plot(s.description_str(), history)
        utils.save_model(s)

    if predict is True:
        model_str = 'Deep_Full_Feat_Reg_588476_train_588476_test_128_features_adam_categorical_crossentropy_categorical_12_epochs_.h5'
        path = '../res/model/'+model_str
        m = utils.load_model(path)

        sub = data.load_submission()
        u_spectral_sub = np.load('../data/embeddings/users_spectral_full.npy')
        sub_spectral_u = u_spectral_sub[sub.User - 1]
        u_lle_sub = np.load('../data/embeddings/users_lle_full.npy')
        sub_lle_u = u_lle_sub[sub.User - 1]

        i_spectral_sub = np.load('../data/embeddings/items_spectral_full.npy')
        sub_spectral_i = i_spectral_sub[sub.Item - 1]
        i_lle_sub = np.load('../data/embeddings/items_lle_full.npy')
        sub_lle_i = i_lle_sub[sub.Item - 1]

        pred = m.predict([sub.Item, sub.User, sub_spectral_i, sub_spectral_u, sub_lle_i, sub_lle_u], batch_size=batch_size)

        sub['Prediction'] = run.predict(pred)
        model_uid = datetime.datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
        sub.to_csv('../res/pred/submission_test_weighted_deep_full_feat_full_reg_'+model_uid+'.csv', columns=['Id', 'Prediction'], index=False)

    if setup is True:
        train_x, train_y, test_x, test_y = data.load_full(categorical=False, test_split=0.0)
        # # Training data embedding
        # embedding_pipeline(train_x, train_y)
        # # Test data embedding
        # embedding_pipeline(test_x, test_y, suffix='_test')
        full_x, full_y, _, _ = data.load_full(categorical=False, test_split=0.0)
        embedding_pipeline(full_x, full_y, suffix='_full')

def deep_feat_net_pipeline(train, predict, setup):
    if train is True:
        # train_x, train_y, test_x, test_y = data.load_subset()
        train_x, train_y, test_x, test_y = data.load_full(test_split=0.4)
        n_users = train_x.User.max()
        n_items = train_x.Item.max()

        s = models.DeepNetworkFeat(n_items=n_items,
                                   n_users=n_users,
                                   n_train_samples=len(train_x),
                                   n_test_samples=len(test_x),
                                   optimizer='adam',
                                   k_features=128)

        history = run.fit(model=s,
                          train_x=train_x,
                          train_y=train_y,
                          test_x=test_x,
                          test_y=test_y,
                          embedding=True,
                          epochs=20,
                          batch_size=256)

        run.plot(s.description_str(), history)
        utils.save_model(s)

    if predict is True:
        model_str = 'Deep_Full_Feat_588476_train_588476_test_128_features_adam_categorical_crossentropy_categorical_'
        path = '../res/model/'+model_str
        m = utils.load_model(path)

        sub = data.load_submission()
        u_tsne_sub = np.load('../data/embeddings/users_t_sne_full.npy')
        sub_tsne_u = u_tsne_sub[sub.User - 1]
        u_spectral_sub = np.load('../data/embeddings/users_spectral_full.npy')
        sub_spectral_u = u_spectral_sub[sub.User - 1]
        u_lle_sub = np.load('../data/embeddings/users_lle_full.npy')
        sub_lle_u = u_lle_sub[sub.User - 1]

        i_tsne_sub = np.load('../data/embeddings/items_t_sne_full.npy')
        sub_tsne_i = i_tsne_sub[sub.Item - 1]
        i_spectral_sub = np.load('../data/embeddings/items_spectral_full.npy')
        sub_spectral_i = i_spectral_sub[sub.Item - 1]
        i_lle_sub = np.load('../data/embeddings/items_lle_full.npy')
        sub_lle_i = i_lle_sub[sub.Item - 1]

        pred = m.predict([sub.Item, sub.User, sub_tsne_i, sub_tsne_u, sub_spectral_i, sub_spectral_u, sub_lle_i, sub_lle_u], batch_size=256)

        sub['Prediction'] = run.predict(pred)
        model_uid = datetime.datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
        sub.to_csv('../res/pred/submission_test_weighted_deep_full_feat_full_'+model_uid+'.csv', columns=['Id', 'Prediction'], index=False)

    if setup is True:
        train_x, train_y, test_x, test_y = data.load_full(categorical=False, test_split=0.0)
        # # Training data embedding
        # embedding_pipeline(train_x, train_y)
        # # Test data embedding
        # embedding_pipeline(test_x, test_y, suffix='_test')
        full_x, full_y, _, _ = data.load_full(categorical=False, test_split=0.0)
        embedding_pipeline(full_x, full_y, suffix='_full')

def embedding_pipeline(x,y,suffix=''):
    ##################################
    # Interaction Matrix
    ##################################
    # path = utils.build_interaction_matrix(users=np.asarray(x.User),
    #                                       items=np.asarray(x.Item),
    #                                       ratings=np.asarray(y),
    #                                       suffix=suffix)
    path = '../data/embeddings/interaction_matrix_full'
    ##################################
    # t-SNE Embedding
    ##################################
    #utils.build_tSNE_embedding(path+'.npy', suffix=suffix)

    ##################################
    # Spectral Embedding
    ##################################
    #utils.build_spectral_embedding(path+'.npy', suffix=suffix)

    ##################################
    # Locally Linear Embedding (LLE)
    ##################################
    #utils.build_lle_embedding(path+'.npy', suffix=suffix)


if __name__ == '__main__':
    deep_regularized_feat_net_pipeline(train=False, predict=True, setup=False)
