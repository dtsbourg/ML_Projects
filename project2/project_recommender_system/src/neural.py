from sklearn import model_selection
import pandas as pd
import numpy as np

import models
import run
import data
import utils

train   = True
predict = False
setup   = False

if __name__ == '__main__':
    deep_net_pipeline(train, predict, setup)

def deep_net_pipeline(train, predict, setup):
    if train is True:
        train_x, train_y, test_x, test_y = data.load_full()
        n_users = train_x.User.max()
        n_items = train_x.Item.max()

        s = models.DeepNetwork(n_items=n_items, n_users=n_users)
        history = run.fit(model=s.model,
                          train_x=train_x,
                          train_y=train_y,
                          test_x=test_x,
                          test_y=test_y,
                          epochs=10,
                          batch_size=1024)

        s.descr = s.description_str(suffix= str(max(history.epoch)+1) + "_epochs_", uid=True)

        run.plot(s.descr, history)
        utils.save_model(s)

    if predict is True:
        model_str = 'Simple_Deep_10_train_10_test_64_features_sgd_categorical_crossentropy_categorical_10_epochs_.h5'
        path = '../res/model/'+model_str
        m = utils.load_model(path)

        sub = data.load_submission()
        u_emb_sub = np.load('../data/embeddings/users_t_sne.npy')
        sub_emb_u = u_emb_sub[sub.User - 1]

        i_emb_sub = np.load('../data/embeddings/items_t_sne.npy')
        sub_emb_i = i_emb_sub[sub.Item - 1]

        pred = m.predict([sub.Items, sub.User, sub_emb_i, sub_emb_u], batch_size=1024)

        sub['Prediction'] = run.predict(pred)
        sub.to_csv('../res/pred/submission_test_weighted_deep_full.csv', columns=['Id', 'Prediction'], index=False)

def deep_feat_net_pipeline(train, predict, setup):
    if train is True:
        train_x, train_y, test_x, test_y = data.load_subset()
        n_users = train_x.User.max()
        n_items = train_x.Item.max()

        s = models.DeepNetworkFeat(n_items=n_items, n_users=n_users, )
        history = run.fit(model=s.model,
                          train_x=train_x,
                          train_y=train_y,
                          test_x=test_x,
                          test_y=test_y,
                          user_embedding=True,
                          item_embedding=True,
                          epochs=10,
                          batch_size=1024)

        s.descr = s.description_str(suffix= str(max(history.epoch)+1) + "_epochs_", uid=True)

        run.plot(s.descr, history)
        utils.save_model(s)

    if predict is True:
        model_str = 'Simple_Deep_10_train_10_test_64_features_sgd_categorical_crossentropy_categorical_10_epochs_.h5'
        path = '../res/model/'+model_str
        m = utils.load_model(path)

        sub = data.load_submission()
        u_emb_sub = np.load('../data/embeddings/users_t_sne.npy')
        sub_emb_u = u_emb_sub[sub.User - 1]

        i_emb_sub = np.load('../data/embeddings/items_t_sne.npy')
        sub_emb_i = i_emb_sub[sub.Item - 1]

        pred = m.predict([sub.Items, sub.User, sub_emb_i, sub_emb_u], batch_size=1024)

        sub['Prediction'] = run.predict(pred)
        sub.to_csv('../res/pred/submission_test_weighted_deep_feat.csv', columns=['Id', 'Prediction'], index=False)

    if setup is True:
        train_x, train_y, test_x, test_y = data.load_subset(categorical=False)
        # TODO : Run a single embedding for the dataset

        # utils.build_interaction_matrix(users=np.asarray(train_x.User),
        #                                items=np.asarray(train_x.Item),
        #                                ratings=np.asarray(train_y))
        # utils.build_tSNE_embedding('../data/embeddings/interaction_matrix.npy')

        # utils.build_interaction_matrix(users=np.asarray(test_x.User),
        #                                items=np.asarray(test_x.Item),
        #                                ratings=np.asarray(test_y),
        #                                suffix='_test')
        # utils.build_tSNE_embedding('../data/embeddings/interaction_matrix_test.npy', suffix='_test')
