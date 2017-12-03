from sklearn import model_selection
import pandas as pd
import numpy as np

import models
import run
import data
import utils

train = False
predict = True

if __name__ == '__main__':
    if train is True:
        train_x, train_y, test_x, test_y = data.load_subset()
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
        pred = m.predict([sub.Item, sub.User], batch_size=1024)

        sub['Prediction'] = run.predict(pred)
        sub.to_csv('../res/pred/submission_test_weighted.csv', columns=['Id', 'Prediction'], index=False)
