from sklearn import model_selection
import pandas as pd
import numpy as np

def load_data(path='../data/data_train.csv', categorical=True, test_size=0.05, train_size=0.1):
    ratings = pd.read_csv(path, dtype={'Prediction': np.int})
    pos = ratings.Id.str.extract('r([0-9]+)_c([0-9]+)', expand=True)
    ratings['User'] = pos[0].astype(np.int)
    ratings['Item'] = pos[1].astype(np.int)

    train_x, test_x = model_selection.train_test_split(ratings, test_size=0.05, train_size=0.1)

    if categorical is True:
        categorical_train_y = np.zeros([train_x.shape[0], 5])
        categorical_train_y[np.arange(train_x.shape[0]), train_x.Prediction - 1] = 1

        categorical_test_y = np.zeros([test_x.shape[0], 5])
        categorical_test_y[np.arange(test_x.shape[0]), test_x.Prediction - 1] = 1
        return train_x, categorical_train_y, test_x, categorical_test_y
    else:
        return train_x, train_x.Ratings, test_x, test_x.Ratings

def load_subset(categorical=True):
    return load_data(categorical=categorical)

def load_full(categorical=True, test_split=0.2):
    return load_data(categorical=categorical, test_size=test_split, train_size=1.)
