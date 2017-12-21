"""
CS-433 : Machine Learning
Project 2 -- Recommender Systems

Team :
* Dylan Bourgeois
* Antoine Mougeot
* Philippe Verbist

---

baselines.py : interface for running the baselines.

"""

import pandas as pd
import numpy as np
import os
import sys
import pickle
import time
import scipy as sc
import data

from sklearn.metrics import mean_squared_error
import math
import statistics as stat

def dict_mean_user(df):
    """ dictionary with key UserID and value User Mean """
    return dict(df.groupby('User').mean().Rating)

def dict_mean_movie(df):
    """ dictionary with key UserID and value Movie Mean """
    return dict(df.groupby('Item').mean().Rating)

def dict_median_user(df):
    """ dictionary with key UserID and value User Mean """
    return dict(df.groupby('User').median().Rating)

def dict_median_movie(df):
    """ dictionary with key UserID and value Movie Mean """
    return dict(df.groupby('Item').median().Rating)

def dict_nbrating_movie(df):
    """ dictionary with key UserID and value Movie Mean """
    return dict(df.groupby('Item').count().Rating)



def baselines():
    """
    Run baselines algorithms:
    * movie mean
    * global mean
    * global median
    * user median
    * movie median
    * movie regularized
    * movie mean corrected
    * movie median corrected

    Loads data from "data/data/data_train.csv".
    divides it in "train" (for learning) and "test" (for evaluating),
    and runs the aforementioned algorithms on it.
    The "test" data is stored in a pickle for later use. 
    """
    
    users_bs_path = os.path.join('..','data','baselines','users_baseline.p')
    movies_bs_path = os.path.join('..','data','baselines','movies_baseline.p')
    train_bs_path = os.path.join('..','data','baselines','train_baseline.p')
    test_bs_path = os.path.join('..','data','baselines','test_baseline.p')

    train, _, test, _ = data.load_data(categorical=False, test_size=0.1, train_size=0.9)
    train["Rating"] = train.Prediction
    test["Rating"]  = test.Prediction

    train.sort_index(inplace=True)
    test.sort_index(inplace=True)

    users = pd.DataFrame(index = range(10000), columns=['User', 'Mean'])
    movies = pd.DataFrame(index = range(1000), columns=['Item', 'Mean'])

    median = train['Rating'].median()
    mean = train['Rating'].mean()

    if False and os.path.isfile(users_bs_path) and os.path.isfile(movies_bs_path):
        users  = pd.read_pickle(users_bs_path)
        movies = pd.read_pickle(movies_bs_path)
    else:
        for i in range(0, 10000):
            users.at[i,'User'] = i+1

        for i in range(0, 1000):
            movies.at[i,'Movie'] = i+1

        mean_u = dict_mean_user(train)
        for i in range(0,10000):
             users.at[i,'Mean'] = list(mean_u.values())[i]

        mean_m = dict_mean_movie(train)
        for i in range(1000):
            movies.at[i,'Mean'] = list(mean_m.values())[i]

        median_u = dict_median_user(train)
        for i in range(0,10000):
            users.at[i,'Median'] = list(median_u.values())[i]

        median_m = dict_median_movie(train)
        for i in range(1000):
            movies.at[i,'Median'] = list(median_m.values())[i]

        nbrating_m = dict_nbrating_movie(train)
        for i in range(1000):
            movies.at[i,'Nb Ratings'] = list(nbrating_m.values())[i]
        m, b = np.polyfit(movies['Nb Ratings'],movies['Mean'], 1)

        for i in range(1000):
            movies.at[i,'Reg Lin Value'] =  m * movies['Nb Ratings'][i] + b

        users['Difference_To_Mean'] = users['Mean'].mean() - users['Mean']


    if os.path.isfile(train_bs_path):
        train = pd.read_pickle(train_bs_path)
    else:
        print("Generate Train estimations")
        for index, row in train.iterrows():
            train.at[index, 'User_Mean']  =  users.at[ int(train.at[index,'User'])-1,'Mean']
            train.at[index, 'Movie_Mean'] =  movies.at[int(train.at[index,'Item'])-1,'Mean']

            train.at[index, 'Global_Mean'] = mean
            train.at[index, 'Global_Median'] =median

            train.at[index, 'User_Median']  = users.at[ int(train.at[index,'User']) -1,'Median']
            train.at[index, 'Movie_Median'] = movies.at[int(train.at[index,'Item'])-1,'Median']

            train.at[index, 'Movie_RegLin'] = movies.at[int(train.at[index,'Item'])-1,'Reg Lin Value']

            train.at[index, 'Movie_Mean_Corrected']   = users.at[int(train.at[index,'Item'])-1, 'Mean']   + users.at[int(train.at[index,'User'])-1,'Difference_To_Mean']
            train.at[index, 'Movie_Median_Corrected'] = users.at[int(train.at[index,'Item'])-1, 'Median'] + users.at[int(train.at[index,'User'])-1,'Difference_To_Mean']

            if index % 10000 == 0:
                print("Train: {0:.2f}".format(index/len(train["User"])))

        train.to_pickle(train_bs_path)

    if os.path.isfile(test_bs_path):
        test = pd.read_pickle(test_bs_path)
    else:
        print("Generate Test estimations")
        counter = 0
        for index, row in test.iterrows():
            test.at[index, 'User_Mean']  = users.at[ int(test.at[index, 'User'])-1, 'Mean']
            test.at[index, 'Movie_Mean'] = movies.at[int(test.at[index, 'Item'])-1, 'Mean']

            test.at[index, 'Global_Mean'] = mean
            test.at[index, 'Global_Median'] = median

            test.at[index, 'User_Median']  = users.at[ int(test.at[index,'User'])-1, 'Median']
            test.at[index, 'Movie_Median'] = movies.at[int(test.at[index,'Item'])-1, 'Median']

            test.at[index, 'Movie_RegLin'] = movies.at[int(test.at[index, 'Item'])-1, 'Reg Lin Value']

            test.at[index, 'Movie_Mean_Corrected']  = users.at[int(test.at[index,'Item'])-1,'Mean']   + users.at[int(test.at[index,'User'])-1, 'Difference_To_Mean']
            test.at[index, 'Movie_Median_Corrected']= users.at[int(test.at[index,'Item'])-1,'Median'] + users.at[int(test.at[index,'User'])-1, 'Difference_To_Mean']

            if counter % 10000 == 0:
                print("Test: {0:.2f}".format(index/len(train["User"])))
            counter += 1
        test.to_pickle(test_bs_path)


    baselines = ['Global_Mean', 'User_Mean', 'Movie_Mean', 'Movie_Mean_Corrected',
                 'Global_Median', 'User_Median', 'Movie_Median', 'Movie_Median_Corrected', 'Movie_RegLin']



    for i in range(len(baselines)):
        print('RMSE Train for {} is : {:.4f}'.format(baselines[i],
                                           math.sqrt(mean_squared_error(train[baselines[i]],
                                                                        train['Rating']))))

        print('RMSE Test  for {} is : {:.4f}'.format(baselines[i],
                                           math.sqrt(mean_squared_error(test[baselines[i]],
                                                                        test['Rating']))))



def run():
    baselines()
