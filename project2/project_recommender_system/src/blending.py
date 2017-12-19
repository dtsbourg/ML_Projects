"""
CS-433 : Machine Learning
Project 2 -- Recommender Systems

Team :
* Dylan Bourgeois
* Antoine Mougeot
* Philippe Verbist

---

baselines.py : interface for running the blending procedure.

"""

import pandas as pd
import numpy as np
import os
import sys
import pickle
import time
import scipy as sc
import data
import baselines
import surprise_lib as sl

from sklearn.metrics import mean_squared_error
import math
import statistics as stat


def blending_fun(x):
    """
    Function to be optimized. All outputs of the algorithms are given a weight (x),
    and are then summed up. The resulting estimated rating
    are then evaluted with RMSE.
    The lower value, the best.

    Args:
        * x (list): the weights to be given to each function. Must be of length 12
    
    """
    Baselines = ['Global_Mean', 'User_Mean', 'Movie_Mean', 'Movie_Mean_Corrected',
                 'Global_Median', 'User_Median', 'Movie_Median', 'Movie_Median_Corrected', 'Movie_RegLin']

    Surprise = ['SVD', 'SVDpp', 'NML']

    res = 0
    for i in range(len(Baselines)):
        res += x[i] * test_baseline[Baselines[i]]

    res += x[7] * test_sl_svd["Result"]
    res += x[8] * test_sl_svdpp["Result"]
    res += x[9] * test_sl_nmf["Result"]

    rmse = math.sqrt(mean_squared_error(res, test_baseline['Rating']))
    return rmse


def blending():
    """
    Runs the optimization on 'blending_fun', to find the best vector x.
    """
    myList = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
    x0 = [x / 9.0 for x in myList]

    res = sc.optimize.minimize(blending_fun, x0, method='SLSQP', options={'disp': True})
    print(res)

def run()
    """
    Runs the baselines algorithms (the result is stored in a pickle).
    Runs the Surprise algorithms (the result is stored in a pickle).
    Starts the blending.
    """
    baselines.run()
    sl.runAll()
    test_baseline  = pd.read_pickle(os.path.join('..','data','baselines','baseline_test.p'))
    test_sl_svd    = pd.read_pickle(os.path.join('..','data','baselines','surprise_SVD_test.p'))
    test_sl_svdpp  = pd.read_pickle(os.path.join('..','data','baselines','surprise_SVDPP_test.p'))
    test_sl_nmf    = pd.read_pickle(os.path.join('..','data','baselines','surprise_NMF_test.p'))

    blending()
