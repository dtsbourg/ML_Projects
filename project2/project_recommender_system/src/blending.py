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
    myList = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
    x0 = [x / 9.0 for x in myList]
    
    res = sc.optimize.minimize(blending_fun, x0, method='SLSQP', options={'disp': True})
    print(res)


baselines.run()
sl.runAll()
test_baseline  = pd.read_pickle('svg/baseline_test.p')
test_sl_svd    = pd.read_pickle('svg/surprise_SVD_test.p')
test_sl_svdpp  = pd.read_pickle('svg/surprise_SVDpp_test.p')
test_sl_nmf    = pd.read_pickle('svg/surprise_NMF_test.p')

blending()
