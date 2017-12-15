import pandas as pd
import numpy as np
import os
import sys
import pickle
import time
import scipy as sc
import data
import baselines

from sklearn.metrics import mean_squared_error
import math
import statistics as stat


def blending_fun(x):
    Baselines = ['Global_Mean', 'User_Mean', 'Movie_Mean', 'Movie_Mean_Corrected', 
                 'Global_Median', 'User_Median', 'Movie_Median', 'Movie_Median_Corrected', 'Movie_RegLin']

    res = 0
    for i in range(len(Baselines)):
        res += x[i] * test_baseline[Baselines[i]]
        
    rmse = math.sqrt(mean_squared_error(res, train['Rating']))
    return rmse


def blending():
    myList = [1., 1., 1., 1., 1., 1., 1., 1., 1.]
    x0 = [x / 9.0 for x in myList]
    
    res = sc.optimize.minimize(blending_fun, x0, method='SLSQP', options={'disp': True})
    print(res)


baselines()
test_baseline  = pd.read_pickle('svg/baseline_test.p')

blending()
