# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    y_copy = np.copy(y)
    x_copy = np.copy(x)
    
    x_train = np.array([])
    y_train = np.array([])
    
    N = y.shape[0]
    N_sel = 0.0
    
    while(N_sel/N < ratio):
        #print(N_sel/N)
        idx = np.random.randint(0, N-N_sel, 1)
        x_train = np.append(x_train, x_copy[idx])
        y_train = np.append(y_train, y_copy[idx])
        
        x_copy = np.delete(x_copy, idx,axis=0)
        y_copy = np.delete(y_copy, idx,axis=0)
        N_sel +=1.0
    
    x_test = x_copy
    y_test = y_copy
    
    return x_train, x_test, y_train, y_test
