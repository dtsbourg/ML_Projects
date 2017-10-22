# -*- coding: utf-8 -*-
"""Function used to compute the loss."""

import numpy as np

def compute_loss(y, tx, w, MAE=False):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    L = 0
    e = y - tx.dot(w)
    
    if(MAE):
        L = np.sum(np.abs(e),axis=0)/len(y)
    else:
        L = np.sum(np.power(e,2),axis=0)/(2.0*len(y))
    return L
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute loss by MSE / MAE
    # ***************************************************

    
def compute_mse(y, tx, w):
    """Calculate the loss."""
    e = y - tx.dot(w)
    L = np.sum(np.power(e,2),axis=0)/(2.0*len(y))
    return L