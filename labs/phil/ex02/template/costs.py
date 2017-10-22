# -*- coding: utf-8 -*-
"""Function used to compute the loss."""

def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    MAE = True
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
