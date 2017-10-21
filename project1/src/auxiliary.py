import numpy as np

# Contains all auxiliary functions used in ML_Methods

def compute_mse(e):
    return 1/2 * np.mean(e**2)

def compute_loss(y, tx, w):
    e = y - tx.dot(w)
    return calculate_mse(e)

def compute_gradient(y, tx, w):
    e = y - tx.dot(w)
    N = y.shape[0]

    grad = 1/N * np.dot(tx.T, e)
    return grad
