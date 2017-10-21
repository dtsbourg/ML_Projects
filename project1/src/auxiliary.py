import numpy as np

# Contains all auxiliary functions used in implementations.py

def compute_mse(e):
    return 1/2 * np.mean(e**2)

def compute_loss(y, tx, w):
    e = y - tx.dot(w)
    return compute_mse(e)

def compute_gradient(y, tx, w):
    e = y - tx.dot(w)
    N = y.shape[0]

    grad = 1/N * np.dot(tx.T, e)
    return grad

def sigmoid(t):
    return 1 / (1 + np.exp(-t))

def compute_logistic_loss(y, tx, w):
    prediction = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(prediction)) + (1-y).T.dot(np.log(1 - prediction))

    return - loss[0][0]

def compute_logistic_gradient(y,tx, w):
    prediction = sigmoid(tx.dot(w))
    gradient = tx.T.dot(prediction - y)

    return gradient

def regularizer(lambda_, w):
    loss_reg = lambda_ * w.T.dot(w)[0][0]
	gradient_reg = 2 * lambda_ * w

	return loss_reg, gradient_reg
