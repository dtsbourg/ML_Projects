import numpy as np
from auxiliary import *

# Contains all methods asked in step 2

def least_squares(y, tx):
    # Linear regresison using normal equations
    # Returns optimal weights and associated minimum loss
    a = tx.T.dot(tx)
    b = tx.T.dot(y)

    w = np.linalg.solve(a, b)
    loss = compute_loss(t, tx, w)
    return w, loss

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    # Linear regression using gradient descent
    # Returns optimal weights and associated minimum loss
    w_start = initial_w
    w = w_start

	for n_iter in range(max_iters):
		gradient = compute_gradient(y, tx, w)
		loss = compute_loss(y,tx,w)
		w = w - gamma * gradient

	return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    # Linear regression using stochastic gradient descent
    # Returns optimal weights and associated minimum loss
    w = initial_w
    loss = compute_loss(y, tx, w)
    
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, max_iters):
        loss = compute_loss(minibatch_y, minibatch_tx, w)
        gradients = compute_stoch_gradient(minibatch_y, minibatch_tx, w)
        w = w - [gamma * g for g in gradients ]

    return w, loss

def ridge_regression(y, tx, lambda_):
    # Ridge regression using normal equations
    # Returns optimal weights and associated minimum loss
    a = tx.T.dot(tx) + (2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1]))
    b = tx.T.dot(y)

    w = np.linalg.solve(a, b)
	loss = compute_loss(y, tx, w)

	return w, loss

def logistic_regression(y, tx, initial_w,max_iters, gamma) :
    return NotImplemented

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    return NotImplemented
