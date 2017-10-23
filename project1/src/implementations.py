import numpy as np
from auxiliary import *

# Contains all methods asked in step 2

def least_squares(y, tx):
    """
    Linear regresison using normal equations
    Returns optimal weights and associated minimum loss
    """
    a = tx.T.dot(tx)
    b = tx.T.dot(y)

    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    return w, loss

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent
    Returns optimal weights and associated minimum loss
    """
    w_start = initial_w
    w = w_start

    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y,tx,w)
        w = w - gamma * gradient

    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using stochastic gradient descent
    Returns optimal weights and associated minimum loss
    """
    w = initial_w
    loss = compute_loss(y, tx, w)

    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, max_iters):
        loss = compute_loss(minibatch_y, minibatch_tx, w)
        gradients = compute_stoch_gradient(minibatch_y, minibatch_tx, w)
        w = w - [gamma * g for g in gradients]

    return w, loss

def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations
    Returns optimal weights and associated minimum loss
    """
    a = tx.T.dot(tx) + (2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1]))
    b = tx.T.dot(y)

    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)

    return w, loss

def logistic_regression(y, tx, initial_w,max_iters, gamma) :
    """
    Logistic regression using gradient descent
    Returns optimal weights and associated minimum loss
    """
    w_start = initial_w
    w = w_start

    for n_iter in range(max_iters):
        loss = calculate_loss(y, tx, w)
        gradient = calculate_gradient(y, tx, w)
        w = w - gamma * gradient

    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent
    Returns optimal weights and associated minimum loss
    """
    verbose = False
    threshold = 1e-8
    w_start = initial_w
    w = w_start
    loss_old = 0.0
    
    for n_iter in range(max_iters):
        #print('compute loss')
        loss = compute_logistic_loss(y, tx, w)
        #print('compute gradient')
        gradient = compute_logistic_gradient(y, tx, w)
        #print('compute regularizers')
        loss_reg, gradient_reg = regularizer(lambda_, w)
        loss = loss + loss_reg
        
        if(abs(loss_old/loss) < 1.0+threshold # stop automatically when loss does not change significantly anymore
           and abs(loss_old/loss) > 1.0-threshold
           and n_iter !=0):
            break
        loss_old = loss
        
        gradient = gradient + gradient_reg
        w = w - gamma * gradient
        if verbose:
            print("Current iteration={i}, loss={l}".format(i=n_iter, l=loss))
        if n_iter == max_iters-1:
            print('\t reg_logistic_regression: stop due to max_iters')

    return w, loss

def logistic_regression_SGD(y, tx, initial_w,max_iters, gamma) :
    """
    Logistic regression using stochastic gradient descent
    Returns optimal weights and associated minimum loss
    """
    w_start = initial_w
    w = w_start

    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, max_iters):
        loss = compute_loss(minibatch_y, minibatch_tx, w)
        gradients = compute_gradient(minibatch_y, minibatch_tx, w)

        w = w - [gamma * g for g in gradients]

    return w, loss

def reg_logistic_regression_SGD(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using stochastic gradient descent
    Returns optimal weights and associated minimum loss
    """
    w_start = initial_w
    w = w_start

    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, max_iters):
        loss = compute_loss(minibatch_y, minibatch_tx, w)
        gradients = compute_gradient(minibatch_y, minibatch_tx, w)
        loss_reg, gradient_reg = regularizer(lambda_, w)

        loss = loss + loss_reg
        gradient = gradient + gradient_reg

        w = w - [gamma * g for g in gradients]

    return w, loss
