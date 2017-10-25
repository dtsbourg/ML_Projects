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

def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    """
    Linear regression using stochastic gradient descent
    Returns optimal weights and associated minimum loss
    """
    w = initial_w
    loss = compute_loss(y, tx, w)

    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, max_iters):
        loss = compute_loss(y, tx, w)
        gradients = compute_gradient(minibatch_y, minibatch_tx, w)
        w = w - gamma * gradients

    return w, loss

def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations
    Returns optimal weights and associated minimum loss
    """
    lambda_prime = lambda_ / (2*tx.shape[0])
    a = np.linalg.inv(tx.transpose().dot(tx) + lambda_*np.eye(tx.shape[1]))
    
    b = tx.transpose().dot(y)
    w_star = a.dot(b)
    
    e = y - tx.dot(w_star)
    
    mse = np.asarray((e**2).mean())
    return w_star, mse

def logistic_regression(y, tx, initial_w, max_iters, gamma) :
    """
    Logistic regression using gradient descent
    Returns optimal weights and associated minimum loss
    """
    w_start = initial_w
    w = w_start
    loss_old = 0.0

    for n_iter in range(max_iters):
        loss = compute_logistic_loss(y, tx, w)
        gradient = compute_logistic_gradient(y, tx, w)
        w = w - gamma * gradient
        
        if check_stop(loss, loss_old):
            print('break!')
            w = w_new
            break;
        loss_old = loss
        
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent
    Returns optimal weights and associated minimum loss
    """
    w = initial_w
    loss_old = 0.0
    
    for n_iter in range(max_iters):
        loss = compute_logistic_loss(y, tx, w)
        gradient = compute_logistic_gradient(y, tx, w)
        loss_reg, gradient_reg = regularizer(lambda_, w)
        loss_new = loss + loss_reg
        gradient = gradient + gradient_reg
        w = w - gamma * gradient
            
        if check_stop(loss, loss_old):
            print('break!')
            break;
        loss_old = loss
    return w, loss

def logistic_regression_SGD(y, tx, initial_w,max_iters, gamma) :
    """
    Logistic regression using stochastic gradient descent
    Returns optimal weights and associated minimum loss
    """
    w_start = initial_w
    w = w_start
    loss_old = 0.0
    
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, max_iters):
        loss = compute_loss(y, tx, w)
        gradient = compute_gradient(minibatch_y, minibatch_tx, w)
        w = w - gamma * gradient
            
        if check_stop(loss, loss_old):
            print('break!')
            break;
        loss_old = loss
        
    return w, loss

def reg_logistic_regression_SGD(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using stochastic gradient descent
    Returns optimal weights and associated minimum loss
    """
    w_start = initial_w
    w = w_start
    loss_old = 0.0
    
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, max_iters):
        loss = compute_loss(minibatch_y, minibatch_tx, w)
        gradient = compute_gradient(minibatch_y, minibatch_tx, w)
        loss_reg, gradient_reg = regularizer(lambda_, w)
        loss = loss + loss_reg
        gradient = gradient + gradient_reg
        w = w - gamma * gradient
            
        if check_stop(loss, loss_old):
            print('break!')
            break;
        loss_old = loss
        
    return w, loss
