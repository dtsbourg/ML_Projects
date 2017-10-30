import numpy as np
from auxiliary import *

# Contains all methods asked in step 2

def least_squares(y, tx):
    """
    Linear regresison using normal equations

        Inputs:
            y : Predictions
            tx : Samples

        Ourputs :
            w : Best weights
            loss : Minimum loss
    """
    a = tx.T.dot(tx)
    b = tx.T.dot(y)

    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    return w, loss

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent

        Inputs :
            y : Predictions
            tx : Samples
            initial_w : Initial weights
            max_iters : Maximum number of iterations
            gamma : Step size

        Outputs :
            w : Best weights
            loss : Minimum loss
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

        Inputs :
            y : Predictions
            tx : Samples
            initial_w : Initial weights
            max_iters : Maximum number of iterations
            gamma : Step size

        Outputs :
            w : Best weights
            loss : Minimum loss
    """
    w = initial_w
    loss = compute_loss(y, tx, w)
    batch_size=1

    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, max_iters):
        loss = compute_loss(y, tx, w) # avant: minibatch_y et minibatch_tx
        gradients = compute_gradient(minibatch_y, minibatch_tx, w)
        w = w - gamma * gradients

    return w, loss

def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations

        Inputs :
            y : Predictions
            tx : Samples
            lambda_ : regularization parameter

        Outputs :
            w : Best weights
            loss : Minimum loss
    """
    lambda_prime = lambda_ * 2*tx.shape[0]

    a = tx.T.dot(tx) + lambda_prime*np.eye(tx.shape[1])
    b = tx.T.dot(y)
    w_star = np.linalg.solve(a, b)

    loss = compute_loss(y, tx, w_star)

    return w_star, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma, SGD=False, batch_size=-1) :
    """
    Logistic regression using gradient descent

        Inputs :
            y : Predictions
            tx : Samples
            lambda_ : regularization parameter
            initial_w : Initial weights
            max_iters : Maximum number of iterations
            gamma : Step size
            batch_size :

        Outputs :
            w : Best weights
            loss : Minimum loss
    """
    w_start = initial_w
    w = w_start
    loss_old = 0.0

    for n_iter in range(max_iters):
        loss = compute_logistic_loss(y, tx, w)
        gradient = compute_logistic_gradient(y, tx, w)
        w = w - gamma * gradient

        if check_stop(loss, loss_old):
            #print('break!')
            break;
        loss_old = loss

    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, SGD=False, batch_size=-1):
    """
    Regularized logistic regression using gradient descent

        Inputs :
            y : Predictions
            tx : Samples
            lambda_ : regularization parameter
            initial_w : Initial weights
            max_iters : Maximum number of iterations
            gamma : Step size
            batch_size :

        Outputs :
            w : Best weights
            loss : Minimum loss
    """
    w_start = initial_w
    w = w_start
    loss_old = 0.0

    if SGD:
        if(batch_size==-1): # compute automatically the maximum batch size
            batch_size = int(y.shape[0]/max_iters)
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, max_iters):
            loss = compute_loss(minibatch_y, minibatch_tx, w)
            gradient = compute_gradient(minibatch_y, minibatch_tx, w)
            loss_reg, gradient_reg = regularizer(lambda_, w)
            loss = loss + loss_reg
            gradient = gradient + gradient_reg
            w = w - gamma * gradient

            if check_stop(loss, loss_old):
                #print('break!')
                break;
            loss_old = loss
        return w, loss

    else:
        for n_iter in range(max_iters):
            loss = compute_logistic_loss(y, tx, w)
            gradient = compute_logistic_gradient(y, tx, w)
            loss_reg, gradient_reg = regularizer(lambda_, w)
            loss_new = loss + loss_reg
            gradient = gradient + gradient_reg
            w = w - gamma * gradient

            if check_stop(loss, loss_old):
                #print('break!')
                break;
            loss_old = loss
        return w, loss

def logistic_regression_SGD(y, tx, initial_w, max_iters, gamma, batch_size) :
    """
    Logistic regression using stochastic gradient descent

        Inputs :
            y : Predictions
            tx : Samples
            initial_w : Initial weights
            max_iters : Maximum number of iterations
            gamma : Step size
            batch_size

        Outputs :
            w : Best weights
            loss : Minimum loss
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

        Inputs :
            y : Predictions
            tx : Samples
            lambda_ : regularization parameter
            initial_w : Initial weights
            max_iters : Maximum number of iterations
            gamma : Step size

        Outputs :
            w : Best weights
            loss : Minimum loss
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
