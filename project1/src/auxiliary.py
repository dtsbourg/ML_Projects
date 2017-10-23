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

def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        gradients = compute_gradient(y,tx,w)
        loss = compute_loss(y,tx,w)
        w = w - [gamma * g for g in gradients]
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x

def sigmoid(t):
    return 1 / (1 + np.exp(-t))

def compute_logistic_loss(y, tx, w):
    #prediction = sigmoid(tx.dot(w))
    #loss = -y.T.dot(np.log(prediction)) - (1-y).T.dot(np.log(1 - prediction)) # gives NaN ...
    loss = np.sum(np.log(1+np.exp(tx.dot(w))) - y*(tx.dot(w))) 
    return loss#[0][0]

def compute_logistic_gradient(y, tx, w):
    prediction = sigmoid(tx.dot(w))
    gradient = tx.T.dot(prediction - y)
    return gradient

def regularizer(lambda_, w):
    loss_reg = lambda_ * w.T.dot(w)[0][0]
    gradient_reg = 2 * lambda_ * w
    return loss_reg, gradient_reg

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    if(degree < 1):
        return x
    
    new_x = np.zeros(np.concatenate([np.array(x.shape), [degree+1]]))
    new_x[..., 0] = x**0 # add zeros (for degree 0)
    new_x[..., 1] = x**1 # add x (for degree 1)

    if degree <= 1:
        return new_x
    else:  
        for i in range(degree-1):
            cur_degree = i+2
            new_x[...,cur_degree] = x**cur_degree

    return new_x

def generate_w(dim, num_intervals, upper, lower):
    """Generate a grid of values for [w0, ..., wd]."""
    return [[np.linspace(lower,upper,num_intervals)] for _ in range(dim)]

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def build_idx(ranges):
    clean_idx = np.asarray([list(range(i,j)) for i,j in ranges])
    return [item for sublist in clean_idx for item in sublist]

def compute_classification_error(y, tx, w):
    # TODO : Check impl
    s = sigmoid(tx.dot(w))
    result = np.ones(y.shape[0])
    for idx, y_n in enumerate(y):
        #print(idx, y_n, s[idx])
        if y_n == 1 and s[idx] >= 0.5:
            result[idx] = 0.0
        elif y_n == -1 and s[idx] < 0.5:
            result[idx] = 0.0
    return result.sum()/y.shape[0]