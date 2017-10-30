import numpy as np
from proj1_helpers import *

# Contains all auxiliary functions used in implementations.py

def compute_mse(e):
    """
        Compute the mean square error of a given vector

        Inputs :
            e : vector

        Output :
            Mean Square Error
    """
    return 1/2 * np.mean(e**2)

def compute_loss(y, tx, w):
    """
        Compute the loss between prediction vectors and the product of samples and weights

        Inputs :
            y : Predictions vector
            tx : Samples
            w : Weights

        Output :
            Loss
    """
    e = y - tx.dot(w)
    return compute_mse(e)

def compute_gradient(y, tx, w): # OK
    """
        Compute the gradient for the Gradient Descent method

        Inputs :
            y : Predictions vector
            tx : Samples
            w : Weights

        Output :
            grad : Gradient for the given input
    """
    e = y - tx.dot(w)
    N = y.shape[0]

    grad = -1.0/N * np.dot(tx.T, e)
    return grad

def standardize(x): # OK
    """
        Standardize the original data set.

        Input :
            x : array N x D

        Outputs :
            x : array size N x D
            mean_x : vec size D
            std_x : vec size D
    """
    mean_x = np.mean(x, axis =0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x

def de_standardize(x, mean_x, std_x): # OK
    """
        Reverse the procedure of standardization.

        Inputs :
            x : array size N x D
            mean_x : vec size D
            std_x : vec size D

        Output :
            x : array N x D
    """
    x = x * std_x
    x = x + mean_x
    return x

def sigmoid(t): # OK
    """
    Returns the sigmoid function of t

        Input :
            t : scalar or np.array

        Output :
            scalar or np.array, depending on t
    """
    t[t>20.0]=20.0
    t[t<-20.0]=-20.0

    return 1. / (1. + np.exp(-t))

def compute_logistic_loss(y, tx, w):
    prediction = sigmoid(tx.dot(w))
    loss = -y.T.dot(np.log(prediction)) - (1.-y).T.dot(np.log(1. - prediction)) # gives NaN ...
    #loss = np.sum(np.log(1.0+np.exp(tx.dot(w))) - y*(tx.dot(w)))
    return loss#[0][0]

def compute_logistic_gradient(y, tx, w):
    prediction = sigmoid(tx.dot(w))
    gradient = tx.T.dot(prediction - y)
    return gradient

def regularizer(lambda_, w):
    loss_reg = lambda_ * w.T.dot(w)
    gradient_reg = 2 * lambda_ * w
    return loss_reg, gradient_reg

def build_poly_old(x, degree): # A supprimer?
    """
        polynomial basis functions for input data x, for j=0 up to j=degree

        Inputs :
            x
            degree

        Ouputs :
            an array containing the polynomial expension of x

    """
    manifold = lambda x, degree: [x**j for j in range(0,degree)]
    poly = [np.asarray(manifold(val, degree)).flatten() for val in x]
    return np.asarray(poly)


def build_poly(x, degree): # OK fonctionne, mais pas le plus beau
    """
    polynomial basis functions for input data x, for j=0 up to j=degree.

        INPUT:
            matrix N x D
                [ X_01 X_02 ... X_0D
                  X_11 X_12 ... X_1D
                    .    .   .    .
                  X_N1 X_N2 ... X_ND]
        OUTPUT
            matrix  N x (D * degree + 1)
                [ 1 X_01 X_02 ... X_0D  X_01^2 X_02^2 ... X_0D^degree
                  1 X_11 X_12 ... X_1D  ...
                     .    .   .    .
                  1 X_N1 X_N2 ... X_ND  X_N1^2 X_N2^2 ... X_ND^degree]
    """
    if(degree < 1): # should not happend
        return x

    new_x = np.c_[np.ones((x.shape[0], 1)), x]

    if degree <= 1:
        return new_x
    else:
        for i in range(degree-1):
            cur_degree = i+2
            new_x =  np.c_[new_x, x**cur_degree]
    return new_x




def generate_w(dim, num_intervals, upper, lower):
    """
    Generate a grid of values for [w0, ..., wd].
    """
    return [[np.linspace(lower,upper,num_intervals)] for _ in range(dim)]

def build_k_indices(y, k_fold, seed):
    """
    build k indices for k-fold.
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices).astype(int)

def build_idx(ranges):
    clean_idx = np.asarray([list(range(i,j)) for i,j in ranges])
    return [item for sublist in clean_idx for item in sublist]

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def compute_classification_error(y, tx, w, logistic_reg=True):
    y_pred = predict_labels(w, tx, logistic_reg)
    s = np.sum(y != y_pred)
    return s, s/y.shape[0]


def sample_data(y, x, seed, size_samples): # OK
    """
    sample from dataset.
    """
    np.random.seed(seed)
    num_observations = y.shape[0]
    random_permuted_indices = np.random.permutation(num_observations)
    y = y[random_permuted_indices]
    x = x[random_permuted_indices]
    return y[:size_samples], x[:size_samples]



def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8
    you will have 80 % of your data set dedicated to training
    and the rest dedicated to testing
    """

    N = y.shape[0]
    idx = np.arange(N)
    np.random.seed(seed)
    np.random.shuffle(idx)

    x_shuffle = x[idx,:]
    y_shuffle = y[idx]

    ind_lim = int(N*ratio)

    x_train = x_shuffle[0:ind_lim]
    y_train = y_shuffle[0:ind_lim]

    x_test = x_shuffle[ind_lim:]
    y_test = y_shuffle[ind_lim:]

    return x_train, x_test, y_train, y_test



def split_data_old(x, y, ratio, seed=1): # to be deleted
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

    first_loop = True

    while(N_sel/N < ratio):
        idx = np.random.randint(0, N-N_sel, 1)

        if first_loop:
            x_train = x_copy[idx]
            y_train = y_copy[idx]
            first_loop=False
        else:
            x_train = np.vstack((x_train, x_copy[idx]))
            y_train = np.append(y_train, y_copy[idx])

        x_copy = np.delete(x_copy, idx, axis=0)
        y_copy = np.delete(y_copy, idx, axis=0)
        N_sel +=1.0

    x_test = x_copy
    y_test = y_copy

    return x_train, x_test, y_train, y_test


def check_stop(a, b, threshold=1e-7):
    if abs(a-b) < threshold:
        return True
    else:
        return False


def split_data_k_fold(x, y, k_indices, k):
    k_indices_train = np.array([])

    for i in range(k_indices.shape[0]):
        if i != k:
            k_indices_train = np.append(k_indices_train, k_indices[i])

    k_indices_test = k_indices[k]

    x_train = x[k_indices_train.astype(int)]
    y_train = y[k_indices_train.astype(int)]

    x_test = x[k_indices_test.astype(int)]
    y_test = y[k_indices_test.astype(int)]

    return x_train, x_test, y_train, y_test



def reduce_size_of_losses(losses):
    red = [0] * len(losses)
    for r in range(len(losses)):
        red[r] = np.linalg.norm(losses[r])
    return red
