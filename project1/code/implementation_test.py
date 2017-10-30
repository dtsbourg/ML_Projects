from proj1_helpers import *
from implementations import *
from auxiliary import *
import pickle
import random
import copy


# ==========
# PARAMETERS
# ==========
verbose = True


# =====================
# VARIABLES DECLARATION
# =====================
nb_subsets=1





# ==================
# LOAD TRAINING DATA
# ==================

if verbose:
    print("Load training data")

ids, y, X = load_csv_data('train.csv') # load the data (no subset)

ranges = [(1,4), (7,12), (13,23), (29,30)]
keep_idx = build_idx(ranges)
X = X[:,keep_idx]                # remove all features with at least one NaN

x, _, _ = standardize(X)         # standardize

        
        

# ========================
# PERFORM MACHINE LEARNING
# ========================
split_ratio = 0.75
seed=6
x_train, x_test, y_train, y_test = split_data(x, y, split_ratio, seed)
tx_train = build_poly(x_train, 1)      # add column of 1's
tx_test  = build_poly(x_test, 1)      # add column of 1's
w_initial = np.zeros([tx_train.shape[1]])


# Least squares Gradient Descent

max_iters = 100
gamma = 0.2

w_LS_GD, _ = least_squares_GD(y_train, tx_train, w_initial, max_iters, gamma)

_, error_ratio = compute_classification_error(y_test, tx_test, w_LS_GD, logistic_reg=False)

print(" --> LS GD        : Succeed ratio={sr:5.2f}%, gamma={ga}, max_iter={mi}".format(
           sr=(1.-error_ratio)*100.0, ga=gamma, mi=max_iters))




# Least squares Stochastic Gradient Descent

max_iters = 1000
gamma = 1e-2

w_LS_SGD, _ = least_squares_SGD(y_train, tx_train, w_initial, max_iters, gamma)

_, error_ratio = compute_classification_error(y_test, tx_test, w_LS_SGD, logistic_reg=False)

print(" --> LS SGD       : Succeed ratio={sr:5.2f}%, gamma={ga}, max_iter={mi}".format(
           sr=(1.-error_ratio)*100.0, ga=gamma, mi=max_iters))



# Least squares

w_LS, loss = least_squares(y_train,tx_train)
_, error_ratio = compute_classification_error(y_test, tx_test, w_LS, logistic_reg=False)

print(" --> LS           : Succeed ratio={sr:5.2f}%".format(sr=(1.-error_ratio)*100.0))



# Ridge regression

lambda_ = 0.01
    
w_ridge_reg, _ = ridge_regression(y_train, tx_train, lambda_)
_, error_ratio = compute_classification_error(y_test, tx_test, w_ridge_reg, logistic_reg=False)

print(" --> Ridge        : Succeed ratio={sr:5.2f}%, lambda={l}".format(sr=(1.-error_ratio)*100.0, l=lambda_))


# Logistic regression

max_iters = 1000
gamma = 1e-6

w_LR, _ = logistic_regression(y_train, tx_train, w_initial, max_iters, gamma)
_, error_ratio = compute_classification_error(y_test, tx_test, w_LR, logistic_reg=True)

print(" --> Logistic     : Succeed ratio={sr:5.2f}%, gamma={g}, max_iter={mi}".format(
    sr=(1.-error_ratio)*100.0, g=gamma, mi=max_iters))


# Regularized Logistic regression

max_iters = 1000
gamma = 1e-5
lambda_ = 0.01

w_LR_reg, _ = reg_logistic_regression(y_train, tx_train, lambda_, w_initial, max_iters, gamma)
_, error_ratio = compute_classification_error(y_test, tx_test, w_LR_reg, logistic_reg=True)

print(" --> Logistic Reg : Succeed ratio={sr:5.2f}%, gamma={g}, lambda={l}, max_iter={mi}".format(
    sr=(1.-error_ratio)*100.0, g=gamma, l=lambda_, mi=max_iters))


              