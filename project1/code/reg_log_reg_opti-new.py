from proj1_helpers import *
from implementations import *
from auxiliary import *
from data_splitting import *
import pickle
import matplotlib.pyplot as plt
import datetime
import random
import copy
import argparse




def main(v, V, ps, pl):
    
    verbose=True
    vverbose=True
    
    if V:
        verbose=True
        vverbose=True
    elif v:
        verbose=True
        vverbose=False
    else:
        verbose=False
        vverbose=False

    # =====================
    # VARIBALES DECLARATION
    # =====================
    nb_subsets=8


    # ==================
    # LOAD TRAINING DATA
    # ==================

    if verbose:
        print(" - Load training data")

    # declare all variables as lists
    ids   =[0]*nb_subsets # ids
    y     =[0]*nb_subsets # labels
    X     =[0]*nb_subsets # non-standardzed features
    x     =[0]*nb_subsets # standardized features
    phi   =[0]*nb_subsets # extended feature vector

    prop  =[0]*nb_subsets # proportion of the r^th subset compared to the total set of data 
                          # (used to evaluate the performances of the estimator 


    ids[0], y[0], X[0] = load_csv_data('../data/train.csv') # load the data (no subset)
    len_tot = X[0].shape[0]

    if verbose:
        print(" - Split training data")

    ids, y, X = split_data_boson(ids[0], X[0], y[0])        # split the data in 8 subsets, and delete all feature that are -999


    for r in range(len(X)):
        x[r], _, _ = standardize(X[r])    # standardize
        #clean_data_ratio_for_subset = len([x for x in x[r] if -999. in x])/len(x[r])*100
        phi[r] = build_poly(x[r], 1)     # add column of 1's
        prop[r] = len(x[r])/len_tot
        if vverbose:
            print("Training subset {r} represents {prop_r:6.2f}% of the total training set and has {c}% of remaining NaN values".format(
                  r=r, prop_r=prop[r]*100, c=clean_data_ratio_for_subset))


    # =====================================
    # VARIABLE NEEDED FOR MACHINE LEARNING
    # =====================================

    w =[0]*nb_subsets

    # ========================
    # PERFORM MACHINE LEARNING
    # ========================

    if verbose:
        print(" - Perform learning")




    def reg_logistic_regression_split_k_fold(x, y, gamma, lambda_, k_indices, k):
        """polynomial regression with different split ratios and different degrees."""

        x_train, x_test, y_train, y_test = split_data_k_fold(x, y, k_indices, k) # split the data

        w_initial = np.array([0]*(x_train.shape[1]))
        max_iters = 1000

        w,_ = reg_logistic_regression(y_train, x_train, lambda_, w_initial, max_iters, gamma)

        # compute error and loss for train and test data
        _, ratio_error_train = compute_classification_error(y_train, x_train, w)
        _, ratio_error_test = compute_classification_error(y_test, x_test, w)

        return w, ratio_error_train, ratio_error_test


    seed=7
    split_ratio=0.7
    for r in range(len(x)):
        x[r], _, y[r], _ = split_data(x[r], y[r], split_ratio, seed) # reduce size to make optimisation quicker


    degrees = [3, 5, 7, 9]
    gammas = np.logspace(-7, -1, 7)
    lambdas = np.logspace(-5, 1, 7)

    ws = []
    ratio_err_trains = []
    ratio_err_tests = []
    errs = []
    
    ws_np = np.empty((len(x),len(degrees),len(gammas), len(lambdas)))
    err_np = np.empty((len(x),len(degrees),len(gammas), len(lambdas)))

    k_fold = 4
    w_k_fold=[0]*k_fold
    ratio_err_trains_k_fold=[0]*k_fold
    ratio_err_tests_k_fold=[0]*k_fold
    k_indices=[0]*8

    exp=0

    for r in range(len(x)):
        k_indices[r] = build_k_indices(y[r], k_fold, seed)

    if vverbose:
        print("There will be {exp} experiments".format(exp=len(degrees)*len(gammas)*len(lambdas)))

    
    for r in range(len(x)):
        for idx1, degree in enumerate(degrees):
            phi = build_poly(x[r], degree)
            for idx2, gamma in enumerate(gammas):
                for idx3, lambda_ in enumerate(lambdas):
                    w_k = []; error_k = []
                    for k in range(k_fold):
                        w, ratio_error_train, ratio_error_test = reg_logistic_regression_split_k_fold(phi, y[r], gamma, lambda_, k_indices[r], k)
                        error_k.append(((ratio_error_train+ratio_error_test)/2.0))
                     
                    idx_k = np.argmin(error_k)
                    err_np[r, idx1, idx2, idx3] = error_k[idx_k]
    
    
    best_params = []
    for r in range(len(x)):
        subset_params = {}
        idx_best = np.unravel_index(err_np[r].argmin(), err_np[r].shape)
        d = degrees[idx_best[0]]
        g = gammas[idx_best[1]]
        l = lambdas[idx_best[2]]
        print("Subset # {}".format(r))
        print("Best degree = {}".format(d))
        print("Best gamma = {}".format(g))
        print("Best lambda = {}".format(l))
        
        print("Predicted error = {}".format(err_np[r,d,g,l]))
        
        subset_params['degree'] = d
        subset_params['gamma'] = g
        subset_params['lambda'] = l

        best_params.append(subset_params)
    
    
    pickle.dump(ws_np,  open( 'weights_log_reg.p', 'wb' ))
    pickle.dump(err_np,  open( 'errors_log_reg.p', 'wb' ))
    pickle.dump(best_params, open( 'best_params.p', 'wb' ))

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform optimization on the parameters of reg_logistic_regression')
    parser.add_argument('-v', action='store_true', help='Show some steps', default=False)
    parser.add_argument('-V', action='store_true', help='Show a lot of steps', default=False)
    parser.add_argument('-ps', action='store_true', help='Store data to pickle', default=False)
    parser.add_argument('-pl', action='store_true', help='Load data from pickle', default=False)
    args = parser.parse_args()

    main(args.v, args.V, args.ps, args.pl)
        
              