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


    ids[0], y[0], X[0] = load_csv_data('train.csv') # load the data (no subset)
    len_tot = X[0].shape[0]

    if verbose:
        print(" - Split training data")

    ids, y, X = split_data_boson(ids[0], X[0], y[0])        # split the data in 8 subsets, and delete all feature that are -999


    for r in range(len(X)):
        x[r], _, _ = standardize(X[r])    # standardize
        clean_data_ratio_for_subset = len([x for x in x[r] if -999. in x])/len(x[r])*100
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
    split_ratio=0.4
    for r in range(len(x)):
        x[r], _, y[r], _ = split_data(x[r], y[r], split_ratio, seed) # reduce size to make optimisation quicker


    degrees = [1, 2, 3, 4, 5]
    gammas = np.logspace(-7, -1, 15)
    lambdas = np.logspace(-5, 1, 15)

    #degrees = [2]
    #gammas = [1e-6, 1e-5]
    #lambdas = [0.001, 0.01]


    ws = []
    ratio_err_trains = []
    ratio_err_tests = []

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


    for idx1, degree in enumerate(degrees):
        for r in range(len(x)):
            phi[r]=build_poly(x[r], degree)

        for idx2, gamma in enumerate(gammas):
            for idx3, lambda_ in enumerate(lambdas):
                for k in range(k_fold):
                    ratio_err_trains_k_fold[k]= 0.0
                    ratio_err_tests_k_fold[k] = 0.0
                    for r in range(len(x)):
                        w[r], ratio_error_train, ratio_error_test = reg_logistic_regression_split_k_fold(
                            phi[r], y[r], gamma, lambda_, k_indices[r], k)

                        ratio_err_trains_k_fold[k]+=ratio_error_train*prop[r]
                        ratio_err_tests_k_fold[k] +=ratio_error_test*prop[r]

                    w_k_fold[k]=copy.copy(w)

                idx_k=np.argmin(ratio_err_tests_k_fold)
                ws.append(w_k_fold[idx_k])
                ratio_err_trains.append(ratio_err_trains_k_fold[idx_k])
                ratio_err_tests.append(ratio_err_tests_k_fold[idx_k])
                
                if vverbose:
                    print("{exp}: k_sel={k}, degree={d}, gamma={g}, lambda={l}, ratio_err_train={er_tr:.3f}, ratio_err_test={er_te:.3f}".format(
                           exp=exp, k=idx_k,  
                           d=degree, g=gamma, l=lambda_,
                           er_tr=ratio_err_trains[exp], er_te=ratio_err_tests[exp]))

                exp += 1



    idx = np.argmin(ratio_err_tests)
    w_reg_log_reg = ws[idx]


    idx_split = int(idx/(len(degrees)*len(gammas)*len(lambdas)))
    idx_degree_gamma_lamb = idx%(len(degrees)*len(gammas)*len(lambdas))
    idx_degree = int(idx_degree_gamma_lamb/(len(gammas)*len(lambdas)))
    idx_gamma_lamb = idx_degree_gamma_lamb%(len(gammas)*len(lambdas))
    idx_gamma = int(idx_gamma_lamb/len(gammas))
    idx_lamb = (idx_gamma_lamb%len(lambdas))


    if verbose:
        print("Take experiments {i} out of {tot} experiments, error tr={er_tr:.3f}, error test={er_te:.3f}, k_fold={k_fold} degree={deg}, gamma={g}".format(
               i=idx, tot=len(ws), er_tr=ratio_err_trains[idx], er_te=ratio_err_tests[idx], 
               k_fold=k_fold,  deg=degrees[idx_degree], g=gammas[idx_gamma]))




    if verbose:
        print(" - Learning done. Expected success ratio: {sr}".format(sr=1.0-ratio_err_tests[idx]))
        
    pickle.dump(w_reg_log_reg, open( 'w_reg_log_reg.p', 'wb' ))

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform optimization on the parameters of reg_logistic_regression')
    parser.add_argument('-v', action='store_true', help='Show some steps', default=False)
    parser.add_argument('-V', action='store_true', help='Show a lot of steps', default=False)
    parser.add_argument('-ps', action='store_true', help='Store data to pickle', default=False)
    parser.add_argument('-pl', action='store_true', help='Load data from pickle', default=False)
    args = parser.parse_args()

    main(args.v, args.V, args.ps, args.pl)
        
              