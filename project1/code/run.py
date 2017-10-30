from proj1_helpers import *
from implementations import *
from auxiliary import *
from data_splitting import *
import pickle
import datetime
import random
import copy
import sys, argparse


def main(v, V):

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
        printf("No log messages will be displayed. To see some information, please use the option -v or -V")


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

    ratio_err_train_final = 0.0
    ratio_err_test_final  = 0.0


    # ========================
    # PERFORM MACHINE LEARNING
    # ========================

    if verbose:
        print(" - Perform learning")


    seed = 7
    split_ratio = 0.7
    max_iters = 1000

    # hard-coded parameters for jets n°1-8
    degree = [2, 	3, 	2, 	3, 	3, 	3, 	3, 	3] #[3]
    gamma  = [1e-6, 	5e-6, 	1e-4, 	1e-6, 	1e-4, 	1e-6,	1e-4, 	5e-6]#[1e-6]   
    lambda_= [1e-4, 	1e-3, 	10, 	10, 	1e-3, 	1e-3, 	0, 	3e-6]#[0]*8


    for r in range(len(x)):
        if vverbose:
            print("Learning for subset {r}/8".format(r=r+1))
        x_train, x_test, y_train, y_test = split_data(x[r], y[r], split_ratio, seed)

        # form train and test data with polynomial basis function
        phi_train = build_poly(x_train, degree[r])
        phi_test  = build_poly(x_test, degree[r])

        w_initial = np.array([0]*(phi_train.shape[1]))

        w[r],_ = reg_logistic_regression(y_train, phi_train, lambda_[r], w_initial, max_iters, gamma[r])

        # compute error and loss for train and test data
        _, ratio_error_train = compute_classification_error(y_train, phi_train, w[r], logistic_reg=True)
        _, ratio_error_test = compute_classification_error(y_test, phi_test, w[r], logistic_reg=True)

        ratio_err_train_final += ratio_error_train*prop[r]
        ratio_err_test_final  += ratio_error_test*prop[r]
        # print("Error for subset {r}: {er_tr}, {er_te}".format(r=r, er_tr=ratio_error_train, er_te=ratio_error_test))


    if vverbose:
        print("Learning done. Expected success ratio: {sr}".format(sr=1.0-ratio_err_test_final))


    # =====================
    # LOAD DATA TO EVALUATE
    # =====================  

    if verbose:
        print(" - Load data to be classified")

    # declare variables
    ids_ukn    =[0]*nb_subsets
    y_ukn      =[0]*nb_subsets
    X_ukn      =[0]*nb_subsets
    x_ukn      =[0]*nb_subsets
    phi_ukn    =[0]*nb_subsets
    mean_x_ukn =[0]*nb_subsets
    std_x_ukn  =[0]*nb_subsets

    ids_ukn[0], y_ukn[0], X_ukn[0] = load_csv_data('test.csv')

    if verbose:
        print(" - Split data to be classified")

    ids_ukn, y_ukn, X_ukn = split_data_boson(ids_ukn[0], X_ukn[0], y_ukn[0])

    for r in range(len(X_ukn)):
        x_ukn[r], mean_x_ukn[r], std_x_ukn[r] = standardize(X_ukn[r])    # standardize
        clean_data = [x for x in x_ukn[r] if -999. in x]                 # check tidyness of data
        clean_data_ratio_for_subset =  len(clean_data)/len(x_ukn[r])*100
        phi_ukn[r] = build_poly(x_ukn[r], 1)                              # add column of 1's 
        if vverbose:
            print("Testing subset {r} has {c}% of remaining NaN values".format(
                  r=r, c=clean_data_ratio_for_subset))


    # =============================================
    # EVALUATE LABELS AND WRITE RESULT IN CSV FILE
    # =============================================

    if verbose:
        print(" - Perform classification")

    ids_to_write = np.array([])
    y_to_write = np.array([])
    filename='submission_run.csv'

    for r in range(len(x_ukn)):
        if vverbose:
            print("Evaluate labels for subset {r}/8".format(r=r+1))
        phi_ukn[r] = build_poly(x_ukn[r], degree[r]) 
        y_ukn[r] = predict_labels(w[r], phi_ukn[r], logistic_reg=True)
        y_ukn[r][y_ukn[r]==0] = -1

        ids_to_write = np.concatenate([ids_to_write, ids_ukn[r]], axis=0)
        y_to_write = np.concatenate([y_to_write, y_ukn[r]], axis=0)

    create_csv_submission(ids_to_write, y_to_write, filename)

    if verbose:
        print(" - {fn} written!".format(fn=filename))        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-v', action='store_true', help='Show some steps', default=False)
    parser.add_argument('-V', action='store_true',
                        help='Show a lot of steps',
                        default=False)
    args = parser.parse_args()

    main(args.v, args.V)