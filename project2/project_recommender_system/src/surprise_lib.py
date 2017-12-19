"""
CS-433 : Machine Learning
Project 2 -- Recommender Systems

Team :
* Dylan Bourgeois
* Antoine Mougeot
* Philippe Verbist

---

surprise_lib.py : interface for running the baselines with Suprise.

We used the Surprise lib to compute baselines in this project.
"""

from surprise import SVD, SVDpp, NMF
from surprise import Dataset, Reader
from surprise import evaluate, print_perf
import pandas as pd
import random
import data as datahelper
import os

def runSurprise(algo, train, test, algo_string, n_folds=5, writeCSV=False, file_name="result.csv"):
    """
    Run the fitting procedure on the training data. Write the result for the test
    data in its "Result" data field.

    Args:
        algo: Surprise algorithm (SVD, SVDpp, NMF, etc)
        train (Panda DataFrame): training data
        test (Panda dDtaFrame): test data
        algo_string (string): printable name of the algorithm
        n_folds (int): Number of k-folds
        writeCSV (bool): set to True to write a .csv submission
        file_name (string): name for the .csv file
    """
    if writeCSV:
        sub = datahelper.load_submission()

    df = pd.DataFrame(train)

    # A reader is needed but only the rating_scale param is requiered.
    reader = Reader(rating_scale=(1, 5))
    # The columns must correspond to user id, item id and ratings (in that order).
    data = Dataset.load_from_df(df[['User', 'Item', 'Rating']], reader)
    random.seed(42)
    data.split(n_folds=n_folds)

    # Evaluate performances of our algorithm on the dataset.
    perf = evaluate(algo, data, measures=['RMSE'])
    print_perf(perf)

    for index, row in test.iterrows():
        test.at[index, "Result"] = algo.estimate(row['User']-1, row['Item']-1)

    if writeCSV:
        file_out = open(file_name, 'w')
        file_out.truncate()
        file_out.write('Id,Prediction\n')

        for index, row in sub.iterrows():
            file_out.write("r{us}_c{mo},{res}\n".format(us=row['User'],mo= row['Item'],
                        res=algo.estimate(row['User']-1, row['Item']-1)))
        file_out.close()


def runAll():
    """
    Run 3 different algorithms from the Surprise library: SVD, SVDpp and NMF.
    Loads data from "data/data/data_train.csv",
    divides it in "train" (for learning) and "test" (for evaluating),
    and runs the Surprise algorithms on it.
    The "test" data is stored in a pickle for later use.
    """
    
    svd_path = os.path.join('..','data','baselines','surprise_SVD_test.p')
    if not os.path.isfile(svd_path):
        train, _, test, _ = datahelper.load_data(test_size=0.1, train_size=0.9)
        train["Rating"] = train.Prediction
        test["Rating"] = test.Prediction

        runSurprise(SVD(n_epochs=30, lr_all=0.001, reg_all=0.001), train, test, "SVD")
        test.to_pickle(svd_path)

    nmf_path = os.path.join('..','data','baselines','surprise_NMF_test.p')
    if not os.path.isfile(nmf_path):
        train, _, test, _ = datahelper.load_data(test_size=0.1, train_size=0.9)
        train["Rating"] = train.Prediction
        test["Rating"] = test.Prediction

        runSurprise(NMF(), train, test, "NMF")
        test.to_pickle(nmf_path)

    svdpp_path = os.path.join('..','data','baselines','surprise_SVDPP_test.p')
    if not os.path.isfile(svdpp_path):
        train, _, test, _ = datahelper.load_data(test_size=0.1, train_size=0.2)
        train["Rating"] = train.Prediction
        test["Rating"] = test.Prediction

        runSurprise(SVDpp(), train, test, "SVDpp")
        test.to_pickle(svdpp_path)
