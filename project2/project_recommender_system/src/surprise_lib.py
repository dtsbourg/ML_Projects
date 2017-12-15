from surprise import SVD, SVDpp, NMF
from surprise import Dataset, Reader
from surprise import evaluate, print_perf
import pandas as pd
import random
import data as datahelper

def runSurprise(algo, n_folds=5, train_size=1.0, write=False, file_name="result.csv"):
    train, _, _, _ = datahelper.load_data(path='../data/data_train.csv', test_size=0.0, train_size=train_size)
    train["Rating"] = train.Prediction

    if write:
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

    if write:
        file_out = open(file_name, 'w')
        file_out.truncate()
        file_out.write('Id,Prediction\n')
        
        for index, row in sub.iterrows():
            file_out.write("r{us}_c{mo},{res}\n".format(us=row['User'],mo= row['Item'],
                        res=algo.estimate(row['User']-1, row['Item']-1)))
        file_out.close()

runSurprise(algo=SVD(n_epochs=30, lr_all=0.001, reg_all=0.001))
runSurprise(algo=SVDpp(), train_size=0.1)
runSurprise(algo=NMF())
