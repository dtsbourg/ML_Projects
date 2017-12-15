from surprise import SVD, SVDpp, NMF
from surprise import Dataset, Reader
from surprise import evaluate, print_perf
import pandas as pd
import random
import data as datahelper

def load_csv(filename):
    df = pd.read_csv(filename)
    df['User'] = df['Id'].apply(lambda x: int(x.split('_')[0][1:]))
    df['Movie'] = df['Id'].apply(lambda x: int(x.split('_')[1][1:]))
    df['Rating'] = df['Prediction']
    df = df.drop(['Id', 'Prediction'], axis=1)
    return df

def runSVD():
    train, _, _, _ = datahelper.load_data(path='../data/data_train.csv', test_size=0.0, train_size=1.0)
    train["Rating"] = train.Prediction

    sub = datahelper.load_submission()

    df = pd.DataFrame(train)

    # A reader is needed but only the rating_scale param is requiered.
    reader = Reader(rating_scale=(1, 5))
    # The columns must correspond to user id, item id and ratings (in that order).
    data = Dataset.load_from_df(df[['User', 'Item', 'Rating']], reader)
    random.seed(42)
    data.split(n_folds=5)  # data can now be used normally

    # We'll use the famous SVD algorithm.
    algo = SVD(n_epochs=30, lr_all=0.001, reg_all=0.001)

    # Evaluate performances of our algorithm on the dataset.
    perf = evaluate(algo, data, measures=['RMSE'])
    print_perf(perf)


    file_out = open('resultSVD.csv', 'w')
    file_out.truncate()
    file_out.write('Id,Prediction\n')

    for index, row in test.iterrows():
        file_out.write("r{us}_c{mo},{res}\n".format(us=row['User'],mo= row['Item'],
                    res=algo.estimate(row['User'], row['Item'])))
    file_out.close()



def runSVDpp():
    train, _, _, _ = datahelper.load_data(path='../data/data_train.csv', test_size=0.0, train_size=1.0)
    train["Rating"] = train.Prediction

    sub = datahelper.load_submission()

    df = pd.DataFrame(train)

    # A reader is needed but only the rating_scale param is requiered.
    reader = Reader(rating_scale=(1, 5))
    # The columns must correspond to user id, item id and ratings (in that order).
    data = Dataset.load_from_df(df[['User', 'Item', 'Rating']], reader)
    random.seed(42)
    data.split(n_folds=2)  # data can now be used normally

    # We'll use the famous SVD algorithm.
    algo = SVDpp()

    # Evaluate performances of our algorithm on the dataset.
    perf = evaluate(algo, data, measures=['RMSE'])
    print_perf(perf)

    file_out = open('resultSVDpp.csv', 'w')
    file_out.truncate()
    file_out.write('Id,Prediction\n')

    for index, row in sub.iterrows():
        file_out.write("r{us}_c{mo},{res}\n".format(us=row['User'],mo= row['Item'],
                    res=algo.estimate(row['User'], row['Item'])))
    file_out.close()


def runNMF():
    train, _, _, _ = datahelper.load_data(path='../data/data_train.csv', test_size=0.0, train_size=1.0)
    train["Rating"] = train.Prediction

    sub = datahelper.load_submission()

    df = pd.DataFrame(train)

    # A reader is needed but only the rating_scale param is requiered.
    reader = Reader(rating_scale=(1, 5))
    # The columns must correspond to user id, item id and ratings (in that order).
    data = Dataset.load_from_df(df[['User', 'Item', 'Rating']], reader)
    random.seed(42)
    data.split(n_folds=2)  # data can now be used normally

    # We'll use the famous SVD algorithm.
    algo = NMF()

    # Evaluate performances of our algorithm on the dataset.
    perf = evaluate(algo, data, measures=['RMSE'])
    print_perf(perf)

    file_out = open('resultNMF.csv', 'w')
    file_out.truncate()
    file_out.write('Id,Prediction\n')

    for index, row in sub.iterrows():
        file_out.write("r{us}_c{mo},{res}\n".format(us=row['User'],mo= row['Item'],
                    res=algo.estimate(row['User'], row['Item'])))
    file_out.close()



runNMF()

