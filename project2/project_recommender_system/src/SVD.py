from surprise import SVD, SVDpp
from surprise import Dataset, Reader
from surprise import evaluate, print_perf
import pandas as pd
import random


def load_csv(filename):
    df = pd.read_csv(filename)
    df['User'] = df['Id'].apply(lambda x: int(x.split('_')[0][1:]))
    df['Movie'] = df['Id'].apply(lambda x: int(x.split('_')[1][1:]))
    df['Rating'] = df['Prediction']
    df = df.drop(['Id', 'Prediction'], axis=1)
    return df

train = load_csv('../data/data/data_train.csv')
test = load_csv('../data/data/sampleSubmission.csv')

df = pd.DataFrame(train)

# A reader is still needed but only the rating_scale param is requiered.
reader = Reader(rating_scale=(1, 5))
# The columns must correspond to user id, item id and ratings (in that order).
data = Dataset.load_from_df(df[['User', 'Movie', 'Rating']], reader)
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
 #   row['Prediction'] = algo.estimate(row['User'], row['Movie'])
    file_out.write("r{us}_c{mo},{res}\n".format(us=row['User'],mo= row['Movie'],
                res=algo.estimate(row['User'], row['Movie'])))
file_out.close()
