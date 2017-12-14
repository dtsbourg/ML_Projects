from surprise import SVD
from surprise import Dataset
from surprise import evaluate, print_perf
import pandas as pd
import numpy as np
import datetime

def load_csv(filename):
    df = pd.read_csv(filename)
    df['User'] = df['Id'].apply(lambda x: int(x.split('_')[0][1:]))
    df['Movie'] = df['Id'].apply(lambda x: int(x.split('_')[1][1:]))
    df['Rating'] = df['Prediction']
    df = df.drop(['Id', 'Prediction'], axis=1)
    return df

def load_submission():
    path = '../../data/sampleSubmission.csv'
    ratings = pd.read_csv(path, dtype={'Prediction': np.int})
    pos = ratings.Id.str.extract('r([0-9]+)_c([0-9]+)', expand=True)
    ratings['User'] = pos[0].astype(np.int)
    ratings['Item'] = pos[1].astype(np.int)
    return ratings


def save_csv(filename, data):
    for index, row in test.iterrows():
        a = algo.estimate(row['User'], row['Movie'])

model_uid = datetime.datetime.now().strftime("%d-%m-%Y-%H:%M:%S")

#test = load_csv('../../data/data/sampleSubmission.csv')

sub = load_submission()
print(type(sub))

#for index, row in test.iterrows():    
#    print (row['User'])


# Load the movielens-100k dataset (download it if needed),
# and split it into 3 folds for cross-validation.
data = Dataset.load_builtin('ml-100k')
data.split(n_folds=2)

# We'll use the famous SVD algorithm.
algo = SVD()

# Evaluate performances of our algorithm on the dataset.
perf = evaluate(algo, data, measures=['RMSE', 'MAE'])

print_perf(perf)

file_out = open('result_suprise.csv', 'w')
file_out.truncate()
file_out.write('Id,Prediction\n')

for index, row in sub.iterrows():
     row['Prediction'] = algo.estimate(row['User'], row['Item'])
     file_out.write("r{us}_c{mo},{res}\n".format(us=row['User'],mo= row['Item'],res=int(round(row['Prediction']))))
file_out.close()
    
