# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csgraph
import math
import pickle
import os.path


def explore(quick_load = False):
    # load data into a table by extracting user and item numbers
    ratings = pd.read_csv('../data/data/data_train.csv', dtype={'Prediction': np.int})

    print("There are {} ratings".format(len(ratings)))
    print("The smallest rating is {} and the biggest is {}".format(
        ratings.Prediction.min(), ratings.Prediction.max()))

    idx = ratings.Id.str.extract('r([0-9]+)_c([0-9]+)', expand=True)

    user_idx = idx[0].astype(int)
    film_idx = idx[1].astype(int)

    matrix = np.zeros((max(user_idx), max(film_idx)), dtype=np.int)

    if os.path.isfile("svg/matrix.p"):
        matrix = pickle.load(open( 'svg/matrix.p', 'rb' ))
    else:
        for i in range(len(user_idx)): # takes time!
            matrix[user_idx[i]-1, film_idx[i]-1] = ratings.Prediction[i]
        pickle.dump(matrix, open( 'svg/matrix.p', 'wb' ))

    print("There are {} users and {} movies".format(matrix.shape[0], matrix.shape[1]))
    
    fig = plt.figure(figsize = (5,5))
    plt.imshow(matrix, aspect='auto')
    plt.title("Full matrix")
    plt.xlabel("Users")
    plt.ylabel("Movies")
    fig.savefig('../res/img/fullmatrix.png')
    plt.show()


    fig = plt.figure(figsize = (10,6))
    plt.subplot(131)
    plt.imshow(matrix[:150, :200], aspect='auto', interpolation='nearest')
    plt.tight_layout()
    plt.title("Matrix subset")

    occupancy_stat = np.count_nonzero(matrix) / matrix.size
    print("There is an occupancy of {} %".format(str(occupancy_stat*100)))
    print("The data is {} % sparse !".format((1-occupancy_stat)*100))

    matrix_subset = matrix[:,:1000]
    #occupancy_stat = np.count_nonzero(matrix_subset) / matrix_subset.size
    #print("Matrix subset: {} % occupancy".format(str(occupancy_stat*100)))

    # Leave one out test / train split
    # Adapted from https://gist.github.com/Wann-Jiun/d91f7ccbd20659e9725052a9ac5aed10#file-nycdsa_p5_split-py
    train_matrix = matrix_subset.copy()
    test_matrix = np.zeros(matrix_subset.shape)

    if os.path.isfile("svg/train_matrix.p") and os.path.isfile("svg/test_matrix.p"):
        train_matrix = pickle.load(open( 'svg/train_matrix.p', 'rb' ))
        test_matrix = pickle.load(open( 'svg/test_matrix.p', 'rb' ))
    else:
        np.random.seed(42)
        for i in range(1,len(matrix_subset[0])):
            rating_idx = np.random.choice(
                matrix_subset[i, :].nonzero()[0], 
                size=3)
            train_matrix[i, rating_idx] = 0.0
            test_matrix[i, rating_idx] = matrix_subset[i, rating_idx]
        pickle.dump(train_matrix, open( 'svg/train_matrix.p', 'wb' ))
        pickle.dump(test_matrix, open( 'svg/test_matrix.p', 'wb' ))
    
    plt.subplot(132)
    plt.imshow(test_matrix[:150, :200], aspect='auto', interpolation='nearest')
    plt.tight_layout()
    plt.title("Test  Matrix (subset)")

    plt.subplot(133)
    plt.imshow(train_matrix[:150, :200], aspect='auto', interpolation='nearest')
    plt.tight_layout()
    plt.title("Train matrix (subset)")
    plt.show()
    fig.savefig('../res/img/matrixsubset.png')


    ## sample submission
    submission_ratings = pd.read_csv('../data/data/sampleSubmission.csv', dtype={'Prediction': np.int})
    submission_idx = ratings.Id.str.extract('r([0-9]+)_c([0-9]+)', expand=True)

    user_idx = submission_idx[0].astype(int)
    film_idx = submission_idx[1].astype(int)
    sub_matrix = np.zeros((max(user_idx), max(film_idx)), dtype=np.int)

    if os.path.isfile("svg/sub_matrix.p"):
        sub_matrix = pickle.load(open( 'svg/sub_matrix.p', 'rb' ))
    else:
        for i in range(len(user_idx)): # takes time!
            sub_matrix[user_idx[i]-1, film_idx[i]-1] = submission_ratings.Prediction[i]
        pickle.dump(sub_matrix, open( 'svg/sub_matrix.p', 'wb' ))
        
    plt.figure(figsize = (5,5))
    plt.imshow(sub_matrix[:150, :200], aspect='auto', interpolation='nearest')
    plt.tight_layout()
    plt.title("Sample submission")
    plt.show()

    # data exploration
    ratings_per_movie = (matrix != 0).sum(0)
    ratings_per_user = (matrix != 0).sum(1)


    # number of ratings
    print('Mean number of ratings per movie : %s' % ratings_per_movie.mean())

    plt.style.use('ggplot')
    fig = plt.figure(figsize = (5,5))
    hist = plt.hist(ratings_per_movie, bins = range(min(ratings_per_movie), max(ratings_per_movie) + 250, 250), color = 'b')
    plt.xlabel('Number of ratings', fontsize = 16)
    plt.ylabel('Frequency', fontsize = 16)
    plt.title("Ratings per movie")
    fig.savefig('../res/img/movieshist.png')
    plt.show()

    ###

    N = len(ratings_per_movie)
    Z = ratings_per_movie
    X2 = np.sort(Z)
    F2 = np.array(range(N))/float(N)

    fig = plt.figure(figsize = (8,5))
    h = plt.plot(X2, F2, c='b', linewidth=1.5)
    plt.ylim((0,1.1))
    plt.xlabel('Number of ratings', fontsize = 16)
    plt.ylabel('Probability', fontsize = 16)
    plt.title("Ratings per movie")
    fig.savefig('../res/img/moviescdf.png')
    plt.show()



    # users

    print('Mean number of ratings per user : %s' % ratings_per_user.mean())
    plt.style.use('ggplot')

    fig = plt.figure(figsize = (8,5))
    hist = plt.hist(ratings_per_user, bins = range(min(ratings_per_user), max(ratings_per_user) + 25, 25), color = 'b')
    plt.xlabel('Number of ratings', fontsize = 16)
    plt.ylabel('Frequency', fontsize = 16)
    plt.title("Ratings per user")
    fig.savefig('../res/img/userhist.png')
    plt.show()

    ###

    N = len(ratings_per_user)
    Z = ratings_per_user
    X2 = np.sort(Z)
    F2 = np.array(range(N))/float(N)

    fig = plt.figure(figsize = (8,5))
    h = plt.plot(X2, F2, c='b', linewidth=1.5)
    plt.ylim((0,1.1))
    plt.xlabel('Number of ratings', fontsize = 16)
    plt.ylabel('Probability', fontsize = 16)
    plt.title("Ratings per user")
    fig.savefig('../res/img/usercdf.png')
    plt.show()


    # Mean rating per movie
    mean_per_movie = matrix.sum(0)/(matrix!=0).sum(0)

    ratings_per_movie = (matrix != 0).sum(0)

    #print(np.corrcoef(ratings_per_movie,mean_per_movie))

    fig = plt.figure(figsize = (8,5))
    h = plt.scatter(ratings_per_movie,mean_per_movie, c='b', linewidth=1)

    m, b = np.polyfit(ratings_per_movie,mean_per_movie, 1)
    plt.style.use('ggplot')
    plt.plot(ratings_per_movie, m*ratings_per_movie + b, '-')
    plt.xlim((-200,5000))
    plt.xlabel('Number of ratings per movie', fontsize = 16)
    plt.ylabel('Average rating', fontsize = 16)
    fig.savefig('../res/img/ratingmovielinear.png')
    plt.show()



    n = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i][j] != 0:
                n += matrix[i][j]
                
    avg = n / np.count_nonzero(matrix)

    mean_per_user = matrix.sum(1)/(matrix!=0).sum(1)
    difference_to_mean = (mean_per_user - avg).tolist()

    plt.style.use('ggplot')
    fig = plt.figure(figsize = (8,5))
    hist = plt.hist(difference_to_mean, bins = 30, color = 'b')
    plt.xlabel('Ratings distance to global mean', fontsize = 16)
    plt.ylabel('Number of users', fontsize = 16)
    fig.savefig('../res/img/differencetomean.png')
    plt.show()
        
    
explore()

