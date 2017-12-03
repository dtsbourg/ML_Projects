# Project Recommender System

## Usage

Start by installing the dependencies :

```bash
sudo python3 -m pip install -r requirements.txt
```

You can then declare your model, train and evaluate it :

```bash
python3 neural.py
```  

> TODO : Document the API

## Context

### Class project

> For this choice of project task, you are supposed to predict good recommendations, e.g. of movies to users. We have acquired ratings of 10000 users for 1000 different items (think of movies). All ratings are integer values between 1 and 5 stars. No additional information is available on the movies or users.
>
> #### Evaluation Metric
>
> Your collaborative filtering algorithm is evaluated according to the prediction error, measured by root-mean-squared error (RMSE).
>
> #### Submission system environment setup:
>
> 1. The dataset is available from the Kaggle page, as linked in the PDF project description
> 2. All information of the task and some baselines are provided in Exercise 10:
>
>  [https://github.com/epfml/ML_course/tree/master/labs/ex10]

### Data

Part of the [Netflix Prize](https://www.netflixprize.com/) : user ratings for movies.

**data_train.csv** - the training set. Each entry consists of an ID of the form r3_c6 (meaning row 3 column 6 <=> user 3, movie 6) and the value between 1-5 stars, given by the user for this user/movie combination

## A primer on Recommender systems

#### Types of recommender systems

* **Collaborative filtering**
  * Use existing ratings to generate recommendations
  * we don't know the content of features in advance, so we learn latent features and make predictions on user-to-item ratings at the same time
  * **This is the mode we have for the project**
* Content filtering
  * Use similarities between products and customers
  *  Regression problem : we try to make a user-to-item rating prediction using the content of items as features.
* Hybrid
  * use both methods

### Matrix Completion

Recommender systems are formally *Matrix completion* problems : columns and rows are users and items, with values being the score (the rating). Given a small subset of the elements in the matrix the goal is to compute the missing values.

## Proposed approach

The proposed project structure would compare several methods of collaborative filtering. Each will have their particularities we can study so we can also choose an informative subset.

### Recommender system

Below are some of the ways we can solve the problem with a non-exhaustive list of properties we can compare / study.

1. [DONE [@dtsbourg](https://github.com/dtsbourg)] Similarity measures

   * Compute similarity scores (e.g. cosine similarity) based on the ratings, for items and users
   * Use the similarity score to predict a rating

2. [[@Mgt-A](https://github.com/Mgt-A)] Matrix Factorization

   * The goal is to factorize the (user, item, rating) MxN matrix into two matrices U (size LxM) and I (size NxL) where L is a parameter of the model : it represents the dimension of the latent space (basically we are embedding the users and items into a higher dimensional space, representing features that will help the prediction)
   * The latent spaces are computed by minimizing a reconstruction loss function, which can be augmented to include more precise behaviors (e.g. regularization on the latent spaces, …)
   * The loss minimization process can be run with any EM method. Most common : Alternating Least Squares (ALS), SGD, BPR, SVD ...
   * The prediction is the obtained by multipling the latent space representations : R(predicted) = U . I

3. Ensemble methods

   * Instead of using linear combinations of factors to predict the rating, we can train a decision tree to use different features to decide what the final score should be. This allows to take into account lots of heuristic features (Number of movies each user rated, Number of users that rated each movie, Factor vectors of users and movies, Hidden units of a restricted Boltzmann Machine, ...)

4. [IN PROGRESS [@dtsbourg](https://github.com/dtsbourg)] Deep Recommender Systems

   * use a convolutional Neural Network to build the representations of the user, item interactions. These features are then fed to a fully connected layer, with softmax activation to get the confidence for the 5 classes, one for each rating. These are then combined to get the predicted score (MLP, weighted sum which includes confidence, ...)
   * Note that additional features can be added as inputs to the Neural Net (e.g. user embeddings computed from other methods, ...)

5. Autoencoders

   * Using autoencoders as a method to reconstruct a dense version of the sparse output


   * Using an autoencoder to build a latent space, then feed that to a regression algorithm (e.g. NN)

6. Lots of other methods to play around with ...

   1. RBM
   2. MultiGraph CNNs
   3. RNN / LSTM
   4. GANs
   5. ...

### Data exploration

A list of things to do in order to get a better idea of what the data is like and characterize our methods.

* User + Item distributions
  * How many movies do users rate ?
  * What are the most rated movies ?
  * Mean scores ?
  * Show example rating behavior for a user
* User + Item similarity
  * Show the similarity scores (can be shown as a graph, with e.g. KNN)
* Plotting latent spaces
  * Compare [t-SNE](https://distill.pub/2016/misread-tsne/) , PCA, and the new [UMAP](https://github.com/lmcinnes/umap)

### Misc

* A common method to compute the test set is to use *leave k-out* where we randomly remove k ratings for each user
