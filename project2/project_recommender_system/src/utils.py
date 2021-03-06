"""
CS-433 : Machine Learning
Project 2 -- Recommender Systems

Team :
* Dylan Bourgeois
* Antoine Mougeot
* Philippe Verbist

---

utils.py : Defining several useful utilities.

Includes functions to manipulate the model (loading and saving pre-trained models),
plotting the model's performance, building embeddings.
"""
import warnings
from keras.utils import plot_model
from keras   import models
from sklearn import manifold
from sklearn import decomposition
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def save_model_graph(model):
    """
    Save the model architecture.
    """
    plot_model(model, to_file='../res/model/' + model.descr + '.png')

def save_model(cm):
    """
    Save the model to binary for later loading.
    """
    try:
        path = '../res/model/'+cm.descr
        cm.model.save(path+'.h5')
        return path
    except Exception as e:
        raise

def load_model(path):
    """
    Load a saved model.
    """
    return models.load_model(path)

def plot(description, history, show=False):
    """
    Plot the loss of a training procedure. The figure can be shown
    or saved to an image.
    """
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    if show is False:
        plt.savefig('../res/img/' + description + '.png')

def build_interaction_matrix(users, items, ratings, suffix=''):
    """
    Build user/movie interaction matrix, a nb(users)xnb(items) matrix,
    where each entry is the rating user u gave item i.
    """
    matrix = np.zeros((max(users), max(items)), dtype=np.int)

    for i in range(len(users)):
        matrix[users[i]-1, items[i]-1] = ratings[i]
    path_str = '../data/embeddings/interaction_matrix'+suffix
    np.save(path_str, matrix)
    return path_str

def build_embedding(path, embedding=None):
    """
    Build the desired embedding and save to specified path.

    Available embeddings :
        * Interaction Matrix
        * t-SNE (unused)
        * Spectral Embedding
        * Locally Linear Embedding
        * Non-negative Matrix Factorisation
        * Factor Analysis
    """
    if embedding == 'spectral':
        mat = np.load(path)
        u_spectral = manifold.SpectralEmbedding(n_components=64, random_state=0, n_jobs=8).fit_transform(mat)
        i_spectral = manifold.SpectralEmbedding(n_components=64, random_state=0, n_jobs=8).fit_transform(mat.T)
        return u_spectral, i_spectral
    elif embedding == 'lle':
        mat = np.load(path)
        u_lle = manifold.LocallyLinearEmbedding(n_components=64, random_state=0, n_jobs=8).fit_transform(mat)
        i_lle = manifold.LocallyLinearEmbedding(n_components=64, random_state=0, n_jobs=8).fit_transform(mat.T)
        return u_lle, i_lle
    elif embedding == 'fa':
        mat = np.load(path)
        u_fa = decomposition.FactorAnalysis(n_components=64, random_state=0).fit_transform(mat)
        i_fa = decomposition.FactorAnalysis(n_components=64, random_state=0).fit_transform(mat.T)
        return u_fa, i_fa
    elif embedding == 'nmf':
        mat = np.load(path)
        u_nmf = decomposition.NMF(n_components=64, random_state=0).fit_transform(mat)
        i_nmf = decomposition.NMF(n_components=64, random_state=0).fit_transform(mat.T)
        return u_nmf, i_nmf

def build_tSNE_embedding(path, suffix="", save=True):
    warnings.warn("deprecated", DeprecationWarning)
    print("t-SNE is not used in our models anymore. Make you are calling this method purposefully.")
    u_e, i_e = build_embedding(path=path, embedding='t-SNE')
    if save is True:
        np.save('../data/embeddings/users_t_sne_'+suffix, u_e)
        np.save('../data/embeddings/items_t_sne_'+suffix, i_e)
    return u_e, i_e

def build_spectral_embedding(path, suffix="", save=True):
    u_e, i_e = build_embedding(path=path, embedding='spectral')
    if save is True:
        np.save('../data/embeddings/users_spectral_'+suffix, u_e)
        np.save('../data/embeddings/items_spectral_'+suffix, i_e)
    return u_e, i_e

def build_lle_embedding(path, suffix="", save=True):
    u_e, i_e = build_embedding(path=path, embedding='lle')
    if save is True:
        np.save('../data/embeddings/users_lle_'+suffix, u_e)
        np.save('../data/embeddings/items_lle_'+suffix, i_e)
    return u_e, i_e

def build_fa_embedding(path, suffix="", save=True):
    u_e, i_e = build_embedding(path=path, embedding='fa')
    if save is True:
        np.save('../data/embeddings/users_fa_'+suffix, u_e)
        np.save('../data/embeddings/items_fa_'+suffix, i_e)
    return u_e, i_e

def build_nmf_embedding(path, suffix="", save=True):
    u_e, i_e = build_embedding(path=path, embedding='nmf')
    if save is True:
        np.save('../data/embeddings/users_nmf_'+suffix, u_e)
        np.save('../data/embeddings/items_nmf_'+suffix, i_e)
    return u_e, i_e

def load_full_embedding(data, suffix='64'):
    """
    Load the full pre-computed embeddings.
    """
    item_idx = data[0]; user_idx = data[1]
    data += load_embedding(path='../data/embeddings/items_spectral_'+suffix+'.npy', idx=item_idx)
    data += load_embedding(path='../data/embeddings/users_spectral_'+suffix+'.npy', idx=user_idx)
    data += load_embedding(path='../data/embeddings/items_lle_'+suffix+'.npy',      idx=item_idx)
    data += load_embedding(path='../data/embeddings/users_lle_'+suffix+'.npy',      idx=user_idx)
    data += load_embedding(path='../data/embeddings/items_fa_'+suffix+'.npy',       idx=item_idx)
    data += load_embedding(path='../data/embeddings/users_fa_'+suffix+'.npy',       idx=user_idx)
    data += load_embedding(path='../data/embeddings/items_nmf_'+suffix+'.npy',      idx=item_idx)
    data += load_embedding(path='../data/embeddings/users_nmf_'+suffix+'.npy',      idx=user_idx)
    return data

def load_embedding(path, idx):
    """
    Load a precomputed embedding, returning only the desired subset.
    """
    emb = np.load(path)
    t_emb = emb[idx - 1]
    return [t_emb]
