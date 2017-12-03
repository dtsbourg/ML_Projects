from keras.utils import plot_model
from keras import models
import numpy as np
from sklearn import manifold

def save_model_graph(model):
    plot_model(model, to_file='../res/model/' + model.descr + '.png')

def save_model(cm):
    try:
        cm.model.save('../res/model/'+cm.descr+'.h5')
    except Exception as e:
        raise

def load_model(path):
    return models.load_model(path)

def build_interaction_matrix(users, items, ratings, suffix=''):
    matrix = np.zeros((max(users), max(items)), dtype=np.int)

    for i in range(len(users)):
        matrix[users[i]-1, items[i]-1] = ratings[i]
    np.save('../data/embeddings/interaction_matrix'+suffix, matrix)

def build_embedding(path, embedding=None):
    if embedding == 't-SNE':
        mat = np.load(path)
        tsne = manifold.TSNE(n_components=3, random_state=0)
        u_tsne = tsne.fit_transform(mat)
        i_tsne = tsne.fit_transform(mat.T)
        return u_tsne, i_tsne

def build_tSNE_embedding(path, suffix="", save=True):
    u_e, i_e = build_embedding(path=path, embedding='t-SNE')
    if save is True:
        np.save('../data/embeddings/users_t_sne'+suffix, u_e)
        np.save('../data/embeddings/items_t_sne'+suffix, i_e)
    return u_e, i_e
