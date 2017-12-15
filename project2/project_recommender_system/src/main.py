"""
CS-433 : Machine Learning
Project 2 -- Recommender Systems

Team :
* Dylan Bourgeois
* Antoine Mougeot
* Philippe Verbist

---

main.py : main interface.

Here you should call one of the available or custom pipelines.
"""

import pipeline
import argparse

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-t','--train', dest='mode', action='store_const', help='Run training.', const='train')
parser.add_argument('-s','--setup', dest='mode', action='store_const', help='Run the setup pipeline (building embeddings, ...).', const='setup')
parser.add_argument('-p','--predict', dest='mode', action='store_const', help='Run the submission.', const='setup')

def get_mode(args):
    setup   = args['mode']=='setup'
    train   = args['mode']=='train'
    predict = args['mode']=='setup'
    return setup, train, predict

if __name__ == '__main__':
    args = vars(parser.parse_args())
    setup, train, predict = get_mode(args)
    pipeline.deep_regularized_feat_net_pipeline(setup=setup, train=train, predict=predict)
