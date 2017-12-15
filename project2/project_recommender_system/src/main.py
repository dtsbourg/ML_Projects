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
parser.add_argument('-p','--predict', dest='mode', action='store_const', help='Run the submission.', const='setup')

def get_mode(args):
    train   = args['mode']=='train'
    predict = args['mode']=='setup'
    return train, predict

if __name__ == '__main__':
    args = vars(parser.parse_args())
    train, predict = get_mode(args)
    pipeline.deep_net_pipeline(train=train, predict=predict)
