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
parser.add_argument('-t','--train', action='store_true', help='Run training.')
parser.add_argument('-s','--setup', action='store_true', help='Run the setup pipeline (building embeddings, ...).')
parser.add_argument('-p','--predict', action='store_true', help='Run the submission.')

if __name__ == '__main__':
    args = vars(parser.parse_args())
    if not (args.train or args.setup or args.predict):
        parser.error('No action requested, add --train or --setup or --predict')
    if int(args.train)+int(args.setup)+int(args.predict) > 1:
        parser.error('Please call this module sequentially, selecting only one step at a time (--setup, --train or --predict).')
    deep_regularized_feat_net_pipeline(train=args.train, predict=args.predict, setup=args.setup)
