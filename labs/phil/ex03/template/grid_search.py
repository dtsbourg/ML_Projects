# -*- coding: utf-8 -*-
""" Grid Search"""

import numpy as np
from costs import *


def generate_w(num_intervals):
    """Generate a grid of values for w0 and w1."""
    w0 = np.linspace(-100, 200, num_intervals)
    w1 = np.linspace(-150, 150, num_intervals)
    return w0, w1


def get_best_parameters(w0, w1, losses):
    """Get the best w from the result of grid search."""
    min_row, min_col = np.unravel_index(np.argmin(losses), losses.shape)
    return losses[min_row, min_col], w0[min_row], w1[min_col]


# ***************************************************
# INSERT YOUR CODE HERE
# TODO: Paste your implementation of grid_search
#       here when it is done.
# ***************************************************

def grid_search_part(y, tx, w0, w1):
    """Algorithm for grid search."""
    losses = np.zeros((len(w0), len(w1)))
    for i in range(0,len(w0)):         # !! range(x,y) = [x ... y-1]
        for j in range(0,len(w1)):
            losses[i][j] = compute_loss(y,tx,[w0[i], w1[j]]) 
    return losses
    
def grid_search(y, tx):
    grid_w0, grid_w1 = generate_w(num_intervals=50)
    grid_losses = grid_search_part(y, tx, grid_w0, grid_w1)
    # Select the best combinaison
    loss_star, w0_star, w1_star = get_best_parameters(grid_w0, grid_w1, grid_losses)
    return [w0_star, w1_star], loss_star
    
    
    