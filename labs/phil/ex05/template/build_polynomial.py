# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    if(degree < 1):
        return x
    
    new_x = np.zeros(np.concatenate([np.array(x.shape), [degree+1]]))
    new_x[..., 0] = x**0 # add zeros (for degree 0)
    new_x[..., 1] = x**1 # add zeros (for degree 0)

    if degree <= 1:
        return new_x
    else:  
        for i in range(degree-1):
            cur_degree = i+2
            new_x[...,cur_degree] = x**cur_degree

    return new_x
