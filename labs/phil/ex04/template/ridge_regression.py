# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    lambdap = lambda_ * 2 * y.shape[0]
    a = tx.T.dot(tx) + lambdap*np.eye(tx.shape[1])
    b = tx.T.dot(y)
    return np.linalg.solve(a, b)
