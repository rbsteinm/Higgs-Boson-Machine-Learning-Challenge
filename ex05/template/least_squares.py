# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """calculate the least squares solution."""
    opt_w = np.linalg.inv(tx.T.dot(tx)).dot(tx.T).dot(y)
    #computes the loss using MSE
    mse = compute_mse(y, tx, opt_w)
    # returns mse, and optimal weights
    return mse, opt_w

def compute_mse(y, tx, beta):
    """compute the loss by mse."""
    e = y - tx.dot(beta)
    return 1/2*np.mean(e**2)
