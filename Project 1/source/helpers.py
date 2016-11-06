# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np


def standardize_and_add_ones(x, mean_x=None, std_x=None):
    """
    Standardize the original data set.
    """
    if mean_x is None:
        mean_x = np.mean(x, axis=0)
    tx = x - mean_x
    if std_x is None:
        std_x = np.std(x, axis=0)
    tx[:, std_x > 0] = tx[:, std_x > 0] / std_x[std_x > 0]
    
    tx = np.hstack((np.ones((x.shape[0],1)), tx))
    return tx, mean_x, std_x


def _standardize(x, mean_x=None, std_x=None):
    """
    Standardize the original data set.
    """
    if mean_x is None:
        mean_x = np.mean(x, axis=0)
    tx = x - mean_x
    if std_x is None:
        std_x = np.std(x, axis=0)
    tx[:, std_x > 0] = tx[:, std_x > 0] / std_x[std_x > 0]

    return tx, mean_x, std_x
