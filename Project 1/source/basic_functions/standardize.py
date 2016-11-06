# -*- coding: utf-8 -*-
"""
Project 1
group #28

pierre.colombo@epfl.ch
christian.tresch@epfl.ch
juraj.korcek@epfl.ch
"""
import numpy as np


def standardize(x, mean_x=None, std_x=None, skipped_cols=list()):
    """
    Standardize the original data set except for columns listed in skipped_cols parameter.
    Used to skip standardizing of categorical variable.
    :param x: input variables (N, D)
    :param mean_x: mean to be subtracted; if not provided it is calculated on x
    :param std_x: std to be devided by; if not provided it is calculated on x
    :param skipped_cols: columns to be skipped in standardization (e.g. for categorical variable)
    :return: standardized x, mean used for standardization, std used for standardization
    """
    mask = np.ones(x.shape[1], dtype=bool)
    mask[skipped_cols] = False

    if mean_x is None:
        mean_x = np.mean(x, axis=0)
        mean_x[~mask] = 0
    tx = x - mean_x
    if std_x is None:
        std_x = np.std(x, axis=0)
        std_x[~mask] = 1
    tx[:, std_x > 0] = tx[:, std_x > 0] / std_x[std_x > 0]

    return tx, mean_x, std_x
