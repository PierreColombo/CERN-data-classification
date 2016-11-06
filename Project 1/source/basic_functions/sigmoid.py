# -*- coding: utf-8 -*-
"""
Project 1
group #28

pierre.colombo@epfl.ch
christian.tresch@epfl.ch
juraj.korcek@epfl.ch
"""
import numpy as np


def sigmoid(t):
    """
    Applies sigmoid function on t.
    :param t: tX.T * w
    :return: sigma(t)
    """
    t = np.asarray(t)
    t[t < -500] = -500
    return 1 / (1 + np.exp(-t))
