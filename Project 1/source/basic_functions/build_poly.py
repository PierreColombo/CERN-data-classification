# -*- coding: utf-8 -*-
"""
Project 1
group #28

pierre.colombo@epfl.ch
christian.tresch@epfl.ch
juraj.korcek@epfl.ch
"""
import numpy as np


def build_poly(x, degree, include_cross_terms=False):
    if (degree == 1 ) :
        return np.insert(x, 0, np.ones(x.shape[0]), axis=1)
    mat_degree = x

    for i in range(x.shape[1]) :
        mat_zero = np.hstack((x[:,i][:,None], x[:,i+1:x.shape[1]]))
        mat_A = np.multiply(x[:,i][:,None], mat_zero)
        mat_degree =  np.concatenate((mat_degree, mat_A),axis = 1)


    return np.concatenate((mat_degree, np.ones(x.shape[0])[:,None]),axis = 1)
