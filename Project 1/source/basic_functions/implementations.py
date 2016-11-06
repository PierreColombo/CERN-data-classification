# -*- coding: utf-8 -*-
"""
Project 1
group #28

pierre.colombo@epfl.ch
christian.tresch@epfl.ch
juraj.korcek@epfl.ch
"""
import math

import numpy as np

from basic_functions.sigmoid import sigmoid



def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent
    :param y: output (N, 1)
    :param tx: input variables (N, D)
    :param gamma: step size for gradient descent
    :param max_iters: maximum number of iterations for gradient descent
    :return: weights vector w (D, 1)
    """
    n = tx.shape[0]
    d = tx.shape[1]
    w = initial_w

    for i in range(max_iters):
        e = y - np.dot(tx, w)
        w += gamma / math.sqrt(i + 1) / n * np.dot(tx.T, e)

    return w


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using stochastic gradient descent
    :param y: output (N, 1)
    :param tx: input variables (N, D)
    :param gamma: step size for gradient descent
    :param max_iters: maximum number of iterations for gradient descent
    :return: weights vector w (D, 1)
    """
    n = tx.shape[0]
    d = tx.shape[1]
    w = initial_w
    batch_size = round(math.sqrt(n)) * 2

    for i in range(max_iters):
        ind = np.random.permutation(n)[0: batch_size]
        e = y[ind] - np.dot(tx[ind, :], w)
        w += gamma / math.sqrt(i + 1) / batch_size * np.dot(tx[ind, :].T, e)

    return w


def least_squares(y, tx):
    """
    Least squares regression using normal equations
    :param y: output (N, 1)
    :param tx: input variables (N, D)
    :return: weights vector w (D, 1)
    """
    A = tx.T.dot(tx)
    b = tx.T.dot(y)

    return np.linalg.solve(A, b)



def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations
    :param y: output (N, 1)l
    :param tx: input variables (N, D)
    :param lambda_: regularization parameter
    :return: weights vector w (D, 1)
    """
    n = tx.shape[0]
    d = tx.shape[1]

    A = tx.T.dot(tx) + lambda_ * np.identity(d)
    b = tx.T.dot(y)

    return np.linalg.solve(A, b)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent
    :param y: output (N, 1)
    :param tx: input variables (N, D)
    :param gamma: step size for gradient descent
    :param max_iters: maximum number of iterations for gradient descent
    :return: weights vector w (D, 1)
    """
    n = tx.shape[0]
    d = tx.shape[1]
    w = initial_w

    for i in range(max_iters):
        e = sigmoid(np.dot(tx, w)) - y
        w -= gamma / math.sqrt(i + 1) * np.dot(tx.T, e)

    return w


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, gd_type='b'):
    """
    Regularized logistic regression using gradient descent or SGD
    :param y: output (N, 1)
    :param tx: input variables (N, D)
    :param lambda_: regularization parameter
    :param gamma: step size for gradient descent
    :param max_iters: maximum number of iterations for gradient descent
    :return: weights vector w (D, 1)
    """
    n = tx.shape[0]
    d = tx.shape[1]
    w = initial_w

    for i in range(max_iters):
        w -= gamma / math.sqrt(i + 1) * calculate_gradient(y, tx, w, lambda_, gd_type)

    return w


def calculate_gradient(y, tx, w, lambda_, gd_type='b'):
    """
    Calculates gradient for regularized logistic regression
    :param y: output (N, 1)
    :param tx: input variables (N, D)
    :param w: weights vector w
    :param lambda_:
    :return: gradient vector (D, 1)
    """
    n = tx.shape[0]
    d = tx.shape[1]

    if gd_type == 'b':
        batch_size = n
    elif gd_type == 'mb':
        batch_size = round(math.sqrt(n)) * 2
    elif gd_type == 's':
        batch_size = 1
    else:
        raise Exception('Unknown gradient descent type!')

    ind = np.random.permutation(n)[0: batch_size]
    e = sigmoid(np.dot(tx[ind, :], w)) - y[ind]
    lambda_vect = np.insert(lambda_ * np.ones(w.size - 1), 0, 0)

    return 1 / n * (np.dot(tx[ind, :].T, e) + w * lambda_vect)
