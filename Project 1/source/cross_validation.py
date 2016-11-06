# -*- coding: utf-8 -*-
"""
Project 1
group #28

pierre.colombo@epfl.ch
christian.tresch@epfl.ch
juraj.korcek@epfl.ch
"""

import numpy as np
from basic_functions.build_poly import build_poly
from basic_functions.implementations import reg_logistic_regression, calculate_gradient
from basic_functions.cost import calculate_loss
from plots import cross_validation_visualization


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, x, k_indices, k, lambda_, degree, max_iter, gamma):
    """return the loss of ridge regression."""
    tr_indices = np.concatenate([k_indices[i] for i in range(k_indices.shape[0]) if i != k])
    te_indices = k_indices[k]

    x_tr = x[tr_indices]
    x_te = x[te_indices]
    y_tr = y[tr_indices]
    y_te = y[te_indices]

    tx_tr_poly = build_poly(x_tr, degree)
    tx_te_poly = build_poly(x_te, degree)

    w = reg_logistic_regression(y_tr, tx_tr_poly, lambda_, max_iter, gamma)

    loss_tr = calculate_loss(y_tr, tx_tr_poly, w, lambda_)
    loss_te = calculate_loss(y_te, tx_te_poly, w, lambda_)

    print("Degree: ", degree, " lambda: ", lambda_, " loss_tr: ", loss_tr)
    print("Degree: ", degree, " lambda: ", lambda_, " loss_te: ", loss_te)

    return loss_tr, loss_te, w

def optimize_model(y, x, degree_min, degree_max, lambdas=np.logspace(-4, 2, 30), k_fold=4, max_iter=200, alpha=10 ** -6):
    # lambda_ is penalization for the norm of w
    # max_iter is for the number of iteration when we compute the gradiernt
    # alpha is the size of the step
    seed = 1
    k_fold_multiplier = 1  # -3*k_fold+35 #rule of thumb according to stackexchange
    deg_range = range(degree_min, degree_max + 1)
    print("Number of degrees tested: ", len(deg_range))
    print("Number of lambdas tested: ", len(lambdas))
    print("Number of lambdas tested: ", lambdas)

    min_lambdas = []
    print("List of best lambda per degree tuples (degree, lambda, RMSE, var):")
    for degree in deg_range:
        # define lists to store the loss of training data and test data
        rmse_tr = []
        rmse_te = []
        print("### DEGREE: ", degree)
        for l_idx, lambda_ in enumerate(lambdas):
            rmse_tr_lamb = []
            rmse_te_lamb = []
            for km in range(k_fold_multiplier):
                # get new splits for every iteration int the k fold process
                k_indices = build_k_indices(y, k_fold, seed)
                for k_idx in range(k_fold):
                    mse_tr, mse_te, _ = cross_validation(y, x, k_indices, k_idx, lambda_, degree, max_iter, alpha)
                    rmse_tr_lamb.append(np.sqrt(2 * mse_tr))
                    rmse_te_lamb.append(np.sqrt(2 * mse_te))
            rmse_tr.append(np.mean(rmse_tr_lamb))
            rmse_te.append(np.mean(rmse_te_lamb))
            print("# Degree: ", degree, " lambda: ", lambda_, " mean tr: ", np.mean(rmse_tr_lamb))
            print("# Degree: ", degree, " lambda: ", lambda_, " mean te: ", np.mean(rmse_te_lamb))
        lamb_tuple = (degree, lambdas[np.argmin(rmse_te)], min(rmse_te), np.var(rmse_te_lamb))
        min_lambdas.append(lamb_tuple)
        print(lamb_tuple)
        cross_validation_visualization(degree, lambda_, lambdas, rmse_tr, rmse_te)
    best_rmse = min(min_lambdas, key=lambda t: t[2])
    best_degree = best_rmse[0]
    best_lambda = best_rmse[1]
    print("Best degree: ", best_degree)
    print("Best lambda: ", best_lambda)
    print("Best RMSE: ", best_rmse[2])
    print("Confidence variance: ", best_rmse[3])
    return best_degree, best_lambda