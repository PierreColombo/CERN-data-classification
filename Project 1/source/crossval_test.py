# -*- coding: utf-8 -*-
"""
Project 1
group #28

pierre.colombo@epfl.ch
christian.tresch@epfl.ch
juraj.korcek@epfl.ch
"""
import os
import numpy as np

from proj1_helpers import load_csv_data
from cross_validation import optimize_model
#from tests.test_helpers import load_data
from helpers import _standardize

# Resets cpu core task affinity
os.system("taskset -p 0xff %d" % os.getpid())

# ## Load the training data into feature matrix, class labels, and event ids:
DATA_TRAIN_PATH = '../data/train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

# # ## Our code
# # split categorical variable (23)
# tX = np.vstack((tX.T, (tX[:, 22] == 0).astype(int))).T
# tX = np.vstack((tX.T, (tX[:, 22] == 1).astype(int))).T
# tX = np.vstack((tX.T, (tX[:, 22] == 2).astype(int))).T
# tX = np.vstack((tX.T, (tX[:, 22] == 3).astype(int))).T
# tX.shape
#
#
# model1_datapoints_no = tX.shape[0] - sum(tX[:, 4] == -999)
# tX1 = tX[tX[:, 4] != -999]
# tX1 = np.delete(tX1, 0, axis=1)
# y1 = y[tX[:, 4] != -999]
# (tX1.shape, y1.shape)
#
#
# model2_datapoints_no = sum(tX[:, 4] == -999) - sum(tX[:, 23] == -999)
# tX2 = tX[tX[:, 4] == -999]
# ind_to_rem = tX2[:, 23] != -999
# tX2 = tX2[ind_to_rem]
# tX2 = np.delete(tX2, [0, 4, 5, 6, 12, 26, 27, 28], axis=1)
# y2 = y[tX[:, 4] == -999]
# y2 = y2[ind_to_rem]
# (tX2.shape, y2.shape)
#
#
# model3_datapoints_no = sum(tX[:, 23] == -999)
# tX3 = tX[tX[:, 23] == -999]
# tX3= np.delete(tX3, [0, 4, 5, 6, 12, 23, 24, 25, 26, 27, 28], axis=1)
# y3 = y[tX[:, 23] == -999]
# (tX3.shape, y3.shape)
#
#
# model4_datapoints_no = sum(tX[:, 0] != -999)
# tX4 = tX[tX[:, 0] != -999, 0]
# y4 = y[tX[:, 0] != -999]
# (tX4.shape, y4.shape)
#
#
# # our baseline model: all y set to background(no Higgs; -1)
# sum((y == -1).astype(int)) / y.size


# model5_datapoints_no = tX.shape[0]
tX5 = tX[:, 0: 2]
tX5[tX5 == -999] = 0
tX5, mean_x, std_x = _standardize(tX5)
y5 = y
y5[y5 == -1] = 0
#tX5 = build_poly(tX5, 3)
#tX5 = np.insert(tX5, 0, np.ones(tX5.shape[0]), axis=1)
print(tX5.shape, y5.shape)

#Test dataset Stanford
y0 = np.genfromtxt('tests/reg_logistic_regression_data.txt', delimiter=",", usecols=-1)
tX0 = np.genfromtxt('tests/reg_logistic_regression_data.txt', delimiter=",")
tX0 = tX0[:, 0: -1]
# tx, mean_x, std_x = standardize(tx)
#tX0 = np.insert(tX0, 0, np.ones(y0.size), axis=1)

#Test dataset PCMLEx3
#x, y = load_data("./tests/dataEx3.csv")
#print(x.shape)
#print(y.shape)

# # performance of model 5 on train dataset
# y5_pred = ((tX5[:, 0] > 105) & (tX5[:, 0] < 170) & (tX5[:, 1] < 55)).astype(int)
# y5_pred[y5_pred == 0] = -1
# 1 - sum(abs(y5 - y5_pred) / 2) / y5.size
#
#
# DATA_TEST_PATH = '../data/test.csv'
# _, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
#
# OUTPUT_PATH = '../to_submit/prediction.csv'
# tX5_test = tX_test[:, 0: 2]
# y5_test_pred = ((tX5_test[:, 0] > 105) & (tX5_test[:, 0] < 170) & (tX5_test[:, 1] < 55)).astype(int)
# y5_test_pred[y5_test_pred == 0] = -1
# create_csv_submission(ids_test, y5_test_pred, OUTPUT_PATH)



# Ex4 Ridge test --- only works with ridge regression!
#best_degree, best_lambda = optimize_model(y, x, 1, 11, np.logspace(-4, 2, 30), 4, 600, 10**-3)

# Standford test
#best_degree, best_lambda = optimize_model(y0, tX0, 1, 3, np.logspace(-4, 2, 10), 4, 1700, 10**-2)

# Higgs dataset test
best_degree, best_lambda = optimize_model(y5, tX5, 1, 7, np.logspace(-5, 2, 10), 4, 300, 10**-3)


# Direct crossval function test
#indices = build_k_indices(y5, 4, 1)
#w = cross_validation(y5, tX5, indices, 0, 0, 3, 500, 10**-3)
# w = np.array([-0.00353853, -0.04002332, -0.14212216])
#y5_pred = np.round(sigmoid(np.dot(tX5, w)))
#res = np.mean(np.absolute(y5[:1000] - y5_pred[:1000]))
#print(res)