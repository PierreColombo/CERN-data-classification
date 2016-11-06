# -*- coding: utf-8 -*-
"""
Project 1
group #28

pierre.colombo@epfl.ch
christian.tresch@epfl.ch
juraj.korcek@epfl.ch
"""
import numpy as np

from models.model_class import Model
from basic_functions.implementations import reg_logistic_regression
from basic_functions.standardize import standardize
from basic_functions.build_poly import build_poly
from basic_functions.sigmoid import sigmoid


class ComplexModel(Model):

    def _prepare_x(self, x, mean, std):

        if self._x_mean is None or self._x_std is None:
            self._x_mean = (None, None, None, None, None, None)
            self._x_std = (None, None, None, None, None, None)

        ind1 = np.logical_and.reduce((x[:, 0] != -999, x[:, 4] != -999, x[:, 23] != -999))
        tx1 = x[ind1, :]
        tx1, tx1_mean, tx1_std = standardize(tx1, self._x_mean[0], self._x_std[0], skipped_cols=[22, 30, 31])
        ind2 = np.logical_and.reduce((x[:, 0] == -999, x[:, 4] != -999, x[:, 23] != -999))
        tx2 = x[ind2, :]
        tx2 = np.delete(tx2, 0, axis=1)
        tx2, tx2_mean, tx2_std = standardize(tx2, self._x_mean[1], self._x_std[1], skipped_cols=[21, 29, 30])
        ind3 = np.logical_and.reduce((x[:, 0] != -999, x[:, 4] == -999, x[:, 23] != -999))
        tx3 = x[ind3, :]
        tx3 = np.delete(tx3, [4, 5, 6, 12, 26, 27, 28], axis=1)
        tx3, tx3_mean, tx3_std = standardize(tx3, self._x_mean[2], self._x_std[2], skipped_cols=[18, 23, 24])
        ind4 = np.logical_and.reduce((x[:, 0] == -999, x[:, 4] == -999, x[:, 23] != -999))
        tx4 = x[ind4, :]
        tx4 = np.delete(tx4, [0, 4, 5, 6, 12, 26, 27, 28], axis=1)
        tx4, tx4_mean, tx4_std = standardize(tx4, self._x_mean[3], self._x_std[3], skipped_cols=[17, 22, 23])
        ind5 = np.logical_and.reduce((x[:, 0] != -999, x[:, 4] == -999, x[:, 23] == -999))
        tx5 = x[ind5, :]
        tx5 = np.delete(tx5, [4, 5, 6, 12, 23, 24, 25, 26, 27, 28, 29], axis=1)
        tx5, tx5_mean, tx5_std = standardize(tx5, self._x_mean[4], self._x_std[4], skipped_cols=[18, 19, 20])
        ind6 = np.logical_and.reduce((x[:, 0] == -999, x[:, 4] == -999, x[:, 23] == -999))
        tx6 = x[ind6, :]
        tx6 = np.delete(tx6, [0, 4, 5, 6, 12, 23, 24, 25, 26, 27, 28, 29], axis=1)
        tx6, tx6_mean, tx6_std = standardize(tx6, self._x_mean[5], self._x_std[5], skipped_cols=[17, 18, 19])

        # Standardize
        self._x_mean = (tx1_mean, tx2_mean, tx3_mean, tx4_mean, tx5_mean, tx6_mean)
        self._x_std = (tx1_std, tx2_std, tx3_std, tx4_std, tx5_std, tx6_std)

        orig_n = x.shape[0]
        return {'tX': (tx1, tx2, tx3, tx4, tx5, tx6), 'ind': (ind1, ind2, ind3, ind4, ind5, ind6), 'orig_n': orig_n}

    def _prepare_y(self, y, x):
        y[y == -1] = 0
        y1 = y[x['ind'][0]]
        y2 = y[x['ind'][1]]
        y3 = y[x['ind'][2]]
        y4 = y[x['ind'][3]]
        y5 = y[x['ind'][4]]
        y6 = y[x['ind'][5]]

        return y1, y2, y3, y4, y5, y6

    def _train_model(self):
        best_degree1, best_lambda1 = 2, 0 #optimize_model_ref(self._y[0], self._tX['tX'][0], 1, 3, lambdas=np.logspace(-5, 4, 5), k_fold=10, max_iter=5000, alpha=None)
        tx1 = build_poly(self._tX['tX'][0], best_degree1)
        d = tx1.shape[1]
        w = np.zeros(d)
        w1 = reg_logistic_regression(self._y[0], tx1, best_lambda1, w, 1000, 3)
        print('Model1 accuracy: ' + str(1 - sum(abs(self._y[0] - np.round(sigmoid(np.dot(tx1, w1))))) / tx1.shape[0]))

        best_degree2, best_lambda2 = 2, 0 #optimize_model_ref(self._y[1], self._tX['tX'][1], 1, 3, lambdas=np.logspace(-5, 4, 5), k_fold=10, max_iter=5000, alpha=None)
        tx2 = build_poly(self._tX['tX'][1], best_degree2)
        d = tx2.shape[1]
        w2 = reg_logistic_regression(self._y[1], tx2, best_lambda2, w, 1000, 3)
        print('Model2 accuracy: ' + str(1 - sum(abs(self._y[1] - np.round(sigmoid(np.dot(tx2, w2))))) / tx2.shape[0]))

        best_degree3, best_lambda3 = 2, 0 #optimize_model_ref(self._y[2], self._tX['tX'][2], 1, 3, lambdas=np.logspace(-5, 4, 5), k_fold=10, max_iter=5000, alpha=None)
        tx3 = build_poly(self._tX['tX'][2], best_degree3)
        d = tx3.shape[1]
        w3 = reg_logistic_regression(self._y[2], tx3, best_lambda3, w, 1000, 3)
        print('Model3 accuracy: ' + str(1 - sum(abs(self._y[2] - np.round(sigmoid(np.dot(tx3, w3))))) / tx3.shape[0]))

        best_degree4, best_lambda4 = 2, 0 #optimize_model_ref(self._y[3], self._tX['tX'][3], 1, 3, lambdas=np.logspace(-5, 4, 5), k_fold=10, max_iter=5000, alpha=None)
        tx4 = build_poly(self._tX['tX'][3], best_degree4)
        d = tx4.shape[1]
        w4 = reg_logistic_regression(self._y[3], tx4, best_lambda4, w, 1000, 3)
        print('Model4 accuracy: ' + str(1 - sum(abs(self._y[3] - np.round(sigmoid(np.dot(tx4, w4))))) / tx4.shape[0]))

        best_degree5, best_lambda5 = 2, 0 # optimize_model_ref(self._y[4], self._tX['tX'][4], 1, 3, lambdas=np.logspace(-5, 4, 5), k_fold=10, max_iter=5000, alpha=None)
        tx5 = build_poly(self._tX['tX'][4], best_degree5)
        d = tx5.shape[1]
        w5 = reg_logistic_regression(self._y[4], tx5, best_lambda5, w, 1000, 3)
        print('Model5 accuracy: ' + str(1 - sum(abs(self._y[4] - np.round(sigmoid(np.dot(tx5, w5))))) / tx5.shape[0]))

        best_degree6, best_lambda6 = 2, 0 #optimize_model_ref(self._y[5], self._tX['tX'][5], 1, 3, lambdas=np.logspace(-5, 4, 5), k_fold=10, max_iter=5000, alpha=None)
        tx6 = build_poly(self._tX['tX'][5], best_degree6)
        d = tx6.shape[1]
        w6 = reg_logistic_regression(self._y[5], tx6, best_lambda6, w, 1000, 3)
        print('Model6 accuracy: ' + str(1 - sum(abs(self._y[5] - np.round(sigmoid(np.dot(tx6, w6))))) / tx6.shape[0]))

        self._best_degree = best_degree1, best_degree2, best_degree3, best_degree4, best_degree5, best_degree6
        self.w = w1, w2, w3, w4, w5, w6

    def _predict(self, x):
        """
        Prediction method specific for the model
        :param x: input variables (N, D)
        :return: predicted output y (N, 1)
        """

        y_pred = -1 * np.ones(x['orig_n'])
        tx1 = build_poly(x['tX'][0], self._best_degree[0])
        y_pred[x['ind'][0]] = np.round(sigmoid(tx1.dot(self.w[0])))

        tx2 = build_poly(x['tX'][1], self._best_degree[1])
        y_pred[x['ind'][1]] = np.round(sigmoid(tx2.dot(self.w[1])))

        tx3 = build_poly(x['tX'][2], self._best_degree[2])
        y_pred[x['ind'][2]] = np.round(sigmoid(tx3.dot(self.w[2])))

        tx4 = build_poly(x['tX'][3], self._best_degree[3])
        y_pred[x['ind'][3]] = np.round(sigmoid(tx4.dot(self.w[3])))

        tx5 = build_poly(x['tX'][4], self._best_degree[4])
        y_pred[x['ind'][4]] = np.round(sigmoid(tx5.dot(self.w[4])))

        tx6 = build_poly(x['tX'][5], self._best_degree[5])
        y_pred[x['ind'][5]] = np.round(sigmoid(tx6.dot(self.w[5])))
        y_pred[y_pred == 0] = -1

        return y_pred

if __name__ == "__main__":
    model = ComplexModel()
    res = model.train()
    print(res)
    model.predict_test()

# Model1 accuracy: 0.829256834131
# Model2 accuracy: 0.938586588395
# Model3 accuracy: 0.784687491069
# Model4 accuracy: 0.926077757207
# Model5 accuracy: 0.806179699146
# Model6 accuracy: 0.950809631359
# Total accuracy: 0.827536
