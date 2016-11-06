# -*- coding: utf-8 -*-
"""
Project 1
group #28

pierre.colombo@epfl.ch
christian.tresch@epfl.ch
juraj.korcek@epfl.ch
"""
import abc

import numpy as np

from proj1_helpers import load_csv_data, create_csv_submission


class Model(metaclass=abc.ABCMeta):
    _DATA_TRAIN_PATH = '../data/train.csv'
    _DATA_TEST_PATH = '../data/test.csv'

    def __init__(self):
        self._tX = None
        self._tX_orig = None
        self._x_mean = None
        self._x_std = None
        self._y = None
        self._y_orig = None
        self._orig_train = False
        self._best_degree = None
        self._w = None
        self._tX_test = None
        self._ids_test = None
        self._orig_test = False
        self._output_path = '../../to_submit/prediction' + '_' + self.__class__.__name__ + '.csv'

    @staticmethod
    def prepare_all_data(y, tX):
        """
        Dummy-codes categorical variable.
        :return: y, tX: (N, 1), (N, D)
        """
        # split categorical variable (23); 23 replaced by 23 == 1; 31 is 23 == 2; 32 is 23 == 3
        cat_vect = tX[:, 22]
        tX[:, 22] = (cat_vect == 1).astype(int)
        tX = np.vstack((tX.T, (cat_vect == 2).astype(int))).T
        tX = np.vstack((tX.T, (cat_vect == 3).astype(int))).T

        return y, tX

    def _prepare_model_data(self, y, x):
        """
        Modified in-place as numpy.arrays are passed by reference => no return value
        :param y: output (N, 1); can be None (case of test data); function must be robust to this (if y is not None:)
        :param x: input variables (N, D)
        :return: None
        """
        model_datapoints_no = x.shape[0]
        print('Model data count: ' + str(model_datapoints_no))
        tx = self._prepare_x(x.copy(), self._x_mean, self._x_std)

        if y is not None:
            y = self._prepare_y(y.copy(), tx)

        # if y is not None and (tx.shape[0] != y.shape[0] or (len(y.shape) > 1 and y.shape[1] != 1)):
        #     raise Exception('Wrong dimensions! x:' + str(tx.shape) + ' y: ' + str(y.shape))

        print('Data shape: ' + str(x.shape))

        return y, tx

    @abc.abstractmethod
    def _prepare_x(self, x, mean, std):
        pass

    @abc.abstractmethod
    def _prepare_y(self, y, x):
        pass

    @abc.abstractmethod
    def _train_model(self):
        pass

    @property
    def w(self):
        if self._w is not None:
            return self._w
        else:
            raise Exception('w not initialized! use cross validation or set it manually')

    @w.setter
    def w(self, value):
        self._w = value

    @abc.abstractmethod
    def _predict(self, x):
        """
        Prediction method specific for the model
        :param x: input variables (N, D)
        :return: predicted output y (N, 1)
        """
        raise NotImplementedError

    def train(self, y=None, x=None):
        if y is None or x is None:
            if self._orig_train is False:
                self._x_mean, self._x_std = None, None
                y, tX, _ = load_csv_data(self._DATA_TRAIN_PATH)
                self._y, self._tX = self.prepare_all_data(y, tX)
                self._tX_orig = self._tX.copy()
                self._y_orig = self._y.copy()
                self._y, self._tX = self._prepare_model_data(self._y, self._tX)
                self._orig_train = True
        else:
            self._x_mean, self._x_std = None, None
            self._y, self._tX = self.prepare_all_data(y.copy(), x.copy())
            self._tX_orig = self._tX.copy()
            self._y_orig = self._y.copy()
            self._y, self._tX = self._prepare_model_data(self._y, self._tX)
            self._orig_train = False

        self._train_model()

        return self.eval_train()

    def eval_train(self):
        if self._tX is None:
            y, tX, _ = load_csv_data(self._DATA_TRAIN_PATH)
            self._y, self._tX = self.prepare_all_data(y, tX)
            self._tX_orig = self._tX.copy()
            self._y_orig = self._y.copy()
            self._y, self._tX = self._prepare_model_data(self._y, self._tX)
            self._orig_train = True

        y_pred = self._predict(self._tX)

        # performance of model 5 on train dataset
        acc = 1 - sum(abs(self._y_orig - y_pred) / 2) / self._y_orig.shape[0]
        print('Total accuracy: ' + str(acc))
        return acc

    def predict_test(self, x=None, ids=None):
        if x is None or ids is None:
            if self._orig_test is False:
                _, _tX_test, self._ids_test = load_csv_data(self._DATA_TEST_PATH)
                _, self._tX_test = self.prepare_all_data(None, _tX_test)
                self._tX_orig = self._tX_test.copy()
                _, self._tX_test = self._prepare_model_data(None, self._tX_test)
                self._orig_test = True
        else:
            _, self._tX_test = self.prepare_all_data(None, x.copy())
            self._tX_orig = self._tX_test.copy()
            self._ids_test = ids.copy()
            _, self._tX_test = self._prepare_model_data(None, self._tX_test)
            self._orig_test = False

        y_test_pred = self._predict(self._tX_test)
        create_csv_submission(self._ids_test, y_test_pred, self._output_path)
