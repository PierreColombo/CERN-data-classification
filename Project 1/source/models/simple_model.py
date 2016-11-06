# -*- coding: utf-8 -*-
"""
Project 1
group #28

pierre.colombo@epfl.ch
christian.tresch@epfl.ch
juraj.korcek@epfl.ch
"""
from models.model_class import Model


class SimpleModel(Model):

    def _prepare_x(self, x, mean, std):
        return x[:, 0: 2]

    def _prepare_y(self, y, x):
        return y

    def _train_model(self):
        pass

    def _predict(self, x):
        """
        Prediction method specific for the model
        :param x: input variables (N, D)
        :return: predicted output y (N, 1)
        """
        y_pred = ((x[:, 0] > 105) & (x[:, 0] < 170) & (x[:, 1] < 55)).astype(int)
        y_pred[y_pred == 0] = -1

        return y_pred

if __name__ == "__main__":
    model = SimpleModel()
    model.eval_train()
    # model.predict_test()

# Total accuracy: 0.795672

