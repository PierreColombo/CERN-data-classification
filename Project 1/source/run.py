# -*- coding: utf-8 -*-
"""
Project 1
group #28

pierre.colombo@epfl.ch
christian.tresch@epfl.ch
juraj.korcek@epfl.ch
"""
from models.complex_model import ComplexModel


if __name__ == "__main__":
    model = ComplexModel()
    res = model.train()
    print(res)
    model.predict_test()