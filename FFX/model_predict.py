import numpy as np

from .utils import  construct_matrix


def predict(coef, mapping,r):
    if len(r.shape) == 1:
        r = r.reshape(1, -1)
    X_P = construct_matrix(mapping, r.T).T
    Arrary = np.zeros_like(X_P)
    for i in range(len(X_P)):
        Arrary[i] = X_P[i] * coef.T
    y_pred = Arrary.sum(axis=1)
    return y_pred

