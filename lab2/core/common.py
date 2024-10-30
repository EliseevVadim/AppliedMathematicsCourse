import numpy as np


def gradient(u1: float, u2: float, dj_u1, dj_u2) -> np.array:
    grad_u1 = dj_u1(u1, u2)
    grad_u2 = dj_u2(u1, u2)
    return np.array([grad_u1, grad_u2])


def must_stop(a, b, epsilon=1e-13):
    return np.abs(a[0] - b[0]) < epsilon and np.abs(a[1] - b[1]) < epsilon
