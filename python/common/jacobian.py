import numpy as np
from typing import Callable


def jacobian(f: Callable, x: np.ndarray[float], epsilon: float = 1e-7):
    """
    compute the jacobian matrix of a function at a given point
    """

    assert len(x) == x.size, "x must be a 1D array"

    x = np.asarray(x)
    f_x = np.asarray(f(x))

    assert len(f_x) == f_x.size, "f(x) must be a 1D array"

    jacobian = np.zeros((f_x.size, x.size))

    for i in range(x.size):
        # perturb the ith element of x -> column i of the jacobian
        x_i = x.copy()
        x_i[i] += epsilon
        jacobian[:, i] = (f(x_i) - f_x) / epsilon

    return jacobian
