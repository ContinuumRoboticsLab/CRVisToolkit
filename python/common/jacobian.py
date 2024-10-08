import numpy as np
from typing import Callable

from examples.plot import EXAMPLE_SEGMENTS
from common.robot import ConstantCurvatureCR


# def jacobian(f: Callable, x: np.ndarray[float]):
#     """
#     compute the jacobian matrix of a function at a given point
#     """
#     j_func = nd.Jacobian(f, order=1)
#     return j_func(x)


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


if __name__ == "__main__":
    cr = ConstantCurvatureCR(EXAMPLE_SEGMENTS[:1])
    print(cr.state_vector())
    res = jacobian(cr.pose_vector, cr.state_vector())
    print(f"Jacobian is:\n{res}")
    # print(f"Jacobian2 is:\n{jacobian2(cr.pose_vector, cr.state_vector())}")
