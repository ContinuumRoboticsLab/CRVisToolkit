import numpy as np
from scipy.linalg import expm
from common.utils import up_hat, invert_transformation, up_vee
from typing import Callable


def jacobian(f: Callable, x: np.ndarray[float], epsilon: float = 1e-7):
    """
    compute the jacobian matrix of a function at a given point
    """

    assert len(x) == x.size, "x must be a 1D array"

    x = np.asarray(x)
    f_x = np.asarray(f(x))

    # assert len(f_x) == f_x.size, "f(x) must be a 1D array"

    jacobian = np.zeros((f_x.size, x.size))

    for i in range(x.size):
        # perturb the ith element of x -> column i of the jacobian
        x_i = x.copy()
        x_i[i] += epsilon
        jacobian[:, i] = (f(x_i) - f_x) / epsilon

    return jacobian


def _segment_body_jacobian(length, w1, w2):
    """
    calculates the body jacobian columns for a single segment using kappa and phi.

    used as a helper function in calculating the full boyd jacobian
    """
    w = np.array([w1, w2, 0])
    norm = np.linalg.norm(w)

    if norm < 1e-6:
        # only translation, split across the two coordinates
        half_l = length / 2
        translational = np.array(
            [
                [0, half_l, 0],
                [-half_l, 0, 0],
                [0, 0, 0],
            ]
        )
        return np.vstack([np.eye(3), translational])

    else:
        m = (1 - np.cos(norm)) / norm**2
        n = (norm - np.sin(norm)) / norm**3
        p1_m = w1 / norm**2 - w1 * n - 2 * w1 * m / norm**2
        p2_m = w2 / norm**2 - w2 * n - 2 * w2 * m / norm**2
        p1_n = w1 * m / norm**2 - 3 * w1 * n / norm**2
        p2_n = w2 * m / norm**2 - 3 * w2 * n / norm**2

        temp = length * np.array(
            [
                [p1_m * w2, p2_m * w2 + m, 0],
                [-p1_m * w1, -p2_m * w1, 0],
                [-p1_n * n**2, -p2_n * n**2, 0],
            ]
        )

        return np.vstack([np.eye(3) - m * up_hat(w) + n * up_hat(w) @ up_hat(w), temp])


def body_jacobian(lengths: np.ndarray[float], xi: np.ndarray[float]):
    """
    determines the body jacobian for an n-segment inexensible Continuum Robot
    """
    n = len(lengths)

    assert len(xi) == 2 * n, "xi must be a 2n-vector"

    # updated so that the columns/matrices are stored in increasing "i" order
    jacobian_columns = []
    pose_matrices = []
    pose_inv_matrices = []

    # iterate backwards
    for i in range(n - 1, -1, -1):
        kappa, phi = xi[2 * i], xi[2 * i + 1]
        pose_i = expm(up_hat(np.array([kappa, phi, 0, 0, 0, lengths[i]])))
        pose_i_inv = invert_transformation(pose_i)

        ji = _segment_body_jacobian(lengths[i], kappa, phi)

        forward_transformation = np.eye(4)
        for prev_pose in pose_matrices:
            forward_transformation = forward_transformation @ prev_pose

        inv_transformation = np.eye(4)
        for prev_pose_inv in pose_inv_matrices:
            inv_transformation = prev_pose_inv @ inv_transformation

        pose_matrices = [pose_i] + pose_matrices
        pose_inv_matrices = [pose_i_inv] + pose_inv_matrices

        j_c1 = up_vee(inv_transformation @ up_hat(ji[:, 0]) @ forward_transformation)
        j_c2 = up_vee(inv_transformation @ up_hat(ji[:, 1]) @ forward_transformation)

        j_c1 = np.reshape(j_c1, (6, 1))
        j_c2 = np.reshape(j_c2, (6, 1))

        ji = np.hstack([j_c1, j_c2])

        jacobian_columns = [ji] + jacobian_columns

    return np.hstack(jacobian_columns)
