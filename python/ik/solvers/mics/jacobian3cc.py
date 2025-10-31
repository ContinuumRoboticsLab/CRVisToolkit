"""
Jacobian computation for 3-link robot using product of exponentials formula.

This module provides functions to compute the Jacobian matrix when the forward
kinematics is expressed using the product of exponentials formula.
"""

import numpy as np
from scipy.linalg import expm

from ik.solvers.mics.mics_utils import up_hat, up_vee


def jaco_c12(w1, w2, L):
    """
    Computes the Jacobian for a single link with two joint parameters.

    Parameters
    ----------
    w1 : float
        First joint parameter
    w2 : float
        Second joint parameter
    L : float
        Link length

    Returns
    -------
    Jc : ndarray
        6x2 Jacobian matrix
    """
    w = np.array([w1, w2, 0.0])
    n = np.linalg.norm(w)

    if n == 0:
        ML = L / 2
        Jc = np.vstack([np.eye(3), np.array([[0, ML, 0], [-ML, 0, 0], [0, 0, 0]])])
    else:
        M = (1 - np.cos(n)) / n**2
        N = (n - np.sin(n)) / n**3

        p1M = w1 / n**2 - w1 * N - 2 * w1 * M / n**2
        p2M = w2 / n**2 - w2 * N - 2 * w2 * M / n**2
        p1N = w1 * M / n**2 - 3 * w1 * N / n**2
        p2N = w2 * M / n**2 - 3 * w2 * N / n**2

        pwJleftwv = L * np.array(
            [
                [p1M * w2, p2M * w2 + M, 0],
                [-p1M * w1 - M, -p2M * w1, 0],
                [-p1N * n**2 - 2 * N * w1, -p2N * n**2 - 2 * N * w2, 0],
            ]
        )

        w_hat = up_hat(w)
        w_hat_squared = w_hat @ w_hat

        Jc = np.vstack(
            [np.eye(3) - M * w_hat + N * w_hat_squared, expm(w_hat).T @ pwJleftwv]
        )

    # Return only first 2 columns
    return Jc[:, :2]


def jacobian3cc(L1, L2, L3, xi):
    """
    Computes the Jacobian matrix when the forward kinematics is expressed
    by the product of exponentials formula.

    Parameters
    ----------
    L1 : float
        Length of link 1
    L2 : float
        Length of link 2
    L3 : float
        Length of link 3
    xi : array_like
        6-element array of joint parameters [xi1, xi2, xi3, xi4, xi5, xi6]

    Returns
    -------
    J : ndarray
        6x6 Jacobian matrix
    """
    xi = np.asarray(xi).flatten()

    # Compute j3
    j3 = jaco_c12(xi[4], xi[5], L3)

    # Compute T3 and its inverse
    T3 = expm(up_hat(np.array([xi[4], xi[5], 0, 0, 0, L3])))
    R3 = T3[:3, :3]
    p3 = T3[:3, 3]
    invT3 = np.vstack(
        [np.hstack([R3.T, -R3.T @ p3.reshape(-1, 1)]), np.array([[0, 0, 0, 1]])]
    )

    # Compute j2 and transform it
    j2 = jaco_c12(xi[2], xi[3], L2)
    j2_c1 = up_vee(invT3 @ up_hat(j2[:, 0]) @ T3)
    j2_c2 = up_vee(invT3 @ up_hat(j2[:, 1]) @ T3)
    j2 = np.column_stack([j2_c1, j2_c2])

    # Compute T2 and its inverse
    T2 = expm(up_hat(np.array([xi[2], xi[3], 0, 0, 0, L2])))
    R2 = T2[:3, :3]
    p2 = T2[:3, 3]
    invT2 = np.vstack(
        [np.hstack([R2.T, -R2.T @ p2.reshape(-1, 1)]), np.array([[0, 0, 0, 1]])]
    )

    # Compute j1 and transform it
    j1 = jaco_c12(xi[0], xi[1], L1)
    j1_c1 = up_vee(invT3 @ invT2 @ up_hat(j1[:, 0]) @ T2 @ T3)
    j1_c2 = up_vee(invT3 @ invT2 @ up_hat(j1[:, 1]) @ T2 @ T3)
    j1 = np.column_stack([j1_c1, j1_c2])

    # Assemble full Jacobian
    J = np.hstack([j1, j2, j3])

    return J


if __name__ == "__main__":
    # Example usage
    L1, L2, L3 = 1.0, 1.0, 1.0
    xi = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    J = jacobian3cc(L1, L2, L3, xi)
    print("Jacobian matrix:")
    print(J)
    print(f"\nShape: {J.shape}")
