"""
define the coefficients for the GCRB solver, done
separately to keep the main solver code clean

each function takes the same parameters:
    uq: the unit quaternion in R4 representation
    r_ti: the inverse of the orientation matrix of the tooltip
"""

import numpy as np


# coefficients for the x-parameterized solution
def c0_x(uq: np.ndarray, r_ti: np.ndarray, pt: np.ndarray) -> float:
    raise NotImplementedError


def c1_x(uq: np.ndarray, r_ti: np.ndarray, pt: np.ndarray) -> float:
    raise NotImplementedError


def c2_x(uq: np.ndarray, r_ti: np.ndarray, pt: np.ndarray) -> float:
    raise NotImplementedError


def c3_x(uq: np.ndarray, r_ti: np.ndarray, pt: np.ndarray) -> float:
    raise NotImplementedError


def c4_x(uq: np.ndarray, r_ti: np.ndarray, pt: np.ndarray) -> float:
    raise NotImplementedError


# coefficients for the y-parameterized solution
def c0_y(uq: np.ndarray, r_ti: np.ndarray, pt: np.ndarray) -> float:
    raise NotImplementedError


def c1_y(uq: np.ndarray, r_ti: np.ndarray, pt: np.ndarray) -> float:
    raise NotImplementedError


def c2_y(uq: np.ndarray, r_ti: np.ndarray, pt: np.ndarray) -> float:
    raise NotImplementedError


def c3_y(uq: np.ndarray, r_ti: np.ndarray, pt: np.ndarray) -> float:
    raise NotImplementedError


def c4_y(uq: np.ndarray, r_ti: np.ndarray, pt: np.ndarray) -> float:
    raise NotImplementedError


# coefficients for the z-parameterized solution
def c0_z(uq: np.ndarray, r_ti: np.ndarray, pt: np.ndarray) -> float:
    kap, lam, mu, nu = uq

    t1_kap = kap * (r_ti[2][0] * pt[0] + r_ti[2][1] * pt[1] + r_ti[2][2] * pt[2])
    t1_lam = lam * (r_ti[1][0] * pt[0] + r_ti[1][1] * pt[1] + r_ti[1][2] * pt[2])
    t1 = nu * (t1_kap - t1_lam) / mu

    t2_kap = -kap * (r_ti[1][0] * pt[0] + r_ti[1][1] * pt[1] + r_ti[1][2] * pt[2])
    t2_lam = -lam * (r_ti[2][0] * pt[0] + r_ti[2][1] * pt[1] + r_ti[2][2] * pt[2])
    t2 = t2_kap + t2_lam

    return -t1 + t2


def c1_z(uq: np.ndarray, r_ti: np.ndarray, pt: np.ndarray) -> float:
    kap, lam, mu, nu = uq

    t1 = (mu * r_ti[1][0] + nu * r_ti[2][0]) * pt[0]
    t2 = (mu * r_ti[1][1] + nu * r_ti[2][1]) * pt[1]
    t3 = (mu * r_ti[1][2] + nu * r_ti[2][2]) * pt[2]

    t4_num_t1 = (kap * r_ti[2][0] - lam * r_ti[1][0]) * pt[0]
    t4_num_t2 = (kap * r_ti[2][1] - lam * r_ti[1][1]) * pt[1]
    t4_num_t3 = (kap * r_ti[2][2] - lam * r_ti[1][2]) * pt[2]

    t4_num = -lam * (t4_num_t1 + t4_num_t2 + t4_num_t3)
    t4 = t4_num / mu

    return t1 + t2 + t3 + t4


def c2_z(uq: np.ndarray, r_ti: np.ndarray) -> float:
    kap, lam, mu, nu = uq

    t1 = 2 * nu * lam * (lam * r_ti[1][1] - kap * r_ti[2][1]) / (mu**2)

    t2_num = -mu * r_ti[1][1] + lam * r_ti[1][0] - nu * r_ti[2][1] - kap * r_ti[2][0]
    t2 = -t2_num * nu / mu

    t3 = -mu * r_ti[1][2] + kap * r_ti[1][0] - nu * r_ti[2][2] + lam * r_ti[2][0]

    t4_1 = lam * r_ti[1][2] + kap * r_ti[1][1] - kap * r_ti[2][2] + lam * r_ti[2][1]
    t4 = -lam * t4_1 / mu

    return t1 + t2 + t3 + t4


def c3_z(uq: np.ndarray, r_ti: np.ndarray) -> float:
    kap, lam, mu, nu = uq

    t1 = (lam * r_ti[1][1] - kap * r_ti[2][1]) * (nu**2) / (mu**2)

    t2 = kap * r_ti[1][2] + lam * r_ti[2][2]

    t3_num = lam * r_ti[1][2] + kap * r_ti[1][1] - kap * r_ti[2][2] + lam * r_ti[2][1]
    t3 = -t3_num * nu / mu

    return t1 + t2 + t3


def c4_z(uq: np.ndarray, r_ti: np.ndarray) -> float:
    kap, lam, mu, nu = uq

    t1 = -mu * r_ti[1][0] - nu * r_ti[2][0]

    t2 = (lam * r_ti[1][1] - kap * r_ti[2][1]) * (lam**2) / (mu**2)

    t3_num = -mu * r_ti[1][1] + lam * r_ti[1][0] - nu * r_ti[2][1] - kap * r_ti[2][0]
    t3 = -t3_num * lam / mu

    return t1 + t2 + t3


"""
coefficients for the respective singular cases for each parametrized axis

these parameter values were determined using the sympy solver - source code can be found in
ik/solvers/gcrb/derivation.py

ex. when the z-axis value is parameterized, an expression of the following form is determined:
c4s * y ** 2 + c3s * z ** 2 + c2s * y * z + c1s * y + c0s * z = 0
"""


def c0s_z(uq: np.ndarray, r_ti: np.ndarray, pt: np.ndarray) -> float:
    kappa, lambda_, _, _ = uq
    pt_i, pt_j, pt_k = pt
    r_ti_21, r_ti_22, r_ti_23 = r_ti[1]
    r_ti_31, r_ti_32, r_ti_33 = r_ti[2]

    res = kappa * (pt_i * r_ti_21 + pt_j * r_ti_22 + pt_k * r_ti_23) + lambda_ * (
        2 * pt_i * r_ti_31 + 2 * pt_j * r_ti_32 + 2 * pt_k * r_ti_33
    )
    return res


def c1s_z(uq: np.ndarray, r_ti: np.ndarray, pt: np.ndarray) -> float:
    kappa, lambda_, _, nu = uq
    pt_i, pt_j, pt_k = pt
    r_ti_21, r_ti_22, r_ti_23 = r_ti[1]
    r_ti_31, r_ti_32, r_ti_33 = r_ti[2]

    res = (
        kappa * (pt_i * r_ti_21 + pt_j * r_ti_22 + pt_k * r_ti_23)
        + lambda_ * (pt_i * r_ti_31 + pt_j * r_ti_32 + pt_k * r_ti_33)
        + (nu**2 * pt_i * r_ti_31 + nu**2 * pt_j * r_ti_32 + nu**2 * pt_k * r_ti_33)
        / lambda_
    )

    return res


def c2s_z(uq: np.ndarray, r_ti: np.ndarray) -> float:
    kappa, lambda_, _, nu = uq
    r_ti_21, r_ti_22, r_ti_23 = r_ti[1]
    r_ti_31, r_ti_32, r_ti_33 = r_ti[2]

    res = (
        kappa * (-r_ti_22 + r_ti_33)
        + lambda_ * (-r_ti_23 - r_ti_32)
        + nu * r_ti_21
        + (-kappa * nu * r_ti_31 - nu**2 * r_ti_32) / lambda_
    )
    return res


def c3s_z(uq: np.ndarray, r_ti: np.ndarray) -> float:
    kappa, lambda_, _, nu = uq
    r_ti_21, _, r_ti_23 = r_ti[1]
    r_ti_31, _, r_ti_33 = r_ti[2]

    res = (
        -kappa * r_ti_23
        - lambda_ * r_ti_33
        + nu * r_ti_31
        + (kappa * nu * r_ti_21 - nu**2 * r_ti_33) / lambda_
        + nu**3 * r_ti_31 / lambda_**2
    )
    return res


def c4s_z(uq: np.ndarray, r_ti: np.ndarray) -> float:
    kappa, lambda_, _, _ = uq
    return kappa * r_ti[2][1] - lambda_ * r_ti[1][1]
