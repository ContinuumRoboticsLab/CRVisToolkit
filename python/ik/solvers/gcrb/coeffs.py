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

    num_t1 = kap * (r_ti[2][0] * pt[0] + r_ti[2][1] * pt[1] + r_ti[2][2] * pt[2])
    num_t2 = lam * (r_ti[1][0] * pt[0] + r_ti[1][1] * pt[1] + r_ti[1][2] * pt[2])
    num = nu * (num_t1 - num_t2)

    den_t1 = kap * (r_ti[1][0] * pt[0] + r_ti[1][1] * pt[1] + r_ti[1][2] * pt[2])
    den_t2 = lam * (r_ti[2][0] * pt[0] + r_ti[2][1] * pt[1] + r_ti[2][2] * pt[2])
    den = -nu * (den_t1 + den_t2)

    return num / den


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

    t1 = 2 * mu * lam * (lam * r_ti[1][1] - kap * r_ti[2][1]) / mu

    t2 = -(-mu * r_ti[1][1] + lam * r_ti[1][0] - nu * r_ti[2][1] - kap * r_ti[2][0])
    t2 = -t2 * nu / mu

    t3 = -mu * r_ti[1][2] + kap * r_ti[1][0] - mu * r_ti[2][2] + lam * r_ti[2][0]

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
