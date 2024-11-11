import numpy as np


def sep_from_seg_endpoint(p: np.ndarray[float]):
    assert p.size == 3, "The endpoint must be a 3D point"

    sigma = np.linalg.norm(p)
    eta = np.acos(p[2] / np.linalg.norm(p))
    phi = np.atan2(p[1], p[0])

    return np.array([sigma, eta, phi])


def sep_as_curvature(sigma: float, eta: float, phi: float) -> np.ndarray[float]:
    """
    the curvature formulation provided in the GCRB paper does not use
    the traditional kappa, phi, and length parameters. Instead, it uses:
    - sigma: euclidian distance between the segment endpoints
    - eta: angle between segment base z-axis and the line connecting the distal endpoint
    - phi: same as in the traditional formulation

    this function converts the GCRB parameters to the traditional parameters.
    """

    kappa = 2 * np.sin(eta) / sigma
    length = 2 * eta / kappa
    return np.array([kappa, phi, length])
