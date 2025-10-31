import numpy as np

from common.robot import ConstantCurvatureCR


def arc2xi(arc_params: np.ndarray, cr: ConstantCurvatureCR):
    assert arc_params.shape[0] % 2 == 0, "arc_params should be a 3nx1 array"

    xi = []
    offset = 0
    n_segs = arc_params.shape[0] // 2

    for i in range(n_segs):
        kappa = arc_params[offset]
        phi = arc_params[offset + 1]
        length = cr.segments[i].length

        kappa_eff = kappa * length
        xi1 = -kappa_eff * np.sin(phi)
        xi2 = kappa_eff * -np.cos(phi)

        xi.extend([xi1, xi2])
        offset += 2

    return np.array(xi)


def xi2arc(xi: np.ndarray, cr: ConstantCurvatureCR):
    arc = []

    offset = 0
    for seg in cr.segments:
        kappa = np.sqrt(xi[offset] ** 2 + xi[offset + 1] ** 2)
        kappa = (kappa % np.pi) / seg.length
        phi = np.arctan2(-xi[offset], xi[offset + 1])
        arc.extend([kappa, phi])
        offset += 2

    return np.array(arc)
