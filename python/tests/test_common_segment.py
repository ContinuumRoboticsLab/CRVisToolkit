from unittest import TestCase
import numpy as np

from common.robot import ContinuousCurvatureSegment, ContinuousCurvatureCR
from plotter.tdcr import draw_tdcr


class RobotReprTestCase(TestCase):
    """
    kappa = np.array([1/30e-3, 1/40e-3, 1/15e-3])
    phi = np.array([0, np.deg2rad(160), np.deg2rad(30)])
    ell = np.array([50e-3, 70e-3, 25e-3])
    """

    def test_cr_repr(self):
        seg1 = ContinuousCurvatureSegment(1 / 30e-3, np.deg2rad(0), 50e-3)
        seg2 = ContinuousCurvatureSegment(1 / 40e-3, np.deg2rad(160), 70e-3)
        seg3 = ContinuousCurvatureSegment(1 / 15e-3, np.deg2rad(30), 25e-3)

        cr = ContinuousCurvatureCR([seg1, seg2, seg3])
        draw_tdcr(cr.as_discrete_curve(pts_per_seg=30))


if __name__ == "__main__":
    from unittest import main

    main()
