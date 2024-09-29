from unittest import TestCase
import numpy as np

from common.robot import ConstantCurvatureSegment, ConstantCurvatureCR


class RobotReprTestCase(TestCase):
    """
    kappa = np.array([1/30e-3, 1/40e-3, 1/15e-3])
    phi = np.array([0, np.deg2rad(160), np.deg2rad(30)])
    ell = np.array([50e-3, 70e-3, 25e-3])
    """

    def test_cr_repr(self):
        seg1 = ConstantCurvatureSegment(1 / 30e-3, np.deg2rad(0), 50e-3)
        seg2 = ConstantCurvatureSegment(1 / 40e-3, np.deg2rad(160), 70e-3)
        seg3 = ConstantCurvatureSegment(1 / 15e-3, np.deg2rad(30), 25e-3)

        cr = ConstantCurvatureCR([seg1, seg2, seg3])

        # output of conversion to array, correct as of Sept. 28 2024
        res = np.array(
            [
                0.43582919,
                -0.6060648,
                -0.66538588,
                0.0,
                -0.64002953,
                0.31107526,
                -0.70256272,
                0.0,
                0.63278363,
                0.73206396,
                -0.2523237,
                0.0,
                0.08117057,
                0.03206565,
                0.08115269,
                1.0,
            ]
        ).reshape(1, 16)

        cr_coords = cr.as_discrete_curve(pts_per_seg=30).g
        self.assertTrue(np.allclose(cr_coords[-1], res))


if __name__ == "__main__":
    from unittest import main

    main()
