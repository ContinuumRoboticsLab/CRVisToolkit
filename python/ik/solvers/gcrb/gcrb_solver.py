import numpy as np
from spatialmath import SE3
from copy import deepcopy

from common.robot import ConstantCurvatureCR
from common.utils import curvature_to_se3, se3_to_uq
from common.coordinates import CoordParamValue, ParamableCoord

from ik.solvers.base_solver import CcIkSettings, AnalyticIkSolver, IkResult
from ik.target import IkTarget, IkTargetType


from ik.solvers.gcrb.coeffs import c0_z, c1_z, c2_z, c3_z, c4_z
from ik.solvers.gcrb.utils import sep_as_curvature, sep_from_seg_endpoint


class GcrbIkSettings(CcIkSettings):
    pass


class GcrbSolver2(AnalyticIkSolver):
    """
    the analytic Gcrb solver for two-segment extensible continuum robots.

    This base solver class is only applicable to two-segment extensible robots.

    The solver operates somewhat differently from other solvers in that it will
    provide two solutions for the junction position, one for each segment.

    if .cr is called, one of the two solutions arbitrarily is returned. The alternate
    solution can be obtained by calling .cr2
    """

    def __init__(
        self,
        cr: ConstantCurvatureCR,
        settings: GcrbIkSettings,
        ik_target: IkTarget,
        parameter: CoordParamValue,
        **kwargs,
    ):
        super().__init__(cr, settings, ik_target, **kwargs)

        # the parameterized coordinate
        self.parameter = parameter

        assert (
            self.ik_target.target_type == IkTargetType.SE3
        ), "GcrbSolver2 only supports SE3 targets"

        # the target rotation and position
        # se3 = SE3(self.ik_target.as_array())
        # self.target_pose = se3.A
        # self.target_rotation = se3.A[:3, :3]
        # self.r_ti = se3.A[:3, :3].T
        # self.target_position = se3.A[:3, 3]
        # self.target_quaternion = UnitQuaternion(self.target_rotation).A

        # applies implicit validation on the input
        se3 = SE3(self.ik_target.as_array()).A
        self.target_pose = np.ndarray.copy(se3)
        self.target_rotation = np.ndarray.copy(se3)[:3, :3]
        self.r_ti = np.ndarray.copy(se3)[:3, :3].T
        self.target_position = np.ndarray.copy(se3)[:3, 3]
        # determine R4 quaternion representation of target rotation
        self.target_quaternion = se3_to_uq(self.target_rotation)

        self.cr2 = deepcopy(self.cr)

    def _get_coeffs_x(self):
        """
        returns the constats c_{0, .. , 4} used to solve for the y, z
        coordinates of the junction position when the x-value is known
        """
        raise NotImplementedError

    def _get_coeffs_y(self):
        """
        returns the constats c_{0, .. , 4} used to solve for the x, z
        coordinates of the junction position when the y-value is known
        """
        raise NotImplementedError

    def _get_coeffs_z(self):
        """
        returns the constats c_{0, .. , 4} used to solve for the x, y
        coordinates of the junction position when the z-value is known
        """
        c0 = c0_z(self.target_quaternion, self.r_ti, self.target_position)
        c1 = c1_z(self.target_quaternion, self.r_ti, self.target_position)
        c2 = c2_z(self.target_quaternion, self.r_ti)
        c3 = c3_z(self.target_quaternion, self.r_ti)
        c4 = c4_z(self.target_quaternion, self.r_ti)
        return c0, c1, c2, c3, c4

        # c0_alt = c0_sub(self.target_quaternion, self.r_ti, self.target_position)
        # c1_alt = c1_sub(self.target_quaternion, self.r_ti, self.target_position)
        # c2_alt = c2_sub(self.target_quaternion, self.r_ti)
        # c3_alt = c3_sub(self.target_quaternion, self.r_ti)
        # c4_alt = c4_sub(self.target_quaternion, self.r_ti, self.target_position)

        # return c0_alt, c1_alt, c2_alt, c3_alt, c4_alt

    def _solve_segment_junction_x(self):
        """
        Solve for the y, z coordinates of the junction position
        when the x-value is known
        """
        raise NotImplementedError

    def _solve_segment_junction_y(self):
        """
        Solve for the x, z coordinates of the junction position
        when the y-value is known
        """
        raise NotImplementedError

    def _solve_segment_junction_z(self):
        """
        Solve for the x, y coordinates of the junction position
        when the z-value is known
        """
        c0, c1, c2, c3, c4 = self._get_coeffs_z()

        z = self.parameter.value

        a = c4
        b = c2 * z + c1
        c = c3 * (z**2) + c0 * z

        disc = b**2 - 4 * a * c
        if disc < 0:
            raise ValueError("No real solutions")
        elif disc == 0:
            print("Warning: only one solution")
        disc = np.sqrt(disc)

        x1 = (-b + disc) / (2 * a)
        x2 = (-b - disc) / (2 * a)

        lam, mu, nu = self.target_quaternion[1:]

        y1 = -(lam * x1 + nu * z) / mu
        y2 = -(lam * x2 + nu * z) / mu

        return np.array([x1, y1, z]), np.array([x2, y2, z])

    def solve_segment_junction(self):
        """
        Solve for the R3 coordinates of the junction between the
        two segments
        """
        match self.parameter.coordinate:
            case ParamableCoord.X:
                return self._solve_segment_junction_x()
            case ParamableCoord.Y:
                return self._solve_segment_junction_y()
            case ParamableCoord.Z:
                return self._solve_segment_junction_z()
            case _:
                raise ValueError("Invalid coordinate")

    def _config_from_junction(
        self, junction: np.ndarray[float]
    ) -> list[np.ndarray[float]]:
        """
        Given the junction position, find the curvature parameters
        for the two segments
        """
        # find full curvature params for the first solution
        seg1_sep = sep_from_seg_endpoint(junction)
        seg1_curvature = sep_as_curvature(*seg1_sep)

        # find segment 2
        seg1_t = curvature_to_se3(seg1_curvature)
        seg1_t_i = seg1_t.inv()
        seg2_distal_pose = seg1_t_i * self.target_pose
        seg2_distal_position = seg2_distal_pose[:3, 3]

        seg2_sep = sep_from_seg_endpoint(seg2_distal_position)
        seg2_curvature = sep_as_curvature(*seg2_sep)

        return [seg1_curvature, seg2_curvature]

    def solve(self):
        """
        Solve the IK problem for the GcrbSolver2
        """
        """
        junction = self.solve_segment_junction()

        seg1 = self.junction_to_segment(junction, 0)
        seg2 = self.junction_to_segment(junction, 1)
        return
        """

        junction1, junction2 = self.solve_segment_junction()
        print(f"junction1: {junction1}")
        print(f"junction2: {junction2}")
        self.cr.set_config(self._config_from_junction(junction1))
        self.cr2.set_config(self._config_from_junction(junction2))

        return IkResult.SUCCESS


if __name__ == "__main__":
    from ik.tests import gcrb_tests

    gcrb_tests.run(plot=True)
