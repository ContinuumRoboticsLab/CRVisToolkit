import numpy as np
from spatialmath import UnitQuaternion
from copy import deepcopy

from common.robot import ConstantCurvatureCR
from common.utils import curvature_from_seg_endpoint
from common.coordinates import CoordParamValue, ParamableCoord

from ik.solvers.base_solver import CcIkSettings, AnalyticIkSolver, IkResult
from ik.target import IkTarget, IkTargetType

from ik.solvers.gcrb.coeffs import c0_z, c1_z, c2_z, c3_z, c4_z


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
        se3 = self.ik_target.as_array()
        self.target_rotation = se3[:3, :3]
        self.r_ti = self.target_rotation.T
        self.target_position = se3[:3, 3]

        # determine R4 quaternion representation of target rotation
        self.target_quaternion = UnitQuaternion(self.target_rotation).vec

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
        disc = np.sqrt((c2 * z + c1) ** 2 - 4 * c4(c3 * z**2 + c0 * z))
        x1 = (-c2 * z - c1 + disc) / (2 * c4)
        x2 = (-c2 * z - c1 - disc) / (2 * c4)

        _, lam, nu, mu = self.target_quaternion

        y1 = -(lam * x1 + mu * z) / nu
        y2 = -(lam * x2 + mu * z) / nu

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

        # find full curvature params for the first solution
        sol1_seg1_curvature = curvature_from_seg_endpoint(junction1)
        sol1_seg2_coordinates = self.r_ti @ (self.target_position - junction1)
        sol1_seg2_curvature = curvature_from_seg_endpoint(sol1_seg2_coordinates)

        # find full curvature params for the second solution
        sol2_seg1_curvature = curvature_from_seg_endpoint(junction2)
        sol2_seg2_coordinates = self.r_ti @ (self.target_position - junction2)
        sol2_seg2_curvature = curvature_from_seg_endpoint(sol2_seg2_coordinates)

        # solutions obtained, set the CRs
        self.cr.set_config([sol1_seg1_curvature, sol1_seg2_curvature])
        self.cr2.set_config([sol2_seg1_curvature, sol2_seg2_curvature])

        return IkResult.SUCCESS
