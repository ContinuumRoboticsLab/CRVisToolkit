"""
Implementation of the closed-form solver proposed by Neppalli et. al.
in Closed-Form Inverse Kinematics for Continuum Manipulators, 2009

this implementation uses a KPL representation of the robot throughout
and uses the closed-form solution derived by Neppalli et. al. and requires
coordinates for the desired segment endpoints as a prerequisite.

Note: the solver is agnostic to whether or not the segments are exensible.
The segment endpoints passed will only yield one potential solution for
which the segments lengths will be determined.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from math import pi
from time import time

from common.coordinates import CrConfigurationType
from common.robot import ConstantCurvatureCR

from ik.solvers.base_solver import CcIkSettings, AnalyticIkSolver, IkResult
from ik.target import NeppalliIkTarget


class NeppalliIkSettings(CcIkSettings):
    pass


class NeppalliIkSolver(AnalyticIkSolver):
    """
    The closed form Neppalli solver for inverse kinematics

    Parameters
    ----------
    cr: ConstantCurvatureCR
        the CR object to solve the IK for.
    settings: NeppalliIkSettings
        the settings for the solver
    ik_target: NeppalliIkTarget
        the target for the IK problem - in the case of the Neppalli solver,
        defined as a series of n robot segment endpoint coordinates.
    """

    target_type = NeppalliIkTarget
    settings_class = NeppalliIkSettings

    def __init__(
        self,
        cr: ConstantCurvatureCR,
        settings: NeppalliIkSettings,
        ik_target: NeppalliIkTarget,
        **kwargs,
    ):
        endpoints = ik_target.endpoints()
        assert len(endpoints) == len(cr.segments), "Invalid number of segment endpoints"

        if not cr.repr_type == CrConfigurationType.KPL:
            raise ValueError(
                "Neppalli solver requires the robot to be in KPL representation"
            )

        self.segment_endpoints = endpoints

        self.rotation_matrices = []

        super().__init__(cr, settings, ik_target=ik_target, **kwargs)

    def _get_segment_i_theta(self, i) -> float:
        """
        determine the arc angle for segment i

        if the cache exists, then use it. Otherwise, calculate it
        """

        seg_i = self.cr.segments[i]
        p_i = self.segment_endpoints[i]

        theta_i = np.acos(1 - seg_i.kappa * np.linalg.norm(p_i[:2]))

        if p_i[2] <= 0:
            theta_i = 2 * pi - theta_i

        return theta_i

    def _get_segment_i_omega(self, i) -> np.ndarray[float]:
        """
        determine the arc rotation axis for segment i
        """
        seg = self.cr.segments[i]

        return np.array([-np.sin(seg.phi), np.cos(seg.phi), 0])

    def _solve_segment_i(self, i):
        cur_p = self.segment_endpoints[i]

        kappa_i = 2 * np.linalg.norm(cur_p[:2]) / np.inner(cur_p, cur_p)
        phi_i = np.atan2(cur_p[1], cur_p[0])

        # kappa, phi need to set for seg i for _get_segment_i_theta to work
        self.cr.segments[i].set_config(kappa=kappa_i, phi=phi_i)

        theta = self._get_segment_i_theta(i)
        length_i = theta * (1 / kappa_i)

        self.cr.segments[i].set_config(length=length_i)

    def _transform_next_segments(self, i):
        """
        apply reverse transformation of segment i to all endpoints
        for segments i+1, i+2, ... n.

        this function does NOT update the robot configurations, only
        the internal endpoint coordinates so that each segment's
        individual

        assumes segment i's configuration has been solved + set
        """

        omega = self._get_segment_i_omega(i)
        theta = self._get_segment_i_theta(i)

        p_cur = self.segment_endpoints[i]

        rotation = R.from_rotvec(-theta * omega)

        for j in range(i + 1, len(self.segment_endpoints)):
            p_next = self.segment_endpoints[j]
            self.segment_endpoints[j] = rotation.as_matrix() @ (p_next - p_cur)

    def solve(self):
        """
        solve the IK problem using the Neppalli closed form solution
        """
        try:
            start_time = time()
            for i in range(len(self.cr.segments)):
                self._solve_segment_i(i)
                if i < len(self.cr.segments) - 1:
                    self._transform_next_segments(i)
            self.exec_time = time() - start_time

        # TODO: catch less generic exceptions for solution unviable, etc.
        except Exception as e:
            print(f"Error: {e}")
            raise e
            return IkResult.DNE

        return IkResult.SUCCESS


if __name__ == "__main__":
    from ik.tests import neppalli_tests

    neppalli_tests.run(plot=True, randseed=0)
