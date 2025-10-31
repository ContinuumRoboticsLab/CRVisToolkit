"""
An implementation of the Newton-Raphson solver according to the MICS MATLAB paper
"""

from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation as R

from ik.solvers.base_solver import CcIkSettings, IkResult, IterativeIkSolver
from ik.solvers.mics.mics_utils import xi2arc_robot
from ik.target import IkTargetType

from ik.solvers.mics.revise_newton import revise_newton


@dataclass
class MicsNewtonRaphsonIkSettings(CcIkSettings):
    max_iter: int = 200
    tol = 1e-2


class MicsNewtonRaphsonIkSolver(IterativeIkSolver):
    """
    in keeping true to the MICS implementation, this solver takes a less class-based
    approach than the other solvers
    """

    target_type: IkTargetType = IkTargetType.SE3
    requires_init_guess: bool = True
    settings_class = MicsNewtonRaphsonIkSettings

    def __init__(self, robot, settings, ik_target_pose):
        super().__init__(robot, settings, ik_target_pose)
        assert len(self.cr.segments) == 3

        self.l1 = self.cr.segments[0].length
        self.l2 = self.cr.segments[1].length
        self.l3 = self.cr.segments[2].length

    def solve(self, *args, **kwargs):
        target_pose = self.ik_target_pose.A
        orientation = target_pose[0:3, 0:3]
        position = target_pose[0:3, 3]

        xi_0 = self.cr._get_body_xi()
        rotation = R.from_matrix(orientation)
        q = rotation.as_quat()
        q = np.array([q[3], q[0], q[1], q[2]])  # w, x, y, z

        xi_star, err, noi = revise_newton(
            self.l1,
            self.l2,
            self.l3,
            q,
            position,
            xi_0,
            self.settings.max_iter,
            self.settings.tol,
        )

        print(f"NR2 finished in {noi} iterations with error {err}")

        arc_params = xi2arc_robot(xi_star, self.cr)
        self.cr.set_config(arc_params.reshape(-1, 2))

        success = err < self.settings.tol

        if success:
            return IkResult.SUCCESS
        else:
            return None
