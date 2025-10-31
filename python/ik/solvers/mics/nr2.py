"""
An implementation of the Newton-Raphson solver according to the MICS MATLAB paper
"""

from dataclasses import dataclass
import numpy as np
from scipy.linalg import logm
from scipy.spatial.transform import Rotation as R

from common.utils import up_vee
from ik.solvers.base_solver import CcIkSettings, IkResult, IterativeIkSolver
from ik.solvers.mics.jacobian3cc import jacobian3cc
from ik.solvers.mics.mics_utils import arc2xi, xi2arc
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

    def solve2(self, *args, **kwargs):
        target_pose = self.ik_target_pose.A

        omg_e = np.zeros(self.settings.max_iter)
        v_e = np.zeros(self.settings.max_iter)
        e = np.zeros(self.settings.max_iter)

        k = 0
        xi = self.cr._get_body_xi()

        while k < self.settings.max_iter:
            current_pose = self.cr.t_matrix()
            pose_delta = np.linalg.solve(current_pose, target_pose)
            v = up_vee(logm(pose_delta))
            omg_e[k] = np.linalg.norm(v[0:3])
            v_e[k] = np.linalg.norm(v[3:6])
            e[k] = np.linalg.norm(v)

            print(k, e[k])

            if e[k] < self.settings.tol:
                break
            else:
                # get jacobian, etc.
                J = jacobian3cc(self.l1, self.l2, self.l3, xi)
                J_pinv = np.linalg.pinv(J)
                xi += J_pinv @ v
                arc_params = xi2arc(xi, self.cr)
                self.cr.set_config(arc_params.reshape(-1, 2))
                xi = arc2xi(arc_params, self.cr)
                k += 1

        # we'll return True if converged before max iterations
        success = k != self.settings.max_iter

        return (success, IkResult.SUCCESS)

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

        arc_params = xi2arc(xi_star, self.cr)
        self.cr.set_config(arc_params.reshape(-1, 2))

        success = err < self.settings.tol

        return (success, IkResult.SUCCESS)
