"""
Implementation of the Newton-Rhapson method for inverse kinematics.

this implementation uses a discrete constant curvate representation of a continuum robot
and uses the Newton-Rhapson method to solve for the curvature parameters that constitute
a solution to the IK problem.

The Jacobian is computed using the finite differences method.
"""

import numpy as np
from scipy.linalg import logm
from dataclasses import dataclass

from common.coordinates import CrConfigurationType
from common.robot import ConstantCurvatureCR
from common.jacobian import jacobian

from ik.solvers.base_solver import IterativeIkSolver, CcIkSettings, IkResult
from ik.target import IkTarget, IkTargetType
from ik.index import IkSolverType

from common.utils import se3_to_pose, up_vee


@dataclass
class NewtonRaphsonIkSettings(CcIkSettings):
    position_tolerance: float = 1e-4
    orientation_tolerance: float = 1e-4
    exponential_coord_tolerance: float = 1e-2
    max_iter: int = 100
    clamp_theta: bool = False


class NewtonRaphsonIkSolver(IterativeIkSolver):
    """
    Implementation of the Newton-Rhapson method for inverse kinematics.

    NOTE: this implementation assumes the robot is in KPL representation

    Notation/Convention
    --------
    theta_i: the solution to the IK problem at current iteration


    Parameters
    ----------
    cr: ContinuousCurvatureCR
        the CR object to solve the IK for.
    settings: CcIkSettings
        the settings for the solver
    initial_condition: np.array[float]
        the initial condition theta_0 for the solver.
        The input should be a 3nx1 array as follows:
        [kappa_i, phi_i, length_i] for each segment i in the CR object
    """

    solver_type = IkSolverType.NR

    def __init__(
        self,
        cr: ConstantCurvatureCR,
        settings: NewtonRaphsonIkSettings,
        ik_target_pose: IkTarget,
        **kwargs,
    ):
        self.total_dof = sum([seg.n for seg in cr.segments])
        initial_condition = cr.state_vector()

        # check if the initial condition is of valid dimensionality
        if initial_condition.shape != (self.total_dof, 1):
            try:
                initial_condition = initial_condition.reshape((self.total_dof, 1))
            except Exception:
                raise ValueError(
                    f"Invalid initial condition. Expected shape: {(self.total_dof, 1)}"
                )

        # make sure we're using the right representation
        if not cr.repr_type == CrConfigurationType.KPL:
            raise ValueError("Only KPL representation is supported for the NR solver")

        self.theta_i = np.reshape(initial_condition, (initial_condition.size, 1))

        self.n = self.total_dof

        super().__init__(
            cr,
            settings,
            initial_condition=initial_condition,
            ik_target_pose=ik_target_pose,
            **kwargs,
        )

    def __get_theta(self):
        """
        returns the current configuration vector of the CR object
        as it should be used for determining the Jacobian.
        """
        return self.cr.state_vector()

    def __compute_jacobian(self):
        """
        compute the Jacobian matrix at the current solution
        returns an (m x n) matrix
        """
        if self.ik_target.target_type == IkTargetType.SE3:
            return self.cr.get_body_jacobian()
        else:
            return jacobian(self.get_pose, self.cr.state_vector())

    def __compute_twist(self):
        """
        computes the twist vector at the current solution
        returns an (m x 1) vector
        """
        return self.ik_target_pose - self.get_pose()

    def __update_theta(self, d_theta):
        """
        updates the theta configuration vector of the robot using the delta
        theta `d_theta` computed in the iteration and applies clamping if
        the setting is enabled.
        """
        new_theta_i = self.theta_i + d_theta

        if self.settings.clamp_theta:
            # TODO: iterate over all segments, clamp curvature
            theta_index = 0
            for seg in self.cr.segments:
                theta_values = new_theta_i[theta_index : theta_index + seg.n]

                kappa = theta_values[0]
                new_kappa = np.clip(kappa, 0, seg.max_curvature)
                new_theta_i[theta_index] = new_kappa

                if seg.n == 3:  # segment is extensible
                    length = theta_values[2]
                    new_length = np.clip(length, seg.len_limits[0], seg.len_limits[1])
                    new_theta_i[theta_index + 2] = new_length

        self.theta_i = new_theta_i

    def _perform_iteration(self, *args, **kwargs):
        """
        performs a single iteration of:
        theta_(i+1) = theta_i + pinv(J(theta_i)) * twist(theta_i)

        calculation of both the jacobian and twist are done in other
        object methods
        """

        pose = self.cr.t_matrix()
        vee = up_vee(logm(np.linalg.inv(pose) @ self.ik_target.pose.A))

        error = np.linalg.norm(vee)
        if error < self.settings.exponential_coord_tolerance:
            self.solved = True
            return
        else:
            # perform update step
            j = self.__compute_jacobian()
            xi = self.cr._get_body_xi()

            xi += np.linalg.pinv(j) @ vee

            # set current robot state using xi
            self.cr._set_state_from_xi(xi)

        self.iter_count += 1

    def __update_cr_configuration(self, d_theta):
        """
        updates the CR object's configuration using the current iteration's delta
        theta `d_theta`.

        if the solver is configured to clamp the curvature parameters, then the
        new theta will be clamped to within the valid range before the segment
        is updated.
        """

        # update self.theta_i, clamping if necessary
        d_theta = np.reshape(d_theta, (d_theta.size, 1))
        self.__update_theta(d_theta)

        # validate the new parameters, then update the C
        if not self.theta_i.shape == (self.n, 1):
            try:
                theta_i = self.theta_i.reshape((self.n, 1))
            except Exception:
                raise ValueError(f"Invalid theta_i shape: {self.theta_i.shape}")
        else:
            theta_i = self.theta_i

        # theta_i_per_segment = np.split(theta_i, self.cr.num_segments)
        self.cr.set_config(theta_i)

    def _check_error_in_bounds(self, *args, **kwargs):
        """
        compute the error for the current solution, return True if the error
        is within the acceptable bounds set by the settings object

        the error is computed as the norm of the difference between the current
        pose and the target pose, and tolerances are set in the settings object
        for position and orientation separately
        """

        if self.ik_target.target_type == IkTargetType.SE3:
            error = self.cr.pose_vector() - se3_to_pose(self.ik_target_pose.A)
        else:
            error = self.get_pose() - self.ik_target_pose

        # of form (check result, (position error, orientation error))
        error_res = self.settings.check_error_bounds(error[:3], error[3:])
        return error_res

    @property
    def stopping_condition(self) -> tuple[bool, IkResult | None]:
        if self.solved:
            # possible that other parts of algorithm set solved to True
            return (True, IkResult.SUCCESS)
        (error_in_bounds, error) = self._check_error_in_bounds()
        if error_in_bounds:
            return (True, IkResult.SUCCESS)
        elif self.iter_count > self.settings.max_iter:
            return (True, IkResult.MAX_ITER)
        else:
            # print(f"Iteration {self.iter_count} - Error: {error}")
            return (False, None)


if __name__ == "__main__":
    """
    if the module is run as main, a couple examples of the NR solver will be run
    """

    from ik.tests import nr_tests

    nr_tests.run(plot=True)
