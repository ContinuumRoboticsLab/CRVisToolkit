"""
Implementation of the Newton-Rhapson method for inverse kinematics.

this implementation uses a discrete constant curvate representation of a continuum robot
and uses the Newton-Rhapson method to solve for the curvature parameters that constitute
a solution to the IK problem.

The Jacobian is computed using the finite differences method.
"""

import numpy as np

from common.coordinates import CrConfigurationType
from common.robot import ConstantCurvatureCR
from common.jacobian import jacobian

from ik.solvers.base_solver import IterativeIkSolver, CcIkSettings, IkResult
from ik.index import IkSolverType


class NewtonRhapsonIkSettings(CcIkSettings):
    pass


class NewtonRhapsonIkSolver(IterativeIkSolver):
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
        settings: NewtonRhapsonIkSettings,
        initial_condition: np.ndarray[float] | list[float],
        ik_target_pose: np.ndarray[float],
        **kwargs,
    ):
        if isinstance(initial_condition, list):
            initial_condition = np.array(initial_condition)

        self.total_dof = sum([seg.n for seg in cr.segments])

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

        super().__init__(
            cr,
            settings,
            initial_condition=initial_condition,
            ik_target_pose=ik_target_pose,
            **kwargs,
        )

    def _prepare_solver(self, *args, **kwargs):
        # prepare method-specific attributes

        # dimensionality of the solution
        self.n = self.total_dof
        # dimensionality of relevant task space
        self.ik_target.target_type.constraints

    def __compute_jacobian(self):
        """
        compute the Jacobian matrix at the current solution
        returns an (m x n) matrix
        """
        return jacobian(self.get_pose, self.cr.state_vector())

    def __compute_twist(self):
        """
        computes the twist vector at the current solution
        returns an (m x 1) vector
        """
        return self.ik_target_pose - self.get_pose()

    def _perform_iteration(self, *args, **kwargs):
        """
        performs a single iteration of:
        theta_(i+1) = theta_i + pinv(J(theta_i)) * twist(theta_i)

        calculation of both the jacobian and twist are done in other
        object methods
        """

        # breakpoint()
        old_theta = self.theta_i

        # update the CR internal state
        self._update_cr_configuration()

        j = self.__compute_jacobian()

        self.cr.set_config(old_theta)

        pose = self.get_pose()

        diff = self.ik_target_pose - pose

        if j.shape[0] == j.shape[1]:
            j_inv = np.linalg.inv(j)
        else:
            j_inv = np.linalg.pinv(j)

        d_theta = j_inv @ diff

        # breakpoint()
        # if self.ik_target.target_type == IkTargetType.P3:
        #     j = self.__compute_jacobian()

        #     # should always have same dimensionality
        #     diff = self.ik_target_pose - self.get_pose()

        #     d_theta = np.linalg.pinv(j) @ diff

        # elif self.ik_target.target_type == IkTargetType.SE3:
        #     j = self.__compute_jacobian()

        #     d_theta = np.linalg.pinv(j) @ self.__compute_twist()

        self.theta_i += np.reshape(d_theta, (d_theta.size, 1))

        self.iter_count += 1

    def _update_cr_configuration(self):
        """
        updates the CR object using the current solution
        necessary for performing forward kinematics step
        """

        # validate the parameters
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
            print(f"Iteration {self.iter_count} - Error: {error}")
            return (False, None)


if __name__ == "__main__":
    """
    if the module is run as main, a couple examples of the NR solver will be run
    """

    from ik.tests import nr_tests

    nr_tests.run(plot=True)
