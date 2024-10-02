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
from ik.solvers.base_solver import IterativeIkSolver, CcIkSettings, IkResult


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

    def __init__(
        self,
        cr: ConstantCurvatureCR,
        settings: NewtonRhapsonIkSettings,
        initial_condition: np.array[float] | list[float],
        **kwargs,
    ):
        num_segs = cr.num_segments

        if isinstance(initial_condition, list):
            initial_condition = np.array(initial_condition)

        # check if the initial condition is of valid dimensionality
        if initial_condition.shape != (3 * num_segs, 1):
            try:
                initial_condition = initial_condition.reshape((3 * num_segs, 1))
            except Exception:
                raise ValueError(
                    f"Invalid initial condition. Expected shape: {(3*num_segs, 1)}"
                )

        # make sure we're using the right representation
        if not cr.representation == CrConfigurationType.KPL:
            raise ValueError("Only KPL representation is supported for the NR solver")

        self.theta_i = initial_condition

        super().__init__(cr, settings, initial_condition=initial_condition, **kwargs)

    def _prepare_solver(self, *args, **kwargs):
        # prepare method-specific attributes

        # dimensionality of the solution
        self.n = self.cr.num_segments * 3
        # dimensionality of relevant task space
        self.m = 6

        self.twist = np.zeros((self.n, 1))

    def __compute_jacobian(self):
        """
        compute the Jacobian matrix at the current solution
        returns an (m x n) matrix
        """

    def __compute_twist(self):
        """
        compute the twist vector at the current solution
        returns an (m x 1) vector
        """
        # TODO: before implementing, we need a way to get the
        # end-effector T matrix from the CR object

    def _perform_iteration(self, *args, **kwargs):
        """
        performs a single iteration of:
        theta_(i+1) = theta_i - pinv(J(theta_i)) * twist(theta_i)

        calculation of both the jacobian and twist are done in other
        object methods
        """

        # update the CR internal state
        self._update_cr_configuration()

        j = self.__compute_jacobian()
        v = self.__compute_twist()

        self.theta_i = self.theta_i + np.linalg.pinv(j) @ v

        return super()._perform_iteration(*args, **kwargs)

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

        theta_i_per_segment = np.split(theta_i, self.cr.num_segments)
        self.cr.set_config(theta_i_per_segment)

    def _compute_error(self, *args, **kwargs):
        """
        compute the error for the current solution
        """
        # TODO: implementation requires forward kinematics integration
        # ^ some refactoring of the existing robotindependentmapping function
        # might be worth doing beforehand

        return 0.0

    @property
    def stoppping_condition(self) -> tuple[bool, IkResult | None]:
        if self._compute_error() < self.settings.tolerance:
            return (True, IkResult.SUCCESS)
        elif self.iteration > self.settings.max_iterations:
            return (True, IkResult.MAX_ITERATIONS)
        else:
            return (False, None)
