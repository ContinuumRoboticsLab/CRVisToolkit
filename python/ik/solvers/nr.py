"""
Implementation of the Newton-Rhapson method for inverse kinematics.

this implementation uses a discrete constant curvate representation of a continuum robot
and uses the Newton-Rhapson method to solve for the curvature parameters that constitute
a solution to the IK problem.

The Jacobian is computed using the finite differences method.
"""

import numpy as np

from common.robot import ConstantCurvatureCR
from ik.solvers.base_solver import IterativeIkSolver, CcIkSettings, IkResult


class NewtonRhapsonIkSettings(CcIkSettings):
    pass


class NewtonRhapsonIkSolver(IterativeIkSolver):
    """
    Implementation of the Newton-Rhapson method for inverse kinematics.

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
        the initial condition theta_0 for the solver. The input should be a 3nx1 array as follows:
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

        super().__init__(cr, settings, initial_condition=initial_condition, **kwargs)

    def _prepare_solver(self, *args, **kwargs):
        # prepare method-specific attributes

        # dimensionality of the solution
        self.n = self.cr.num_segments * 3
        # dimensionality of relevant task space
        self.m = 6

        self.twist = np.zeros((self.n, 1))

    def _perform_iteration(self, *args, **kwargs):
        return super()._perform_iteration(*args, **kwargs)

    def _compute_error(self, kappa, phi, length):
        """
        compute the error for the current solution
        """
        # TODO: implementation requires forward kinematics integration
        # ^ some refactoring of the existing robotindependentmapping function
        # might be worth doing beforehand

    @property
    def stoppping_condition(self) -> tuple[bool, IkResult | None]:
        if self._compute_error() < self.settings.tolerance:
            return (True, IkResult.SUCCESS)
        elif self.iteration > self.settings.max_iterations:
            return (True, IkResult.MAX_ITERATIONS)
        else:
            return (False, None)
