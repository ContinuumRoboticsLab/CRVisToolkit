"""
this module does not define any particular solver, but defines a unified interface
for all solvers so that they can be used interchangeably with ease.

The BaseSolver is the most generic, and has two subclasses: one for iterative solvers, and
another for closed form/analytic solvers. All solvers should inherit from one of the two
subclasses, and implement the init and solve methods.

The Settings classes are also structured similarly.
"""

from dataclasses import dataclass
from enum import Enum
from copy import deepcopy
import numpy as np

from common.robot import ConstantCurvatureCR
from ik.index import IkSolverType


@dataclass
class CcIkSettings:
    """
    the base class for the settings for the closed form IK solver

    Note: this settings class does not inclue all paramters required to
    fully define an IK solver, since different solvers require some parameters
    unique to each solver. Additionally, not all classes will require all
    settings defined below

    Parameters
    ----------
    tol: float, default=1e-4
        Tolerance for convergence
    """

    tolerance: float = 1e-4
    max_iter: int = 100


class IkResult(Enum):
    """
    the result of the IK solver
    """

    SUCCESS = 0
    MAX_ITER = 1
    DIVERGED = 2
    DNE = 3

    @property
    def is_success(self):
        return self == IkResult.SUCCESS


class CcIkSolver:
    """
    the base class for all Inverse Kinematics solvers that work
    under the constant curvature assumption
    """

    solver_type: IkSolverType

    def __init__(self, cr: ConstantCurvatureCR, settings: CcIkSettings, **kwargs):
        self.cr = deepcopy(cr)
        self.settings = settings
        self.solved = False

    def solve(self, *args, **kwargs):
        """
        method should be implemented by all subclasses - args, kwargs are placeholders
        """
        raise NotImplementedError

    def solved_cr(self):
        """
        returns the CR object with the updated curvature parameters
        """
        assert self.solved, "The IK problem has not been solved yet"
        return self.cr


class IterativeIkSolver(CcIkSolver):
    """
    Iterative IK solvers should inherit from this class
    """

    def __init__(
        self,
        cr: ConstantCurvatureCR,
        settings: CcIkSettings,
        initial_condition: np.array[float],
        **kwargs,
    ):
        super().__init__(cr, settings, **kwargs)
        self.inital_condition = initial_condition

    def solve(self, *args, **kwargs):
        self._prepare_solver(*args, **kwargs)

        while not self.stopping_condition:
            self._perform_iteration(*args, **kwargs)

        self.solved = True

    def _prepare_solver(self, *args, **kwargs):
        pass

    def _perform_iteration(self, *args, **kwargs):
        raise NotImplementedError

    def _compute_error(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def stopping_condition(self) -> tuple[bool, IkResult | None]:
        """
        private method to check if the stopping condition is met

        returns two values: a boolean indicating if the stopping condition is met,
        and a result value indicating the reason for stopping
        """
        raise NotImplementedError


class AnalyticIkSolver(CcIkSolver):
    """
    Analytic IK solvers should inherit from this class
    """

    def _solve_closed_form(self, *args, **kwargs):
        """
        private method to solve the IK problem
        """
        raise NotImplementedError
