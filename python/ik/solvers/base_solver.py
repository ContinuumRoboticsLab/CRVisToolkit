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
from numpy.typing import ArrayLike
from spatialmath import SE3, Twist3

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

    # tolerance for position convergence in meters
    position_tolerance: float = 1e-5

    # tolerance for orientation convergence in radians
    orientation_tolerance: float = 1e-5

    max_iter: int = 10

    def check_error_bounds(
        self, d_position: np.ndarray[float], d_orientation: np.ndarray[float]
    ):
        """
        checks if the error is within the bounds specified by the settings

        position error is measured as the norm of the difference between the
        target and current position vectors

        orientation error is measured as the norm of the difference between the
        target and current orientation vectors, in the axis-angle representation
        """
        pos_error = np.linalg.norm(d_position)
        ori_error = np.linalg.norm(d_orientation)

        if self.position_tolerance is None:
            position_check = True
        else:
            position_check = pos_error < self.position_tolerance

        if self.orientation_tolerance is None:
            orientation_check = True
        else:
            orientation_check = ori_error < self.orientation_tolerance

        return (position_check and orientation_check, (pos_error, ori_error))


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

    def __init__(
        self,
        cr: ConstantCurvatureCR,
        settings: CcIkSettings,
        target_pose: np.ndarray[float] | SE3 | Twist3,
        **kwargs,
    ):
        self.cr = deepcopy(cr)
        self.settings = settings

        if isinstance(target_pose, SE3):
            self.target_pose = target_pose.twist()
        elif isinstance(target_pose, ArrayLike):
            self.target_pose = Twist3(target_pose)

        self.target_pose = target_pose  # as a 6x1 pose vector
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
        initial_condition: np.ndarray[float],  # robot configuration
        target_pose: np.ndarray[float],
        **kwargs,
    ):
        super().__init__(cr, settings, target_pose, **kwargs)
        self.inital_condition = initial_condition
        self.iter_count = 0

    def solve(self, *args, **kwargs):
        self._prepare_solver(*args, **kwargs)

        while not self.stopping_condition[0]:
            self._perform_iteration(*args, **kwargs)

        self.solved = True

    def _prepare_solver(self, *args, **kwargs):
        pass

    def _perform_iteration(self, *args, **kwargs):
        raise NotImplementedError

    def _check_error_in_bounds(self, *args, **kwargs):
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
