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
from scipy.linalg import logm
import time
from ik.target import IkTarget, IkTargetType

from common.robot import ConstantCurvatureCR
from common.utils import up_vee


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
    position_tolerance: float = 1e-4

    # tolerance for orientation convergence in radians
    orientation_tolerance: float = 1e-4

    max_iter: int = 100

    @classmethod
    def for_target_type(cls, target_type: IkTargetType):
        """
        returns a settings object with default values for the given target type
        certain targets should set certain errors to None
        """
        if target_type == IkTargetType.P3:
            return cls(orientation_tolerance=None)
        elif target_type == IkTargetType.SO3:
            return cls(position_tolerance=None)
        else:
            return cls()

    # check the orientation error checking in MICS, using matrix log
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

    target_type: IkTargetType
    requires_init_guess: bool = False
    settings_class: type[CcIkSettings]

    def __init__(
        self,
        cr: ConstantCurvatureCR,
        settings: CcIkSettings,
        ik_target: IkTarget,
        **kwargs,
    ):
        self.cr = deepcopy(cr)
        self.settings = settings

        self.ik_target = ik_target
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

    def get_pose(self, theta: np.ndarray[float] = None):
        theta = theta if theta is not None else self.cr.state_vector()
        return self.cr.pose_for_target(self.ik_target.target_type, theta)

    def _target_pose(self):
        """
        returns the target pose of the robot as an SE(3) matrix. Must be implemented
        separately for target types that do not inherently use an SE(3) matrix IK target
        """
        return self.ik_target.as_array()

    def get_errors(self) -> tuple[float, float]:
        """
        uses the matrix logarithm of the target pose and current pose
        """
        try:
            target_pose = self._target_pose()
            current_pose = self.cr.t_matrix()

            diff_pose = np.linalg.inv(current_pose) @ target_pose.A

            diff_pose = logm(diff_pose)

            d_position = np.linalg.norm(diff_pose[:3, 3])
            d_orientation = np.linalg.norm(up_vee(diff_pose[:3, :3]))
            return d_position, d_orientation
        except Exception as e:
            print(f"Error in get_errors: {e}")
            return None, None

    @property
    def ik_target_pose(self):
        return self.ik_target.as_array()


class IterativeIkSolver(CcIkSolver):
    """
    Iterative IK solvers should inherit from this class
    """

    def __init__(
        self,
        cr: ConstantCurvatureCR,
        settings: CcIkSettings,
        ik_target_pose: IkTarget,
        **kwargs,
    ):
        super().__init__(cr, settings, ik_target_pose, **kwargs)
        self.iter_count = 0
        self.exec_time = None

    def solve(self, *args, **kwargs):
        start_time = time.time()
        while not self.stopping_condition[0]:
            self._perform_iteration(*args, **kwargs)

        self.exec_time = time.time() - start_time

        return self.stopping_condition[1]

    def _perform_iteration(self, *args, **kwargs):
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

    def get_errors(self):
        return 0.0, 0.0
