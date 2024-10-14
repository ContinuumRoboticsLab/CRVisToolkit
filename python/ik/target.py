import numpy as np
from enum import Enum


class IkTargetType(Enum):
    SE3 = "SE3"
    P3 = "P3"
    SO3 = "SO3"
    POSITION_POINTING = "POSITION_POINTING"
    NEPPALLI = "NEPPALLI"

    @property
    def constraints(self):
        match self:
            case IkTargetType.SE3:
                return 6
            case IkTargetType.P3:
                return 3
            case IkTargetType.SO3:
                return 3
            case IkTargetType.POSITION_POINTING:
                return 5


class IkTarget:
    """
    the base class for the target of the IK problem. Since various solvers will have
    various types of specified end effector targets (i.e. some methods/robot configurations)
    can solve for a target SE(3) pose, while others only offer 5 DoF, etc.
    """

    target_type: IkTargetType

    def as_array(self):
        return self.pose


class SE3IkTarget(IkTarget):
    """
    the target of the IK problem is a SE(3) pose, represented as a 4x4 numpy array
    """

    target_type = IkTargetType.SE3

    def __init__(self, pose: np.ndarray[float]):
        self.pose = pose


class P3IkTarget(IkTarget):
    """
    the target of the IK problem is a R3 position, agnostic to orientation,
    represented as a 3x1 numpy array
    """

    target_type = IkTargetType.P3

    def __init__(self, pose: np.ndarray[float]):
        if isinstance(pose, list):
            pose = np.array(pose)
        elif not isinstance(pose, np.ndarray):
            raise ValueError("Invalid pose type")

        if pose.shape == (6, 1) or pose.shape == (6,):
            pose = pose[:3]
        elif pose.shape != (3, 1):
            raise ValueError("Invalid pose shape")

        self.pose = pose


class SO3IkTarget(IkTarget):
    """
    the target of the IK problem is a SO(3) orientation, agnostic to position,
    represented as a 3x3 numpy array
    """

    target_type = IkTargetType.SO3

    def __init__(self, pose: np.ndarray[float]):
        self.pose = pose


class PositionPointingIkTarget(IkTarget):
    """
    the position-pointing target specifies a target position and a target direction

    this imposes three constraints translationally and two constraints rotationally
    and provides an additional degree of freedom compared to the SE3 target
    """

    target_type = IkTargetType.POSITION_POINTING

    def __init__(
        self, position: np.ndarray[float], pointing_direction: np.ndarray[float]
    ):
        self.position = position
        self.pointing_direction = pointing_direction

    @property
    def as_array(self):
        # semantics here a little odd - target is inherently represented with 2 array entities
        return np.concatenate((self.position, self.pointing_direction))
