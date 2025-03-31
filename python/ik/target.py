import numpy as np
from enum import Enum

from common.coordinates import ParamableCoord, CoordParamValue


class IkTarget:
    """
    the base class for the target of the IK problem. Since various solvers will have
    various types of specified end effector targets (i.e. some methods/robot configurations)
    can solve for a target SE(3) pose, while others only offer 5 DoF, etc.
    """

    def as_array(self) -> np.ndarray[float]:
        return self.pose

    @classmethod
    def from_target_robot(cls, target_robot):
        """
        uses the configuration of a robot to generate an inverse kinematics target
        that is guaranteeed to have a solution.
        """

        raise NotImplementedError


class SE3IkTarget(IkTarget):
    """
    the target of the IK problem is a SE(3) pose, represented as a 4x4 numpy array
    """

    def __init__(self, pose: np.ndarray[float]):
        self.pose = pose

    @classmethod
    def from_target_robot(cls, target_robot):
        return cls(target_robot.t_matrix())


class R6TwistIkTarget(IkTarget):
    """
    an R6 target is a 6x1 numpy array representing a twist in 3D space
    """

    def __init__(self, pose: np.ndarray[float]):
        self.pose = pose

    @classmethod
    def from_target_robot(cls, target_robot):
        return cls(target_robot.pose_vector())


class P3IkTarget(IkTarget):
    """
    the target of the IK problem is a R3 position, agnostic to orientation,
    represented as a 3x1 numpy array
    """

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

    @classmethod
    def from_target_robot(cls, target_robot):
        return cls(target_robot.pose_vector())


class SO3IkTarget(IkTarget):
    """
    the target of the IK problem is a SO(3) orientation, agnostic to position,
    represented as a 3x3 numpy array
    """

    def __init__(self, pose: np.ndarray[float]):
        self.pose = pose


class P3Direction(IkTarget):
    """
    the P3 Direction target specifies a target position and a target direction

    this imposes three constraints translationally and two constraints rotationally
    and provides an additional degree of freedom compared to the SE3 target
    """

    def __init__(
        self, position: np.ndarray[float], pointing_direction: np.ndarray[float]
    ):
        self.position = position
        self.pointing_direction = pointing_direction

    @property
    def as_array(self):
        # semantics here a little odd - target is inherently represented with 2 array entities
        return np.concatenate((self.position, self.pointing_direction))

    @classmethod
    def from_target_robot(cls, target_robot):
        ee_pose = target_robot.t_matrix().A

        z_axis = ee_pose[:3, 2]
        position = ee_pose[:3, 3]

        return cls(position, z_axis)


class NeppalliIkTarget(IkTarget):
    """
    The Neppalli solver requires its own IK target class
    since it does not target a single end-effector pose or
    position as it's target, but rather a set of segment
    endpoint coordinates.
    """

    def __init__(self, seg_endpoints: list[np.ndarray[float]]):
        self.seg_endpoints = seg_endpoints

    def as_array(self):
        raise Exception("Neppalli solver does not target a single pose")

    def endpoints(self):
        return self.seg_endpoints

    @classmethod
    def from_target_robot(cls, target_robot):
        return cls(target_robot.segment_endpoints())


class GcrbIkTarget(IkTarget):
    def __init__(self, ee_target: np.ndarray, param: CoordParamValue):
        self.ee_target = SE3IkTarget(ee_target)
        self.param = param

    def as_array(self):
        return self.ee_target.as_array()

    @classmethod
    def from_target_robot(cls, target_robot, coord: ParamableCoord = ParamableCoord.Z):
        ee_target = target_robot.t_matrix().A

        if coord == ParamableCoord.Z:
            param = CoordParamValue(
                ParamableCoord.Z, target_robot.segment_endpoints()[0][2]
            )
        else:
            raise NotImplementedError

        return cls(ee_target, param)


class IkTargetType(Enum):
    SE3 = "SE3"
    R6 = "R6"
    P3 = "P3"
    SO3 = "SO3"
    POSITION_POINTING = "POSITION_POINTING"
    NEPPALLI = "NEPPALLI"
    GCRB = "GCRB"

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

    def ik_target_class(self):
        match self:
            case IkTargetType.SE3:
                return SE3IkTarget
            case IkTargetType.R6:
                return R6TwistIkTarget
            case IkTargetType.P3:
                return P3IkTarget
            case IkTargetType.SO3:
                return SO3IkTarget
            case IkTargetType.POSITION_POINTING:
                return P3Direction
            case IkTargetType.NEPPALLI:
                return NeppalliIkTarget
            case IkTargetType.GCRB:
                return GcrbIkTarget
