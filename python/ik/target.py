import numpy as np


class IkTarget:
    """
    the base class for the target of the IK problem. Since various solvers will have
    various types of specified end effector targets (i.e. some methods/robot configurations)
    can solve for a target SE(3) pose, while others only offer 5 DoF, etc.
    """

    pass


class SE3IkTarget(IkTarget):
    """
    the target of the IK problem is a SE(3) pose, represented as a 4x4 numpy array
    """

    def __init__(self, pose: np.array[float]):
        self.pose = pose
