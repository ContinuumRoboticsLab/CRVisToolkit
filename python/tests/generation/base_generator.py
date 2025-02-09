from common.robot import (
    RobotSegmentLimits,
    ConstantCurvatureCR,
    ConstantCurvatureSegment,
)
from tests.generation.test_case import IkTestCase

from numpy import random


class IkTestGenerator:
    def __init__(self, num_segs: int, limits: RobotSegmentLimits = None, seed=None):
        self.num_segs = num_segs
        self.limits = limits

        # even different generators that use different distributions
        # use numpy's "generator" to ensure reproducibility
        self.rng = random.default_rng(seed)

    def generate_case(self) -> IkTestCase:
        starter_segments, target_segments = self._generate_cc_robot_segments()

        target_robot = ConstantCurvatureCR(target_segments)
        starting_robot = ConstantCurvatureCR(starter_segments)

        return IkTestCase(target_robot, starting_robot)

    def _generate_cc_robot_segments(
        self,
    ) -> tuple[list[ConstantCurvatureSegment], list[ConstantCurvatureSegment]]:
        """
        various test case generation strategies will use different
        distributions and impose different restrictions on the generated
        test cases. This method should be implemented by the subclass
        """
        raise NotImplementedError
