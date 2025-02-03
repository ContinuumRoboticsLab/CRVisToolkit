from common.robot import (
    RobotSegmentLimits,
    ConstantCurvatureCR,
    ConstantCurvatureSegment,
)
from tests.generation.test_case import IkTestCase

from numpy import random


class UniformDistributionGenerator:
    def __init__(self, num_segs: int, limits: RobotSegmentLimits = None, seed=None):
        self.num_segs = num_segs
        self.limits = limits

        self.rng = random.default_rng(seed)

    def _generate_cc_robot_segments(self, is_extensible: bool = False):
        starter = [
            ConstantCurvatureSegment.random(self.rng, self.limits)
            for _ in range(self.num_segs)
        ]

        if self.limits.is_extensible:
            target = [
                ConstantCurvatureSegment.random(self.rng, self.limits)
                for _ in range(self.num_segs)
            ]
        else:
            target = [
                ConstantCurvatureSegment.random_from_l(
                    starter_seg.length, self.rng, self.limits
                )
                for starter_seg in starter
            ]

        return starter, target

    def generate_case(self) -> IkTestCase:
        starter_segments, target_segments = self._generate_cc_robot_segments()

        target_robot = ConstantCurvatureCR(target_segments)
        starting_robot = ConstantCurvatureCR(starter_segments)

        return IkTestCase(target_robot, starting_robot)
