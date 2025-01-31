from common.robot import (
    RobotSegmentLimits,
    ConstantCurvatureCR,
    ConstantCurvatureSegment,
)
from ik.target import IkTarget
from tests.generation.test_case import IkTestCase

from numpy import random


class UniformDistributionGenerator:
    def __init__(self, num_segs: int, limits: RobotSegmentLimits = None, seed=None):
        self.num_segs = num_segs
        self.limits = limits

        self.rng = random.default_rng(seed)

    def _generate_segments(self):
        return [
            ConstantCurvatureSegment.random(self.rng, self.limits)
            for _ in range(self.num_segs)
        ]

    def generate_case(self, target_type: type[IkTarget]) -> IkTestCase:
        target_segments = self._generate_segments()
        starter_segments = self._generate_segments()

        target_robot = ConstantCurvatureCR(target_segments)

        ik_target = target_type.from_target_robot(target_robot)
        starting_robot = ConstantCurvatureCR(starter_segments)

        return IkTestCase(ik_target, starting_robot)
