from common.robot import (
    ConstantCurvatureSegment,
    RobotSegmentLimits,
    ConstantCurvatureCR,
)

from numpy import random

UNIFORM_TEST_NAME = "start_uniform"


class UniformRobotFactory:
    """
    the uniform distrbution generator helps generate test cases by generating constant
    curvature segments by smapling from a uniform distribution over the segment limits
    specified.
    """

    def __init__(self, n, limits: RobotSegmentLimits, seed=None):
        self.rng = random.default_rng(seed)
        self.num_segs = n
        self.limits = limits

    def __generate_single_segment(self, length=None):
        """
        for the uniform distribution, all segments are generated
        agnostic of the other robot or other segments.
        """

        theta = self.rng.uniform(self.limits.min_theta, self.limits.max_theta)
        phi = self.rng.uniform(self.limits.min_phi, self.limits.max_phi)
        if length is None:
            length = self.rng.uniform(self.limits.min_length, self.limits.max_length)

        return ConstantCurvatureSegment(
            theta / length, phi, length, is_extensible=self.limits.is_extensible
        )

    def generate_cr(self):
        """
        generates a constant curvature robot with `self.num_segs` segments
        """
        segments = [self.__generate_single_segment() for _ in range(self.num_segs)]
        return ConstantCurvatureCR(segments)
