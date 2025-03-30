from common.robot import (
    ConstantCurvatureSegment,
    ConstantCurvatureCR,
    RobotSegmentLimits,
)

from numpy import random


PERTURBATION_VALUES = [0.01, 0.05, 0.1, 0.2, 0.3]
STARTING_POSITION_VARS = [
    f"start_perturbed_{stdev:.2f}".replace(".", "_") for stdev in PERTURBATION_VALUES
]


class RobotPerturber:
    """
    a class used to make repeatedly perturbing robots using the same distributions
    easier.
    """

    def __init__(self, seed, limits: RobotSegmentLimits):
        self.limits = limits
        self.rng = random.default_rng(seed)

    def __get_stdevs(self, stdev_percentage: float):
        """
        returns the standard deviations for the perturbation distributions.
        """

        # get the stdevs for the perturbation distributions
        theta_stdev = stdev_percentage * self.limits.max_theta
        phi_stdev = stdev_percentage * (self.limits.max_phi - self.limits.min_phi)
        length_stdev = (
            stdev_percentage * (self.limits.max_length - self.limits.min_length)
            if self.limits.is_extensible
            else 0
        )

        return theta_stdev, phi_stdev, length_stdev

    def _get_perturbed_segment(
        self, segment: ConstantCurvatureSegment, stdev_percentage: float
    ) -> ConstantCurvatureSegment:
        """
        returns a segment that has been perturbed by sampling a value for each of the
        segment's arc parameters. Each distribution is a normal with the mean at the
        segment's present value and a stdev defined as a percentage of the allowable
        range of values.
        """

        t_stdev, p_stdev, l_stdev = self.__get_stdevs(stdev_percentage)

        segment_theta = segment.kappa * segment.length
        theta = self.rng.normal(segment_theta, t_stdev)
        phi = self.rng.normal(segment.phi, p_stdev)
        if segment.is_extensible:
            length = self.rng.normal(segment.length, l_stdev)
        else:
            length = segment.length

        return ConstantCurvatureSegment(
            theta / length, phi, length, is_extensible=segment.is_extensible
        )

    def get_perturbed_robot(
        self, robot: ConstantCurvatureCR, stdev_percentage
    ) -> ConstantCurvatureCR:
        """
        returns a perturbed robot by perturbing each segment of the robot.
        """
        segments = [
            self._get_perturbed_segment(seg, stdev_percentage) for seg in robot.segments
        ]
        return ConstantCurvatureCR(segments)
