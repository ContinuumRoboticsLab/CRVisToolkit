from tests.generation.base_generator import IkTestGenerator
from common.robot import (
    ConstantCurvatureSegment,
)


class PerturbedRobotGenerator(IkTestGenerator):
    """
    The perturbed robot generator generates test cases by generating a starter robot,
    then perturbing all arc parameters of all segments using a normalized sum of normals
    distribution.

    The two gaussian peaks of the probability distribution are equidistant from zero and
    have the same stdev.

    The aim of this test case generator is to generate test cases that have target robots
    defined by some perturbation to the starting robot in all of it's configuration parameters.
    This helps provide additional information, especially where numerical solvers are concerned,
    providing more extensive data about the conditions under which a numerical solver can be
    expected to converge.
    """

    def __init__(self, num_segs: int, limits, seed, stdev_percentage: float):
        super().__init__(num_segs, limits, seed)

        self.perturbation_percentage = stdev_percentage
        self.theta_stdev = self.limits.max_theta * stdev_percentage
        self.phi_stdev = self.limits.max_phi * stdev_percentage
        self.length_stdev = self.limits.max_length * stdev_percentage

    def __generate_single_segment(self):
        theta = self.rng.uniform(self.limits.min_theta, self.limits.max_theta)
        phi = self.rng.uniform(self.limits.min_phi, self.limits.max_phi)
        length = self.rng.uniform(self.limits.min_length, self.limits.max_length)

        return ConstantCurvatureSegment(
            theta / length, phi, length, is_extensible=self.limits.is_extensible
        )

    def __get_perturbed_segment(self, segment: ConstantCurvatureSegment):
        """
        returns a segment that has been perturbed by sampling a value for each of the
        segment's arc parameters. Each distribution is a normal with the mean at the
        segment's present value and a stdev defined as a percentage of the allowable
        range of values.
        """
        theta = segment.kappa * segment.length
        theta = self.rng.normal(theta, self.theta_stdev)
        phi = self.rng.normal(segment.phi, self.phi_stdev)
        if self.limits.is_extensible:
            length = self.rng.normal(segment.length, self.length_stdev)
        else:
            length = segment.length

        return ConstantCurvatureSegment(
            theta / length, phi, length, is_extensible=self.limits.is_extensible
        )

    def _generate_cc_robot_segments(self):
        starter = [self.__generate_single_segment() for _ in range(self.num_segs)]
        target = [self.__get_perturbed_segment(seg) for seg in starter]
        # breakpoint()

        return starter, target
