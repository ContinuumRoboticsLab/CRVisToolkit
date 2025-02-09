from common.robot import ConstantCurvatureSegment

from tests.generation.base_generator import IkTestGenerator


class UniformDistributionGenerator(IkTestGenerator):
    """
    The uniform distribution generator generates test cases by generating
    a starter and target robot for whom all n segments are generated using
    a uniform distribution for all arc parameters, and independently of one
    another. The target robot state is independent of the starting robot state.
    """

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

    def _generate_cc_robot_segments(self):
        starter = [self.__generate_single_segment() for _ in range(self.num_segs)]

        if self.limits.is_extensible:
            target = [
                self.__generate_single_segment(starter_seg.length)
                for starter_seg in starter
            ]
        else:
            target = [
                self.__generate_single_segment(length=seg.length) for seg in starter
            ]

        return starter, target
