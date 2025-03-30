from common.robot import ConstantCurvatureCR, RobotSegmentLimits

from tests.generation.uniform import UniformDistributionGenerator
from tests.generation.perturbation import RobotPerturber, PERTURBATION_VALUES
from tests.generation.test_case import IkTestCase, STARTING_POSITION_VARS


class IkTestGenerator:
    """
    the IK Test generator works as follows:
    - for each test case, generate a "target" robot
    - generate a "starting" robot for each kind of starting point:
        - random (sample all arc params from same dist. as starter)
        - each perturbation value (arc params sampled from normal dist.)
    """

    def __init__(self, num_segs: int, limits: RobotSegmentLimits = None, seed=None):
        self.num_segs = num_segs
        self.limits = limits if limits else RobotSegmentLimits()
        self.seed = seed

        self.robot_factory = UniformDistributionGenerator(num_segs, limits, seed=seed)
        self.perturber = RobotPerturber(seed, limits)

    def _generate_starters(
        self, target: ConstantCurvatureCR
    ) -> dict[str, ConstantCurvatureCR]:
        uniform_starter = self.robot_factory.generate_cr()

        perturbed_starters = {
            field_name: self.perturber.get_perturbed_robot(
                target, stdev_percentage=stdev
            )
            for field_name, stdev in zip(
                STARTING_POSITION_VARS[1:], PERTURBATION_VALUES
            )
        }
        perturbed_starters["start_uniform"] = uniform_starter

        return perturbed_starters

    def generate_test_case(self) -> IkTestCase:
        target = self.robot_factory.generate_cr()
        starters = self._generate_starters(target)

        return IkTestCase(target_robot=target, **starters)
