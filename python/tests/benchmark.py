from tests.runner import TestRunner
from tests.generation.uniform import UniformDistributionGenerator

from ik.solvers.nr import NewtonRhapsonIkSolver, NewtonRhapsonIkSettings
from ik.solvers.neppalli import NeppalliIkSolver, NeppalliIkSettings, NeppalliIkTarget
from ik.target import SE3IkTarget

from common.robot import RobotSegmentLimits


def run_nr_test(num_segs: int, iternum: int, seed=None):
    settings = NewtonRhapsonIkSettings()
    generator = UniformDistributionGenerator(num_segs, RobotSegmentLimits(), seed)
    runner = TestRunner(
        generator, NewtonRhapsonIkSolver, settings, SE3IkTarget, num_segs
    )
    runner.run(iternum)


def run_neppalli_test(num_segs: int, iternum: int, seed=None):
    settings = NeppalliIkSettings()
    generator = UniformDistributionGenerator(num_segs, RobotSegmentLimits(), seed)
    runner = TestRunner(
        generator, NeppalliIkSolver, settings, NeppalliIkTarget, num_segs
    )
    runner.run(iternum)


if __name__ == "__main__":
    SEED = 1006842534
    # run_nr_test(2, 5, SEED)
    run_neppalli_test(2, 100, SEED)
