from tests.runner import MultiSolverTestRunner
from tests.generation.uniform import UniformDistributionGenerator

from ik.solvers.nr import NewtonRhapsonIkSolver, NewtonRhapsonIkSettings
from ik.solvers.neppalli import NeppalliIkSolver, NeppalliIkSettings, NeppalliIkTarget
from ik.solvers.gcrb.gcrb_solver import GcrbSolver2, GcrbIkSettings, GcrbIkTarget
from ik.solvers.mics import MicsSolver, MicsSolverSettings
from ik.target import R6TwistIkTarget, SE3IkTarget

from common.robot import RobotSegmentLimits

import time
from structlog import get_logger

logger = get_logger()


def run_twoseg_ext_tests(iternum: int, seed=None):
    logger.info("Running two-segment extensible tests")
    start = time.time()
    num_segs = 2

    solver_classes = [NewtonRhapsonIkSolver, NeppalliIkSolver, GcrbSolver2]
    settings = [NewtonRhapsonIkSettings(), NeppalliIkSettings(), GcrbIkSettings()]
    target_types = [R6TwistIkTarget, NeppalliIkTarget, GcrbIkTarget]

    generator = UniformDistributionGenerator(
        num_segs, RobotSegmentLimits(is_extensible=True), seed
    )

    runner = MultiSolverTestRunner(
        generator, solver_classes, settings, target_types, num_segs
    )
    runner.run(iternum)
    logger.info(f"finished tests in {time.time() - start} seconds")


def run_twoseg_inext_tests(iternum: int, seed=None):
    logger.info("Running two-segment inextensible tests")
    start = time.time()
    num_segs = 2

    solver_classes = [NewtonRhapsonIkSolver, NeppalliIkSolver]
    settings = [NewtonRhapsonIkSettings(), NeppalliIkSettings()]
    target_types = [R6TwistIkTarget, NeppalliIkTarget]

    generator = UniformDistributionGenerator(
        num_segs, RobotSegmentLimits(is_extensible=False), seed
    )

    runner = MultiSolverTestRunner(
        generator, solver_classes, settings, target_types, num_segs
    )
    runner.run(iternum)
    logger.info(f"finished tests in {time.time() - start} seconds")


def run_threeseg_ext_tests(iternum: int, seed=None):
    logger.info("Running three-segment extensible tests")
    start = time.time()
    num_segs = 3

    solver_classes = [NewtonRhapsonIkSolver, NeppalliIkSolver]
    settings = [NewtonRhapsonIkSettings(), NeppalliIkSettings()]
    target_types = [R6TwistIkTarget, NeppalliIkTarget]

    generator = UniformDistributionGenerator(
        num_segs, RobotSegmentLimits(is_extensible=True), seed
    )

    runner = MultiSolverTestRunner(
        generator, solver_classes, settings, target_types, num_segs
    )
    runner.run(iternum)
    logger.info(f"finished tests in {time.time() - start} seconds")


def run_threeseg_inext_tests(iternum: int, seed=None):
    logger.info("Running three-segment inextensible tests")
    start = time.time()
    num_segs = 3

    solver_classes = [NewtonRhapsonIkSolver, NeppalliIkSolver, MicsSolver]
    settings = [NewtonRhapsonIkSettings(), NeppalliIkSettings(), MicsSolverSettings()]
    target_types = [R6TwistIkTarget, NeppalliIkTarget, SE3IkTarget]

    generator = UniformDistributionGenerator(
        num_segs, RobotSegmentLimits(is_extensible=False), seed
    )

    runner = MultiSolverTestRunner(
        generator, solver_classes, settings, target_types, num_segs
    )
    runner.run(iternum)
    logger.info(f"finished tests in {time.time() - start} seconds")


if __name__ == "__main__":
    SEED = 1006842534
    start = time.time()
    run_twoseg_ext_tests(1000, SEED)
    # run_twoseg_inext_tests(100, SEED)
    # run_threeseg_ext_tests(100, SEED)
    # run_threeseg_inext_tests(50, SEED)

    logger.info(f"BENCHMARKS COMPLETED IN {time.time() - start} SECONDS")
