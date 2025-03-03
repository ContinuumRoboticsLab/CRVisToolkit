from ik.solvers.gcrb.gcrb_solver import GcrbIkSettings, GcrbIkTarget, GcrbSolver2
from tests.runner import MultiSolverTestRunner
from tests.generation.uniform import UniformDistributionGenerator

from ik.solvers.nr import NewtonRaphsonIkSolver, NewtonRaphsonIkSettings
from ik.solvers.neppalli import NeppalliIkSolver, NeppalliIkSettings, NeppalliIkTarget
from ik.solvers.mics import MicsSolver, MicsSolverSettings
from ik.target import SE3IkTarget

from common.robot import RobotSegmentLimits

import time
from structlog import get_logger

logger = get_logger()


def run_twoseg_ext_tests(iternum: int, seed=None):
    logger.info("Running two-segment extensible tests")
    start = time.time()
    num_segs = 2

    solver_classes = [NewtonRaphsonIkSolver, NeppalliIkSolver, GcrbSolver2]
    settings = [NewtonRaphsonIkSettings(), NeppalliIkSettings(), GcrbIkSettings()]
    target_types = [SE3IkTarget, NeppalliIkTarget, GcrbIkTarget]

    generator = UniformDistributionGenerator(
        num_segs, RobotSegmentLimits(is_extensible=True), seed
    )

    runner = MultiSolverTestRunner(
        generator, solver_classes, settings, target_types, num_segs
    )
    runner.run(iternum)
    logger.info("finished tests in {0:.2f} seconds".format(time.time() - start))


def run_twoseg_inext_tests(iternum: int, seed=None):
    logger.info("Running two-segment inextensible tests")
    start = time.time()
    num_segs = 2

    solver_classes = [NewtonRaphsonIkSolver, NeppalliIkSolver]
    settings = [NewtonRaphsonIkSettings(), NeppalliIkSettings()]
    target_types = [SE3IkTarget, NeppalliIkTarget]

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

    solver_classes = [NewtonRaphsonIkSolver, NeppalliIkSolver]
    settings = [NewtonRaphsonIkSettings(), NeppalliIkSettings()]
    target_types = [SE3IkTarget, NeppalliIkTarget]

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

    solver_classes = [NewtonRaphsonIkSolver, NeppalliIkSolver, MicsSolver]
    settings = [NewtonRaphsonIkSettings(), NeppalliIkSettings(), MicsSolverSettings()]
    target_types = [SE3IkTarget, NeppalliIkTarget, SE3IkTarget]

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
    ITERATIONS = 100
    start = time.time()
    # run_twoseg_ext_tests(ITERATIONS * 5, SEED)
    # run_twoseg_inext_tests(ITERATIONS, SEED)
    # run_threeseg_ext_tests(ITERATIONS, SEED)
    run_threeseg_inext_tests(ITERATIONS, SEED)

    # logger.info(f"BENCHMARKS COMPLETED IN {time.time() - start}s")
