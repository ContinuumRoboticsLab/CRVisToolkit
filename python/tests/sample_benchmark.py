from ik.solvers.gcrb.gcrb_solver import GcrbIkSettings, GcrbIkTarget, GcrbSolver2
from tests.runner import MultiGenerativeTestRunner
from tests.generation.uniform import UniformRobotFactory
from tests.generation.perturbation import PerturbedRobotGenerator

from ik.solvers.nr import NewtonRaphsonIkSolver, NewtonRaphsonIkSettings
from ik.solvers.neppalli import NeppalliIkSolver, NeppalliIkSettings, NeppalliIkTarget
from ik.solvers.mics import MicsSolver, MicsSolverSettings
from ik.solvers.fabrikc import FabrikcIkSettings, FabrikcIkSolver
from ik.target import SE3IkTarget, P3Direction

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

    generator = UniformRobotFactory(
        num_segs, RobotSegmentLimits(is_extensible=True), seed
    )

    runner = MultiGenerativeTestRunner(
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

    generator = UniformRobotFactory(
        num_segs, RobotSegmentLimits(is_extensible=False), seed
    )

    runner = MultiGenerativeTestRunner(
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

    generator = UniformRobotFactory(
        num_segs, RobotSegmentLimits(is_extensible=True), seed
    )

    runner = MultiGenerativeTestRunner(
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

    generator = UniformRobotFactory(
        num_segs, RobotSegmentLimits(is_extensible=False), seed
    )

    runner = MultiGenerativeTestRunner(
        generator, solver_classes, settings, target_types, num_segs
    )
    runner.run(iternum)
    logger.info(f"finished tests in {time.time() - start} seconds")


def run_perturbed_nr_tests(
    iternum: int, perturbation_values: list[float], seed=None, num_segs=3
):
    logger.info("Running perturbed NR tests")
    start = time.time()
    solver_classes = [NewtonRaphsonIkSolver]
    settings = [NewtonRaphsonIkSettings()]
    target_types = [SE3IkTarget]

    for value in perturbation_values:
        logger.info(f"Running tests with perturbation value {value}")
        generator = PerturbedRobotGenerator(
            num_segs,
            RobotSegmentLimits(is_extensible=False),
            seed,
            stdev_percentage=value,
        )
        runner = MultiGenerativeTestRunner(
            generator, solver_classes, settings, target_types, num_segs
        )
        runner.run(iternum)
        logger.info(f"finished tests in {time.time() - start} seconds")

    logger.info(f"finished all tests in {time.time() - start} seconds")


def run_fabrikc_tests(seed=None):
    solver_classes = [FabrikcIkSolver]
    settings = [FabrikcIkSettings()]
    target_types = [P3Direction]

    generator = UniformRobotFactory(3, RobotSegmentLimits(is_extensible=False), seed)

    runner = MultiGenerativeTestRunner(
        generator, solver_classes, settings, target_types, 2
    )
    runner.run(1000)


if __name__ == "__main__":
    SEED = 0
    ITERATIONS = 100
    start = time.time()
    # run_twoseg_ext_tests(ITERATIONS * 5, SEED)
    # run_twoseg_inext_tests(ITERATIONS, SEED)
    # run_threeseg_ext_tests(ITERATIONS, SEED)
    # run_threeseg_inext_tests(ITERATIONS, SEED)
    # run_perturbed_nr_tests(ITERATIONS, [0.02, 0.05, 0.1, 0.2], SEED)
    run_fabrikc_tests(SEED)

    # logger.info(f"BENCHMARKS COMPLETED IN {time.time() - start}s")
