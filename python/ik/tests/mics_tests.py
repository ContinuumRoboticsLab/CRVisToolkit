from math import pi
from common.robot import ConstantCurvatureCR, ConstantCurvatureSegment
from ik.target import SE3IkTarget
from ik.solvers.mics import MicsSolverSettings, MicsSolver
import logging


def test_mics_nominations(logger):
    logger.info("**** Test Case 1: three-segment inextensible CR ****")

    seg1 = ConstantCurvatureSegment(1 / 0.14, -0.8 * pi, 0.05)
    seg2 = ConstantCurvatureSegment(1 / 0.06, 0.4 * pi, 0.03)
    seg3 = ConstantCurvatureSegment(1 / 0.065, -pi, 0.035)
    robot = ConstantCurvatureCR([seg1, seg2, seg3])

    target_seg1 = ConstantCurvatureSegment(1 / 0.13, -pi, 0.05)
    target_seg2 = ConstantCurvatureSegment(1 / 0.07, 0.35 * pi, 0.03)
    target_seg3 = ConstantCurvatureSegment(1 / 0.04, -0.9 * pi, 0.035)
    target_robot = ConstantCurvatureCR([target_seg1, target_seg2, target_seg3])

    settings = MicsSolverSettings()

    target_pose = SE3IkTarget(target_robot.t_matrix().A)

    logger.info(f"target pose: {target_pose.pose}")
    logger.info(f"robot configuration: {robot.state_vector()}")

    solver = MicsSolver(robot, settings, target_pose)

    res = solver.solve()  # noqa
    logger.info(
        f"number of candidate MICS starters: {len(solver.mics_starting_points)}"
    )
    # for soln in solver.mics_starting_points:
    # print(soln)


def run(plot=False, loglevel=logging.INFO):
    log_format = "%(levelname)s: %(message)s"
    logging.basicConfig(format=log_format, level=loglevel)
    logger = logging.getLogger(__name__)
    test_mics_nominations(logger)


if __name__ == "__main__":
    run()
