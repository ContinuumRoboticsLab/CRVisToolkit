"""
Test Cases for the Newton-Rhapson Inverse Kinematics Solver

1. a two segment inextensible robot: tested for target R3 position and SE3 pose

2. a two segment extensible robot: tested for target R3 position and SE3 pose

3. a three-segment inextensible robot: tested for target R3 position and SE3 pose
"""

from math import pi
from common.robot import ConstantCurvatureCR, ConstantCurvatureSegment
from ik.target import SE3IkTarget, P3IkTarget
from ik.solvers.nr import NewtonRhapsonIkSettings, NewtonRhapsonIkSolver
import logging

from plotter.tdcr import draw_tdcr


def _run_solver_test(
    target_robot: ConstantCurvatureCR, solver: NewtonRhapsonIkSolver, logger
):
    res = solver.solve()
    logger.info(f"Solution at: {solver.theta_i} after {solver.iter_count} iterations")
    logger.info(f"yields position: {solver.cr.pose_vector(solver.cr.state_vector())}")
    logger.info(
        f"Error: {solver.get_pose() - target_robot.pose_for_target(solver.ik_target.target_type)}"
    )
    return res


def test_nr_1(plot, logger):
    """
    the test case uses two inextensible segments, with target position, no orientation

    this results in there being a degree of redundancy in the solution, so the
    solution found is not guaranteed to be the same as the target robot used to
    generate the target pose
    """

    logger.info("**** Test Case 1: two-segment inextensible CR ****")

    seg1 = ConstantCurvatureSegment(1 / 0.1, pi / 4, 0.05)
    seg2 = ConstantCurvatureSegment(1 / 0.05, 0, 0.03)
    robot = ConstantCurvatureCR([seg1, seg2])

    target_robot = ConstantCurvatureCR(
        [
            ConstantCurvatureSegment(1 / 0.11, pi / 3, 0.05),
            ConstantCurvatureSegment(1 / 0.07, pi / 10, 0.03),
        ]
    )

    logger.info(
        f"starting curvature is: {robot.state_vector()} \
        yielding state\n {robot.pose_vector(robot.state_vector())}"
    )

    target_pose = SE3IkTarget(target_robot.pose_vector())
    target_position = P3IkTarget(target_robot.pose_vector())

    logger.info(f"target pose: {target_pose.pose}")

    se3_solver = NewtonRhapsonIkSolver(
        robot, NewtonRhapsonIkSettings(), robot.state_vector(), target_pose
    )
    se3_res = _run_solver_test(target_robot, se3_solver, logger)

    p3_solver = NewtonRhapsonIkSolver(
        robot, NewtonRhapsonIkSettings(), robot.state_vector(), target_position
    )
    p3_res = _run_solver_test(target_robot, p3_solver, logger)

    if plot:
        logger.info("plotting position solution")
        draw_tdcr(p3_solver.cr.as_discrete_curve(pts_per_seg=10))
        logger.info("plotting pose solution")
        draw_tdcr(se3_solver.cr.as_discrete_curve(pts_per_seg=10))
        logger.info("plotting target robot")
        draw_tdcr(target_robot.as_discrete_curve(pts_per_seg=10))

    return (se3_res, p3_res)


def test_nr_2(plot, logger):
    # Test Case 2: two-segment extensible robot, target pose is an SE3 pose
    logger.info("**** Test Case 2: two-segment extensible CR****")

    seg1 = ConstantCurvatureSegment(1 / 0.14, -0.8 * pi, 0.05, is_extensible=True)
    seg2 = ConstantCurvatureSegment(1 / 0.06, 0.4 * pi, 0.03, is_extensible=True)
    robot = ConstantCurvatureCR([seg1, seg2])

    target_seg1 = ConstantCurvatureSegment(1 / 0.13, -pi, 0.055, is_extensible=True)
    target_seg2 = ConstantCurvatureSegment(
        1 / 0.07, 0.35 * pi, 0.035, is_extensible=True
    )
    target_robot = ConstantCurvatureCR([target_seg1, target_seg2])
    logger.info(
        f"starting curvature is: {robot.state_vector()} \
        yielding state\n {robot.pose_vector(robot.state_vector())}"
    )

    settings = NewtonRhapsonIkSettings()

    target_pose = SE3IkTarget(target_robot.pose_vector())
    target_position = P3IkTarget(target_robot.pose_vector())

    logger.info(f"target pose: {target_pose.pose}")

    se3_solver = NewtonRhapsonIkSolver(
        robot, settings, robot.state_vector(), target_pose
    )
    p3_solver = NewtonRhapsonIkSolver(
        robot, settings, robot.state_vector(), target_position
    )

    se3_res = _run_solver_test(target_robot, se3_solver, logger)
    p3_res = _run_solver_test(target_robot, p3_solver, logger)

    if plot:
        logger.info("plotting position solution")
        draw_tdcr(p3_solver.cr.as_discrete_curve(pts_per_seg=10))
        logger.info("plotting pose solution")
        draw_tdcr(se3_solver.cr.as_discrete_curve(pts_per_seg=10))
        logger.info("plotting target robot")
        draw_tdcr(target_robot.as_discrete_curve(pts_per_seg=10))

    return (se3_res, p3_res)


def test_nr_3(plot, logger):
    """
    testing a three-segment inextensible robot
    """
    logger.info("**** Test Case 2: two-segment extensible CR****")

    seg1 = ConstantCurvatureSegment(1 / 0.14, -0.8 * pi, 0.05)
    seg2 = ConstantCurvatureSegment(1 / 0.06, 0.4 * pi, 0.03)
    seg3 = ConstantCurvatureSegment(1 / 0.065, -pi, 0.035)
    robot = ConstantCurvatureCR([seg1, seg2, seg3])

    target_seg1 = ConstantCurvatureSegment(1 / 0.13, -pi, 0.05)
    target_seg2 = ConstantCurvatureSegment(1 / 0.07, 0.35 * pi, 0.03)
    target_seg3 = ConstantCurvatureSegment(1 / 0.04, -0.9 * pi, 0.035)
    target_robot = ConstantCurvatureCR([target_seg1, target_seg2, target_seg3])
    logger.info(
        f"starting curvature is: {robot.state_vector()} \
        yielding state\n {robot.pose_vector(robot.state_vector())}"
    )

    settings = NewtonRhapsonIkSettings()

    target_pose = SE3IkTarget(target_robot.pose_vector())
    target_position = P3IkTarget(target_robot.pose_vector())

    logger.info(f"target pose: {target_pose.pose}")

    se3_solver = NewtonRhapsonIkSolver(
        robot, settings, robot.state_vector(), target_pose
    )
    p3_solver = NewtonRhapsonIkSolver(
        robot, settings, robot.state_vector(), target_position
    )

    se3_res = _run_solver_test(target_robot, se3_solver, logger)
    p3_res = _run_solver_test(target_robot, p3_solver, logger)

    if plot:
        logger.info("plotting position solution")
        draw_tdcr(p3_solver.cr.as_discrete_curve(pts_per_seg=10))
        logger.info("plotting pose solution")
        draw_tdcr(se3_solver.cr.as_discrete_curve(pts_per_seg=10))
        logger.info("plotting target robot")
        draw_tdcr(target_robot.as_discrete_curve(pts_per_seg=10))

    return (se3_res, p3_res)


def run(plot=False, loglevel=logging.INFO):
    format = "%(levelname)s: %(message)s"
    logging.basicConfig(level=loglevel, format=format)
    logger = logging.getLogger(__name__)
    test_nr_1(plot, logger)
    test_nr_2(plot, logger)
    test_nr_3(plot, logger)
