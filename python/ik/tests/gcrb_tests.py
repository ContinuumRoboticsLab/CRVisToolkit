"""
Tests for the GCRB analytic IK solver.


"""

from math import pi
import numpy as np
import matplotlib.pyplot as plt

from common.robot import ConstantCurvatureCR, ConstantCurvatureSegment
from common.coordinates import CoordParamValue, ParamableCoord
from common.types import TDCRPlotterSettings
from common.utils import uq_to_so3, se3_to_pose

from ik.target import SE3IkTarget
from ik.solvers.gcrb.gcrb_solver import GcrbSolver2, GcrbIkSettings

from spatialmath import SO3


from plotter.tdcr import draw_tdcr

import logging


def test_base_case(logger, plot=False):
    """
    a two-segment, unrandomized base case
    """

    segment1 = ConstantCurvatureSegment(1 / 0.1, pi / 6, 0.05, is_extensible=True)
    segment2 = ConstantCurvatureSegment(1 / 0.05, 2 * pi / 3, 0.05, is_extensible=True)
    target_robot = ConstantCurvatureCR([segment1, segment2])
    target_pose = target_robot.pose_vector()

    print(f"target robot endpoint 1: {target_robot._endpoints()[0]}")

    # define paramater value: select value for the Z-coord of the segment junction
    # junction refers to endpoint of first segment
    target_robot_junction = target_robot._endpoints()[0]
    coord_param = CoordParamValue(ParamableCoord.Z, target_robot_junction[2])

    settings = GcrbIkSettings()
    ik_target_a = target_robot.t_matrix().A
    ik_target = SE3IkTarget(ik_target_a)

    robot = ConstantCurvatureCR(
        [
            ConstantCurvatureSegment(1, 1, 1, is_extensible=True),
            ConstantCurvatureSegment(1, 1, 1, is_extensible=True),
        ]
    )

    solver = GcrbSolver2(robot, settings, ik_target, coord_param)
    solver.solve()

    soln1 = solver.cr.pose_for_target(ik_target.target_type)
    soln2 = solver.cr2.pose_for_target(ik_target.target_type)

    diff1 = np.linalg.norm(soln1 - target_pose)
    diff2 = np.linalg.norm(soln2 - target_pose)

    logger.info(f"soln1 error: {diff1}")
    logger.info(f"soln2 error: {diff2}")
    if diff1 < 1e-6 and diff2 < 1e-6:
        logger.info("GCRB base case passed")
    else:
        if diff1 < 1e-6:
            logger.error("GCRB base yielded valid soln 1, but not soln 2")
        elif diff2 < 1e-6:
            logger.error("GCRB base yielded valid soln 2, but not soln 1")
        else:
            logger.error("GCRB base case failed (two invalid solutions)")

    if plot:
        draw_tdcr(
            target_robot.as_discrete_curve(pts_per_seg=10),
            TDCRPlotterSettings(plot_title="GCRB (Base Case): Target Robot"),
        )
        draw_tdcr(
            solver.cr.as_discrete_curve(pts_per_seg=10),
            TDCRPlotterSettings(plot_title="GCRB (Base Case): Result Robot 1"),
        )
        draw_tdcr(
            solver.cr2.as_discrete_curve(pts_per_seg=10),
            TDCRPlotterSettings(plot_title="GCRB (Base Case): Result Robot 2"),
        )
        plt.show()


def paper_provided_test():
    # the destination pose, as defined in position + unit quaternion in the paper
    pt = np.array([2.64, 0.92, -0.26])
    qt = np.array([0.87, 0.13, -0.27, 0.4])

    robot = ConstantCurvatureCR(
        [
            ConstantCurvatureSegment(1, 1, 1, is_extensible=True),
            ConstantCurvatureSegment(1, 1, 1, is_extensible=True),
        ]
    )

    r3 = uq_to_so3(qt)

    pose = np.eye(4)
    pose[:3, :3] = r3
    pose[:3, 3] = pt

    ik_target = SE3IkTarget(pose)
    settings = GcrbIkSettings()

    solver = GcrbSolver2(
        robot, settings, ik_target, CoordParamValue(ParamableCoord.Z, 3)
    )

    solver.solve()

    print(solver.cr._endpoints())
    print(solver.cr2._endpoints())
    print(f"desired junction: {[1.4, -3.8, -3]}")


def singularity_test(logger):
    """
    three kinds of singularities to test:
    1. when kappa is 1, there is no rotation at al
    2. when mu and lambda are zero, there is only rotation about the z-axis
    3. mu is zero and there is no rotation about the y-axis
    """
    pose = np.eye(4)
    position = np.array([0, 0, 5])
    pose[:3, 3] = position

    robot = ConstantCurvatureCR(
        [
            ConstantCurvatureSegment(1, 1, 1, is_extensible=True),
            ConstantCurvatureSegment(1, 1, 1, is_extensible=True),
        ]
    )

    ik_target = SE3IkTarget(pose)
    settings = GcrbIkSettings()

    solver = GcrbSolver2(
        robot, settings, ik_target, CoordParamValue(ParamableCoord.Z, 2.5)
    )
    solver.solve()

    state = solver.cr.state_vector()
    assert np.isclose(state, [0, 0, 2.5, 0, 0, 2.5]).all()

    logger.info("Rotation no-op test passed")

    # try rotation only about z-axis
    pose = np.eye(4)
    position = np.array([0, 0, 5])
    pose[:3, 3] = position
    pose[:3, :3] = SO3.Rz(pi / 2).A

    ik_target = SE3IkTarget(pose)
    settings = GcrbIkSettings()

    solver = GcrbSolver2(
        robot, settings, ik_target, CoordParamValue(ParamableCoord.Z, 2.5)
    )
    solver.solve()

    soln1 = solver.cr.pose_for_target(ik_target.target_type)
    soln2 = solver.cr2.pose_for_target(ik_target.target_type)

    target_pose = se3_to_pose(pose)
    diff1 = np.linalg.norm(soln1 - target_pose)
    diff2 = np.linalg.norm(soln2 - target_pose)

    logger.info(f"soln1 error: {diff1}")
    logger.info(f"soln2 error: {diff2}")
    if diff1 < 1e-6 and diff2 < 1e-6:
        logger.info("GCRB z-axis rotation case passed")
    else:
        if diff1 < 1e-6:
            logger.error("GCRB z-axis rotation yielded valid soln 1, but not soln 2")
        elif diff2 < 1e-6:
            logger.error("GCRB z-axis rotation yielded valid soln 2, but not soln 1")
        else:
            logger.error("GCRB z-axis rotation case failed (two invalid solutions)")

    logger.info("Rotation about z-axis test passed")

    # try rotation only about y-axis (TODO)
    segment1 = ConstantCurvatureSegment(1 / 0.1, 0.5, 0.05, is_extensible=True)
    segment2 = ConstantCurvatureSegment(
        1 / (2.5 * 0.05 / pi), pi / 2, 0.05, is_extensible=True
    )
    target_robot = ConstantCurvatureCR([segment1, segment2])
    target_pose = target_robot.pose_vector()

    target_robot_junction = target_robot._endpoints()[0]
    coord_param = CoordParamValue(ParamableCoord.Z, target_robot_junction[2])

    settings = GcrbIkSettings()
    ik_target = SE3IkTarget(target_robot.t_matrix().A)

    robot = ConstantCurvatureCR(
        [
            ConstantCurvatureSegment(1, 1, 1, is_extensible=True),
            ConstantCurvatureSegment(1, 1, 1, is_extensible=True),
        ]
    )

    solver = GcrbSolver2(robot, settings, ik_target, coord_param)
    solver.solve()

    soln1 = solver.cr.pose_for_target(ik_target.target_type)
    soln2 = solver.cr2.pose_for_target(ik_target.target_type)

    diff1 = np.linalg.norm(soln1 - target_pose)
    diff2 = np.linalg.norm(soln2 - target_pose)

    logger.info(f"soln1 error: {diff1}")
    logger.info(f"soln2 error: {diff2}")
    if diff1 < 1e-6 and diff2 < 1e-6:
        logger.info("GCRB singular mu case passed")
    else:
        if diff1 < 1e-6:
            logger.error("GCRB singular mu yielded valid soln 1, but not soln 2")
        elif diff2 < 1e-6:
            logger.error("GCRB singular mu yielded valid soln 2, but not soln 1")
        else:
            logger.error("GCRB singular mu case failed (two invalid solutions)")


def test_curvature_from_junction():
    segment1 = ConstantCurvatureSegment(1 / 0.1, pi / 2, 0.05, is_extensible=True)
    segment2 = ConstantCurvatureSegment(1 / 0.05, pi / 2, 0.05, is_extensible=True)
    target_robot = ConstantCurvatureCR([segment1, segment2])

    # define paramater value: select value for the Z-coord of the segment junction
    # junction refers to endpoint of first segment
    target_robot_junction = target_robot._endpoints()[0]
    coord_param = CoordParamValue(ParamableCoord.Z, target_robot_junction[2])

    settings = GcrbIkSettings()
    ik_target = SE3IkTarget(target_robot.t_matrix().A)

    robot = ConstantCurvatureCR(
        [
            ConstantCurvatureSegment(1, 1, 1, is_extensible=True),
            ConstantCurvatureSegment(1, 1, 1, is_extensible=True),
        ]
    )

    solver = GcrbSolver2(robot, settings, ik_target, coord_param)

    junction = target_robot._endpoints()[0]
    solution_config = np.hstack(solver._config_from_junction(junction))
    expected_solution = target_robot.state_vector()
    assert np.isclose(solution_config, expected_solution).all()


def run(loglevel=logging.INFO, plot=False):
    logging.basicConfig(level=loglevel)
    logger = logging.getLogger(__name__)
    test_base_case(logger, plot)
    # paper_provided_test()
    singularity_test(logger)
    test_curvature_from_junction()
