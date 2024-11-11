"""
Tests for the GCRB analytic IK solver.


"""

from math import pi
import numpy as np
import matplotlib.pyplot as plt
from spatialmath import UnitQuaternion

from common.robot import ConstantCurvatureCR, ConstantCurvatureSegment
from common.coordinates import CoordParamValue, ParamableCoord
from common.types import TDCRPlotterSettings

from ik.target import SE3IkTarget
from ik.solvers.gcrb.gcrb_solver import GcrbSolver2, GcrbIkSettings


from plotter.tdcr import draw_tdcr

import logging


def test_base_case(logger, plot=False):
    """
    a two-segment, unrandomized base case
    """

    segment1 = ConstantCurvatureSegment(1 / 0.1, pi / 6, 0.05, is_extensible=True)
    segment2 = ConstantCurvatureSegment(1 / 0.05, pi / 2, 0.05, is_extensible=True)
    target_robot = ConstantCurvatureCR([segment1, segment2])
    target_pose = target_robot.pose_vector()

    print(f"target robot endpoint 1: {target_robot._endpoints()[0]}")

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
    solver.solve()

    soln1 = solver.cr.pose_for_target(ik_target.target_type)
    soln2 = solver.cr2.pose_for_target(ik_target.target_type)

    # print(f"soln1 endpoint 1: {solver.cr._endpoints()[0]}")
    # print(f"soln2 endpoint 1: {solver.cr2._endpoints()[0]}")

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
    pt = np.array([2.64, 0.92, -0.26])
    qt = UnitQuaternion(np.array([0.87, 0.13, -0.27, 0.4]))

    robot = ConstantCurvatureCR(
        [
            ConstantCurvatureSegment(1, 1, 1, is_extensible=True),
            ConstantCurvatureSegment(1, 1, 1, is_extensible=True),
        ]
    )

    r3 = qt.R

    pose = np.eye(4)
    pose[:3, :3] = r3
    pose[:3, 3] = pt

    ik_target = SE3IkTarget(pose)
    settings = GcrbIkSettings()

    solver = GcrbSolver2(
        robot, settings, ik_target, CoordParamValue(ParamableCoord.Z, -3)
    )

    solver.solve()

    print(solver.cr._endpoints())
    print(solver.cr2._endpoints())
    print(f"desired junction: {[1.4, -3.8, -3]}")


def singularity_test():
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


def run(loglevel=logging.INFO, plot=False):
    logging.basicConfig(level=loglevel)
    logger = logging.getLogger(__name__)
    test_base_case(logger, plot)
    # paper_provided_test()
    # singularity_test()
