"""
Test Cases for the Neppalli Inverse Kinematics Solver

Since this solver is closed-form, different test cases use different
numbers of segments (fixed) and generate the target robot randomly.

Since for all targets, endpoints are generated using an existing robot instance,
the set of target endpoints is guaranteed to reachable by a valid configuration.
"""

from math import pi
import numpy as np
from common.robot import ConstantCurvatureCR, ConstantCurvatureSegment

from ik.solvers.neppalli import NeppalliIkSettings, NeppalliIkSolver, NeppalliIktarget

from plotter.tdcr import draw_tdcr

import logging

MAX_PHI = 2 * pi

MAX_RADIUS = 0.5

MIN_LENGTH = 0.01
LENGTH_RANGE = 0.1


def test_base_case(logger, plot=False):
    """
    the single segment base case
    """
    segment = ConstantCurvatureSegment(1 / 0.14, pi / 3, 0.05)
    target_robot = ConstantCurvatureCR([segment])
    target_pose = target_robot.pose_vector()

    ik_target = NeppalliIktarget(target_robot._endpoints())
    settings = NeppalliIkSettings()

    solver = NeppalliIkSolver(
        ConstantCurvatureCR([ConstantCurvatureSegment()]), settings, ik_target
    )
    result = solver.solve()

    result_pose = solver.get_pose()
    diff = np.linalg.norm(result_pose - target_pose)

    if result.is_success and diff < 1e-6:
        logger.info("Neppalli base case passed")
    else:
        logger.error("Neppalli base case failed")

    if plot:
        draw_tdcr(target_robot.as_discrete_curve(pts_per_seg=10))
        draw_tdcr(solver.cr.as_discrete_curve(pts_per_seg=10))


def test_neppalli_manyseg(logger, num_segs, plot=False):
    """
    Generic test case, number of segments is set by num_segs
    """

    segments = [
        ConstantCurvatureSegment(
            kappa=1 / np.random.uniform(0, MAX_RADIUS),
            phi=np.random.uniform(0, MAX_PHI),
            length=np.random.uniform(MIN_LENGTH, MIN_LENGTH + LENGTH_RANGE),
        )
        for _ in range(num_segs)
    ]
    target_robot = ConstantCurvatureCR(segments)
    target_pose = target_robot.pose_vector()

    ik_target = NeppalliIktarget(target_robot._endpoints())

    settings = NeppalliIkSettings()

    solver = NeppalliIkSolver(
        ConstantCurvatureCR([ConstantCurvatureSegment() for _ in range(num_segs)]),
        settings,
        ik_target,
    )

    result = solver.solve()

    result_pose = solver.get_pose()
    diff = np.linalg.norm(result_pose - target_pose)

    if result.is_success and diff < 1e-6:
        logger.info(f"Neppalli test case for {num_segs} segments passed")
    else:
        logger.error(
            f"Neppalli test case for {num_segs} segments failed: diff = {diff}"
        )

    if plot:
        draw_tdcr(target_robot.as_discrete_curve(pts_per_seg=10))
        draw_tdcr(solver.cr.as_discrete_curve(pts_per_seg=10))


def run(
    loglevel=logging.INFO,
    num_segs: list[int] = [1, 2, 3, 5],
    randseed: int | None = None,
    plot=False,
):
    format = "%(levelname)s: %(message)s"
    logging.basicConfig(level=loglevel, format=format)
    logger = logging.getLogger(__name__)

    if randseed:
        np.random.seed(randseed)

    test_base_case(logger, plot=plot)

    for n in num_segs:
        test_neppalli_manyseg(logger, n, plot=plot)
