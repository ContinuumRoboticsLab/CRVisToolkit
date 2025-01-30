from math import pi
from common.robot import ConstantCurvatureCR, ConstantCurvatureSegment
from ik.target import SE3IkTarget
from ik.solvers.mics import MicsSolverSettings, MicsSolver
import logging
from copy import deepcopy

from plotter.tdcr import draw_tdcr, TDCRPlotterSettings
from matplotlib import pyplot as plt


def test_mics_nominations(logger):
    logger.info("**** Test Case 1: three-segment inextensible CR ****")

    seg1 = ConstantCurvatureSegment(1 / 0.14, -0.8 * pi, 0.05)
    seg2 = ConstantCurvatureSegment(1 / 0.06, 0.4 * pi, 0.03)
    seg3 = ConstantCurvatureSegment(1 / 0.065, -pi, 0.035)
    robot = ConstantCurvatureCR([deepcopy(seg1), deepcopy(seg2), deepcopy(seg3)])

    starting_config_plot = robot.as_discrete_curve(pts_per_seg=10)

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
    logger.info(f"{res}")

    soln_plot = solver.cr.as_discrete_curve(pts_per_seg=10)
    solver.cr.set_config(
        solver.mics_starting_point_configurations[solver.converged_starting_point]
    )
    mics_starter_plot = solver.cr.as_discrete_curve(pts_per_seg=10)
    draw_tdcr(
        soln_plot,
        TDCRPlotterSettings(plot_title="MICS base case 1 Solution"),
    )
    draw_tdcr(
        mics_starter_plot,
        TDCRPlotterSettings(plot_title="MICS base case 1 MICS Starter"),
    )
    draw_tdcr(
        target_robot.as_discrete_curve(pts_per_seg=10),
        TDCRPlotterSettings(plot_title="MICS base case 1 Target Robot"),
    )
    draw_tdcr(
        starting_config_plot,
        TDCRPlotterSettings(plot_title="MICS base case 1 Starting Configuration"),
    )
    plt.show()

    # logger.info("**** Test Case 2: three-segment inextensible CR ****")

    # target_seg1 = ConstantCurvatureSegment(1 / 0.05, 0.35 * pi, 0.04)
    # target_seg2 = ConstantCurvatureSegment(1 / 0.09, -0.6 * pi, 0.03)
    # target_seg3 = ConstantCurvatureSegment(1 / 0.08, 0.8 * pi, 0.035)
    # target_robot = ConstantCurvatureCR([target_seg1, target_seg2, target_seg3])

    # seg1 = ConstantCurvatureSegment(1 / 0.14, -0.8 * pi, 0.05)
    # seg2 = ConstantCurvatureSegment(1 / 0.06, 0.4 * pi, 0.03)
    # seg3 = ConstantCurvatureSegment(1 / 0.065, -pi, 0.035)
    # robot = ConstantCurvatureCR([deepcopy(seg1), deepcopy(seg2), deepcopy(seg3)])

    # starting_config_plot = robot.as_discrete_curve(pts_per_seg=10)
    # solver = MicsSolver(robot, settings, target_pose)

    # res = solver.solve()
    # logger.info(
    #     f"number of candidate MICS starters: {len(solver.mics_starting_points)}"
    # )
    # logger.info(f"{res}")
    # breakpoint()

    # soln_plot = solver.cr.as_discrete_curve(pts_per_seg=10)
    # solver.cr.set_config(
    #     solver.mics_starting_point_configurations[solver.converged_starting_point]
    # )
    # mics_starter_plot = solver.cr.as_discrete_curve(pts_per_seg=10)
    # draw_tdcr(
    #     soln_plot,
    #     TDCRPlotterSettings(plot_title="MICS base case 2 Solution"),
    # )
    # draw_tdcr(
    #     mics_starter_plot,
    #     TDCRPlotterSettings(plot_title="MICS base case 2 MICS Starter"),
    # )
    # draw_tdcr(
    #     target_robot.as_discrete_curve(pts_per_seg=10),
    #     TDCRPlotterSettings(plot_title="MICS base case 2 Target Robot"),
    # )
    # draw_tdcr(
    #     starting_config_plot,
    #     TDCRPlotterSettings(plot_title="MICS base case 2 Starting Configuration"),
    # )

    # plt.show()


def run(plot=False, loglevel=logging.INFO):
    log_format = "%(levelname)s: %(message)s"
    logging.basicConfig(format=log_format, level=loglevel)
    logger = logging.getLogger(__name__)
    test_mics_nominations(logger)


if __name__ == "__main__":
    run()
