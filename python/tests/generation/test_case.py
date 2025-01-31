import numpy as np
from ik.target import IkTargetType, SE3IkTarget, P3IkTarget, P3Direction
from ik.solvers.base_solver import IkResult, CcIkSolver, CcIkSettings

from plotter.tdcr import draw_tdcr, TDCRPlotterSettings
from matplotlib import pyplot as plt
from copy import deepcopy


class IkTestCase:
    def __init__(self, ik_target, starting_robot):
        self.ik_target = ik_target
        self.starting_robot = starting_robot

    def _get_pose(self) -> np.ndarray[float]:
        # 6x1 state vector
        pose = self.starting_robot.pose_vector()

        return pose

    def as_ik_target_type(self, type: IkTargetType) -> type[IkTargetType]:
        """
        construct an instance of target type provided - allows
        for any `IkTestCase` to be used for any target type
        """

        match type.target_type:
            case IkTargetType.SE3:
                return SE3IkTarget()
            case IkTargetType.P3:
                return P3IkTarget
            case IkTargetType.DIRECTION:
                return P3Direction
            case _:
                raise ValueError("Invalid target type")

    def solve_with_solver(
        self,
        solver_class: type[CcIkSolver],
        settings: CcIkSettings,
        debug_mode: bool = False,
    ) -> tuple[IkResult, float]:
        starter = self.starting_robot.as_discrete_curve(pts_per_seg=10)
        solver = solver_class(deepcopy(self.starting_robot), settings, self.ik_target)
        result = solver.solve()
        if debug_mode:
            # plot solutions
            draw_tdcr(starter, TDCRPlotterSettings(plot_title="Starting Robot"))
            draw_tdcr(
                solver.cr.as_discrete_curve(pts_per_seg=10),
                TDCRPlotterSettings(plot_title="Solved Robot"),
            )
            plt.show()

        return result, solver.exec_time
