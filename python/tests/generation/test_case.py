from common.robot import ConstantCurvatureCR, ConstantCurvatureSegment
from ik.target import IkTarget
from ik.solvers.base_solver import CcIkSolver, CcIkSettings

from plotter.tdcr import draw_tdcr, TDCRPlotterSettings

from copy import deepcopy
from dataclasses import dataclass, asdict
import json


@dataclass
class IkTestResult:
    success: bool
    exec_time: float
    iter_count: int

    def as_dict(self):
        return asdict(self)


class IkTestCase:
    def __init__(self, target_robot, starting_robot):
        self.target_robot = target_robot
        self.starting_robot = starting_robot

    @classmethod
    def from_dict(cls, data: dict):
        starting_robot = ConstantCurvatureCR(
            [ConstantCurvatureSegment(**seg) for seg in data["start_robot"]]
        )

        target_robot = ConstantCurvatureCR(
            [ConstantCurvatureSegment(**seg) for seg in data["target_robot"]]
        )

        return cls(target_robot, starting_robot)

    def as_dict(self):
        return {
            "start_robot": [seg.as_dict() for seg in self.starting_robot.segments],
            "target_robot": [seg.as_dict() for seg in self.target_robot.segments],
            "target_pose": self.target_robot.t_matrix().A.tolist(),
        }

    def as_target_type(self, ik_target_class: type[IkTarget]) -> IkTarget:
        return ik_target_class.from_target_robot(self.target_robot)

    def solve_with_solver(
        self,
        solver_class: type[CcIkSolver],
        settings: CcIkSettings,
        target_class: type[IkTarget],
        debug_mode: bool = False,
    ) -> IkTestResult:
        starter_plot = self.starting_robot.as_discrete_curve(pts_per_seg=10)

        ik_target = self.as_target_type(target_class)

        solver = solver_class(deepcopy(self.starting_robot), settings, ik_target)
        result = solver.solve()
        if debug_mode:
            # plot solutions
            draw_tdcr(starter_plot, TDCRPlotterSettings(plot_title="Starting Robot"))
            draw_tdcr(
                solver.cr.as_discrete_curve(pts_per_seg=10),
                TDCRPlotterSettings(plot_title="Solved Robot"),
            )

        if hasattr(solver, "iter_count"):
            iter_count = solver.iter_count
        else:
            iter_count = None

        return IkTestResult(result.is_success, solver.exec_time, iter_count)


"""
utility function for importing serialized JSON files
"""


def import_tests(path: str) -> list[IkTestCase]:
    with open(path, "r") as f:
        data = json.load(f)

    return [IkTestCase.from_dict(test) for test in data]


def import_test_results(path: str) -> list[IkTestResult]:
    with open(path, "r") as f:
        data = json.load(f)

    return [IkTestResult(**result) for result in data]
