from common.robot import ConstantCurvatureCR, ConstantCurvatureSegment
from ik.target import IkTarget
from ik.solvers.base_solver import CcIkSolver, CcIkSettings

from plotter.tdcr import draw_tdcr, TDCRPlotterSettings

from tests.generation import perturbation
from tests.generation.uniform import UNIFORM_TEST_NAME

from copy import deepcopy
from dataclasses import dataclass, asdict, make_dataclass
import json


@dataclass
class IkTestResult:
    success: bool
    exec_time: float
    iter_count: int

    solution_state: ConstantCurvatureCR

    pos_error: float
    orientation_error: float

    def as_dict(self):
        base = asdict(self)
        base["solution_state"] = self.solution_state.as_dict()
        return base


STARTING_POSITION_VARS = [UNIFORM_TEST_NAME] + perturbation.STARTING_POSITION_VARS

_IkTestDataclass = make_dataclass(
    "IkTestCase",
    [("target_robot", ConstantCurvatureCR)]
    + [(name, ConstantCurvatureCR) for name in STARTING_POSITION_VARS],
)


class IkTestCase(_IkTestDataclass):
    @classmethod
    def from_dict(cls, data: dict):
        target_robot = ConstantCurvatureCR(
            [ConstantCurvatureSegment(**seg) for seg in data["target_robot"]]
        )

        starting_positions = {
            type: [ConstantCurvatureCR(**seg) for seg in data[type]]
            for type in STARTING_POSITION_VARS
        }

        return cls(target_robot=target_robot, **starting_positions)

    def as_dict(self):
        base = asdict(self)
        base["target_robot"] = self.target_robot.as_dict()
        for type in STARTING_POSITION_VARS:
            base[type] = getattr(self, type).as_dict()
        return base

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
        pos_error, orientation_error = solver.get_errors()

        return IkTestResult(
            result.is_success,
            solver.exec_time,
            iter_count,
            pos_error=pos_error,
            orientation_error=orientation_error,
            solution_state=solver.cr,
        )


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
