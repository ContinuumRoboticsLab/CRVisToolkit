from common.robot import ConstantCurvatureCR, ConstantCurvatureSegment
from ik.target import IkTarget
from ik.solvers.base_solver import CcIkSolver, CcIkSettings


from tests.generation import perturbation
from tests.generation.uniform import UNIFORM_TEST_NAME

from dataclasses import dataclass, asdict, make_dataclass
from typing import Optional
import json
import gzip
from copy import copy


STARTING_POSITION_VARS = [UNIFORM_TEST_NAME] + perturbation.STARTING_POSITION_VARS


@dataclass
class IkTestResult:
    """
    represents the results of executing an IK solver on a single IK target from a single
    starting robot position
    """

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


_IkTestSetResultFields = make_dataclass(
    "_IkTestSetResult",
    [("target_state", ConstantCurvatureCR)]
    + [(varname, Optional[IkTestResult], None) for varname in STARTING_POSITION_VARS],
)


@dataclass
class IkTestSetResult(_IkTestSetResultFields):
    """
    represents the results of executing an IK solver on a single IK target from multiple
    starting robot positions (i.e. a full test "set")
    """

    def as_dict(self):
        res = dict()
        res["target_state"] = self.target_state.as_dict()
        for varname in STARTING_POSITION_VARS:
            if getattr(self, varname) is not None:
                res[varname] = getattr(self, varname).as_dict()
        return res


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
            type: ConstantCurvatureCR(
                [ConstantCurvatureSegment(**seg) for seg in data[type]]
            )
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

    def _solve_single_starter(self, solver_class, settings, ik_target, starting_robot):
        solver = solver_class(starting_robot, settings, ik_target)
        result = solver.solve()

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

    def solve_with_solver(
        self,
        solver_class: type[CcIkSolver],
        settings: CcIkSettings,
        target_class: type[IkTarget],
    ) -> IkTestSetResult:
        ik_target = self.as_target_type(target_class)

        # run for same target using each starting position
        results = dict()

        starting_positions = copy(STARTING_POSITION_VARS)

        # if solver does not need initial guess, keep just one starting position
        if not solver_class.requires_init_guess:
            starting_positions = starting_positions[:1]

        for varname in starting_positions:
            starting_robot = getattr(self, varname)

            results[varname] = self._solve_single_starter(
                solver_class, settings, ik_target, starting_robot
            )

        return IkTestSetResult(**results, target_state=self.target_robot)


"""
utility function for importing serialized JSON files
"""


def import_tests(path: str, decompress: bool) -> list[IkTestCase]:
    if decompress:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)

    else:
        with open(path, "r") as f:
            data = json.load(f)

    return [IkTestCase.from_dict(test) for test in data]


def import_test_results(path: str) -> list[IkTestSetResult]:
    with open(path, "r") as f:
        data = json.load(f)

    return [IkTestSetResult(**result) for result in data]


def get_test_results(
    path: str, use_starters: list[str] | None = None
) -> list[IkTestResult]:
    """
    returns each individual starting position/target position pair as a single test, and
    outputs the results of the test. Used for determining performance metrics
    """
    test_set_results = import_test_results(path)
    test_results = []
    if use_starters is None:
        use_starters = STARTING_POSITION_VARS
    for test_set in test_set_results:
        for varname in use_starters:
            test_result = getattr(test_set, varname)
            if test_result is None:
                continue
            elif not isinstance(test_result, IkTestResult):
                test_results.append(IkTestResult(**test_result))
            else:
                test_results.append(test_result)
    return test_results
