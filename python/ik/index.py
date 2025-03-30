from enum import Enum

from ik.solvers import (
    NewtonRaphsonIkSolver,
    NeppalliIkSolver,
    GcrbSolver2,
    MicsSolver,
    FabrikcIkSolver,
)


class IkRobotType(Enum):
    """
    an index of the different kind of Robots that can be used in an IK problem.

    Used to help keep track of which solvers can be used with which robots,
    and keeps the testing pipeline easier
    """

    TwoSegExtensible = "2seg_ext"
    TwoSegInExtensible = "2seg_nonext"
    ThreeSegExtensible = "3seg_ext"
    ThreeSegInExtensible = "3seg_nonext"

    def as_filepath_params(self):
        """
        utility function useful in tests/eval.py for generating the filepath
        """
        match self:
            case IkRobotType.TwoSegExtensible:
                return (2, True)
            case IkRobotType.TwoSegInExtensible:
                return (2, False)
            case IkRobotType.ThreeSegExtensible:
                return (3, True)
            case IkRobotType.ThreeSegInExtensible:
                return (3, False)


class IkSolverType(Enum):
    NewtonRaphson = "nr"
    Neppalli = "neppalli"
    Gcrb = "gcrb"
    Fabrikc = "fabrikc"
    Mics = "mics"

    def solver_class(self):
        match self:
            case IkSolverType.NewtonRaphson:
                return NewtonRaphsonIkSolver
            case IkSolverType.Neppalli:
                return NeppalliIkSolver
            case IkSolverType.Gcrb:
                return GcrbSolver2
            case IkSolverType.Fabrikc:
                return FabrikcIkSolver
            case IkSolverType.Mics:
                return MicsSolver
            case _:
                raise NotImplementedError(f"Solver {self} not implemented")

    def is_analytical(self):
        return self in [IkSolverType.Gcrb, IkSolverType.Neppalli]

    def is_numerical(self):
        return not self.is_analytical()

    def applicable_robots(self):
        """
        for testing: mapping of robots that can be used with each solver
        """
        match self:
            case IkSolverType.NewtonRaphson:
                return [
                    IkRobotType.TwoSegInExtensible,
                    IkRobotType.ThreeSegInExtensible,
                ]
            case IkSolverType.Neppalli:
                return [IkRobotType.TwoSegExtensible, IkRobotType.ThreeSegExtensible]
            case IkSolverType.Gcrb:
                return [IkRobotType.TwoSegExtensible]
            case IkSolverType.Fabrikc:
                return [
                    IkRobotType.TwoSegInExtensible,
                    IkRobotType.ThreeSegInExtensible,
                ]
            case IkSolverType.Mics:
                return [
                    IkRobotType.TwoSegInExtensible,
                    IkRobotType.ThreeSegInExtensible,
                ]
            case _:
                raise NotImplementedError(f"Solver {self} not implemented")
