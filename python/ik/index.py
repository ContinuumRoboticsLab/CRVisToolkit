from enum import Enum

from ik.solvers import (
    NewtonRaphsonIkSolver,
    NeppalliIkSolver,
    GcrbSolver2,
    MicsSolver,
    FabrikcIkSolver,
)


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
