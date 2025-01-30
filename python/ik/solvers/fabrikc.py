from common.robot import ConstantCurvatureCR

from ik.solvers.base_solver import CcIkSettings, CcIkSolver


class FabrikcIkSolver(CcIkSolver):
    def __init__(self, robot: ConstantCurvatureCR, settings: CcIkSettings):
        super().__init__(robot, settings)

    def _get_error(self):
        pass

    def _perform_forward_reaching(self):
        pass

    def solve(self):
        iternum = 0
        while (
            self._get_error() > self.settings.orientation_tolerance
            and iternum < self.settings.max_iter
        ):
            self._perform_forward_reaching()
            self._perform_backward_reaching()
            iternum += 1

        # cleanup after FABRIKc algorithm
