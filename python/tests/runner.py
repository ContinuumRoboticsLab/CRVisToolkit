from ik.target import IkTargetType
from tests.generation.uniform import UniformDistributionGenerator
from ik.solvers.base_solver import CcIkSettings, CcIkSolver

from tqdm import tqdm


class TestRunner:
    def __init__(
        self,
        generator: UniformDistributionGenerator,
        solver_class: type[CcIkSolver],
        settings: CcIkSettings,
        target_type: type[IkTargetType],
        num_segs: int,
    ):
        self.generator = generator
        self.solver_class = solver_class
        self.settings = settings
        self.target_type = target_type
        self.num_segs = num_segs

    def run(self, n: int, debug_mode: bool = False):
        avg_execution = 0
        num_success = 0

        for i in tqdm(range(n)):
            test_case_i = self.generator.generate_case(self.target_type)

            result, execution_time = test_case_i.solve_with_solver(
                self.solver_class, self.settings, debug_mode
            )

            if result.is_success:
                num_success += 1

            avg_execution = (avg_execution * i + execution_time) / (i + 1)

        print(f"Success rate: {num_success / n}")
        print(f"Execution time: {avg_execution}")
