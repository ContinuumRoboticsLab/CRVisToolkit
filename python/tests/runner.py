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
            try:
                test_case_i = self.generator.generate_case()

                result, execution_time = test_case_i.solve_with_solver(
                    self.solver_class, self.settings, self.target_type, debug_mode
                )

                if result.is_success:
                    num_success += 1

                avg_execution = (avg_execution * i + execution_time) / (i + 1)
            except Exception as e:
                print(f"Error in iteration {i}: {e}")

        print(f"Success rate: {num_success / n}")
        print(f"Execution time: {avg_execution}")


class MultiSolverTestRunner:
    def __init__(
        self,
        generator: UniformDistributionGenerator,
        solver_classes: list[type[CcIkSolver]],
        settings: list[CcIkSettings],
        target_type: list[type[IkTargetType]],
        num_segs: int,
    ):
        self.generator = generator
        self.solver_classes = solver_classes
        self.settings = settings
        self.target_type = target_type
        self.num_segs = num_segs

    def run(self, n: int, debug_mode: bool = False) -> tuple[list[float], list[float]]:
        assert len(self.solver_classes) == len(self.settings) == len(self.target_type)

        success_counts = [0] * len(self.solver_classes)
        avg_execution_times = [0] * len(self.solver_classes)

        for i in tqdm(range(n)):
            test_case_i = self.generator.generate_case()

            for j, (solver_class, settings, target_type) in enumerate(
                zip(self.solver_classes, self.settings, self.target_type)
            ):
                try:
                    result, execution_time = test_case_i.solve_with_solver(
                        solver_class, settings, target_type, debug_mode
                    )

                    if result.is_success:
                        success_counts[j] += 1 / n

                    avg_execution_times[j] = (
                        avg_execution_times[j] * i + execution_time
                    ) / (i + 1)

                except Exception as e:
                    print(f"Error in iteration {i}: {e}")
                    # raise e

        print("Results:")
        for j, (success_count, avg_execution_time) in enumerate(
            zip(success_counts, avg_execution_times)
        ):
            print(f"Solver {self.solver_classes[j]}:")
            print("Success rate: {0:.2f}%".format(success_count * 100))
            print(f"Execution time: {avg_execution_time}\n")

        return success_counts, avg_execution_times
