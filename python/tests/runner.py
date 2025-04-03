from ik.target import IkTargetType
from tests.generation.test_case import IkTestCase, IkTestSetResult
from tests.generation.uniform import UniformRobotFactory
from ik.solvers.base_solver import CcIkSettings, CcIkSolver

from tqdm import tqdm
import json


class GenerativeTestRunner:
    def __init__(
        self,
        generator: UniformRobotFactory,
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
            except ValueError as e:
                print(f"Error in iteration {i}: {e}")

        print(f"Success rate: {num_success / n}")
        print(f"Execution time: {avg_execution}")


class MultiGenerativeTestRunner:
    def __init__(
        self,
        generator: UniformRobotFactory,
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

    def run(self, n: int, show_plots: bool = False) -> tuple[list[float], list[float]]:
        assert len(self.solver_classes) == len(self.settings) == len(self.target_type)

        success_counts = [0] * len(self.solver_classes)
        avg_execution_times = [0] * len(self.solver_classes)
        total_num_iterations = [0] * len(self.solver_classes)

        for i in tqdm(range(n)):
            test_case_i = self.generator.generate_case()

            for j, (solver_class, settings, target_type) in enumerate(
                zip(self.solver_classes, self.settings, self.target_type)
            ):
                try:
                    result, execution_time, iter_count = test_case_i.solve_with_solver(
                        solver_class, settings, target_type, show_plots
                    )

                    if result.is_success:
                        success_counts[j] += 1

                    avg_execution_times[j] = (
                        avg_execution_times[j] * i + execution_time
                    ) / (i + 1)

                    if iter_count and result.is_success:
                        total_num_iterations[j] += iter_count

                    from plotter.tdcr import draw_tdcr, TDCRPlotterSettings
                    from matplotlib import pyplot as plt

                    if show_plots:
                        draw_tdcr(
                            test_case_i.target_robot.as_discrete_curve(pts_per_seg=10),
                            TDCRPlotterSettings(plot_title="result"),
                        )
                        plt.show()

                except Exception as e:
                    print(f"Error in iteration {i}: {e}")
                    raise e

        print("Results:")
        for j, (success_count, avg_execution_time, iter_count) in enumerate(
            zip(success_counts, avg_execution_times, total_num_iterations)
        ):
            print(f"Solver {self.solver_classes[j]}:")
            print("Success rate: {0:.2f}%".format(success_count / n * 100))
            print(f"Execution time: {avg_execution_time}")
            print(f"Average number of iterations: {iter_count / success_count}\n")

        return success_counts, avg_execution_times


class TestRunner:
    def __init__(self, solver_class, test_cases: list[IkTestCase]):
        self.solver_class: CcIkSolver = solver_class
        self.test_cases = test_cases
        self.results: list[IkTestSetResult] = []

    def run(self, show_plots: bool = False):
        for i in tqdm(range(len(self.test_cases))):
            test_case = self.test_cases[i]
            try:
                target_type: IkTargetType = self.solver_class.target_type
                ik_target_class = target_type.ik_target_class()
                test_result = test_case.solve_with_solver(
                    self.solver_class,
                    self.solver_class.settings_class(),
                    ik_target_class,
                )
                self.results.append(test_result)

            except Exception as e:
                print(f"Error in test case {i}: {e}")
                raise e

    def save_results(self, path):
        results = [r.as_dict() for r in self.results]
        with open(path, "w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    pass
