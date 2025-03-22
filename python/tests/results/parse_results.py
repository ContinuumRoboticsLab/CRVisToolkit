from tests.generation.test_case import import_test_results, IkTestResult
import numpy as np


def eval_success_rate(results: list[IkTestResult]) -> float:
    """
    calculates the success rate of the results
    """
    success_count = sum([result.success for result in results])

    return success_count / len(results)


def eval_execution_time(results: list[IkTestResult]) -> float:
    """
    calculates the mean execution time of the results and the stdev
    """
    exec_times = np.array([result.exec_time for result in results])

    return np.mean(exec_times), np.std(exec_times)


def eval_iteration_counts(results: list[IkTestResult]) -> float:
    """
    calculates the mean iteration count of the results and the stdev
    """
    if results[0].iter_count is None:
        return 0.0, 0.0
    iter_counts = np.array([result.iter_count for result in results])

    return np.mean(iter_counts), np.std(iter_counts)


def eval_success_execution_time(results: list[IkTestResult]) -> float:
    """
    calculates the mean execution time of the results and the stdev, including only
    tests that were succcessful
    """
    exec_times = np.array([result.exec_time for result in results if result.success])

    return np.mean(exec_times), np.std(exec_times)


def eval_success_iteration_counts(results: list[IkTestResult]) -> float:
    """
    calculates the mean iteration count of the results and the stdev, including only
    tests that were succcessful
    """
    if results[0].iter_count is None:
        return 0.0, 0.0
    iter_counts = np.array([result.iter_count for result in results if result.success])

    return np.mean(iter_counts), np.std(iter_counts)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Tool for evaluating json test results"
    )
    parser.add_argument("file", type=str, help="Results to evaluate [json]")

    args = parser.parse_args()

    results = import_test_results(args.file)

    success_rate = eval_success_rate(results) * 100
    met, set = eval_execution_time(results)
    mic, sic = eval_iteration_counts(results)
    smet, sset = eval_success_execution_time(results)
    smic, ssic = eval_success_iteration_counts(results)

    print(
        f"Success rate: {success_rate:.1f}%\n"
        f"Mean execution time: {met:.3E} ± {set:.3E}\n"
        f"Mean iteration count: {mic:.3E} ± {sic:.3E}\n"
        f"Mean execution time (succeeded): {smet:.3E} ± {sset:.3E}\n"
        f"Mean iteration count (succeeded): {smic:.3E} ± {ssic:.3E}"
    )
