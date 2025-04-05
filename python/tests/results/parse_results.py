from tests.generation.test_case import get_test_results, IkTestResult
import numpy as np

import os


def _get_test_files(path: str) -> list[str]:
    if os.path.isfile(path):
        return [path]
    elif os.path.isdir(path):
        entries = [
            os.path.join(path, entry)
            for entry in os.listdir(path)
            if entry.endswith(".json")
        ]

        entries.sort()
        return entries
    else:
        raise OSError(f"Path {path} is not a file or directory")


def _split_success_fail(results: list[IkTestResult]) -> list[IkTestResult]:
    """
    filters the results to only include successful tests
    """
    return [result for result in results if result.success], [
        result for result in results if not result.success
    ]


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


def eval_errors(results: list[IkTestResult]) -> float:
    """
    calculates the mean position error of the results and the stdev
    """
    pos_errors = np.array([result.pos_error for result in results])
    orientation_errors = np.array([result.orientation_error for result in results])

    return (
        np.mean(pos_errors),
        np.std(pos_errors),
        np.mean(orientation_errors),
        np.std(orientation_errors),
    )


def parse_and_print_results(
    filepath: str, use_starters: list[str] | None = None
) -> None:
    """
    parse and print a summary of the test results from a single test file
    """
    results = get_test_results(filepath, use_starters)
    print(f"Ran {len(results)} tests\n")

    success_rate = eval_success_rate(results) * 100
    met, set = eval_execution_time(results)
    mic, sic = eval_iteration_counts(results)
    mpe, spe, moe, soe = eval_errors(results)

    # re-evaluate the successful results
    results, _ = _split_success_fail(results)
    smet, sset = eval_execution_time(results)
    smic, ssic = eval_iteration_counts(results)
    smpe, spe, smoe, ssoe = eval_errors(results)

    print(
        f"Success rate: {success_rate:.2f}%\n"
        f"Mean execution time: {met:.3E} ± {set:.3E}\n"
        f"Mean iteration count: {mic:.3E} ± {sic:.3E}\n"
        f"Mean position error: {mpe:.3E} ± {spe:.3E}\n"
        f"Mean orientation error: {moe:.3E} ± {soe:.3E}\n"
        f"Mean execution time (succeeded): {smet:.3E} ± {sset:.3E}\n"
        f"Mean iteration count (succeeded): {smic:.3E} ± {ssic:.3E}\n"
        f"Mean position error (succeeded): {smpe:.3E} ± {spe:.3E}\n"
        f"Mean orientation error (succeeded): {smoe:.3E} ± {ssoe:.3E}\n"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Tool for evaluating json test results"
    )
    parser.add_argument("path", type=str, help="Results to evaluate [json]")
    parser.add_argument(
        "starters", type=str, nargs="*", help="Starting positions to use [json]"
    )

    args = parser.parse_args()

    if args.starters == []:
        args.starters = None

    test_files = _get_test_files(args.path)
    for file in test_files:
        print(f"Results for {file}")
        parse_and_print_results(file, args.starters)
