"""
runs all the tests to perform an evaluation of the solver.

looks through test case files (all of them or just the ones specified) and saves the
results in the appropriate filepath
"""

import time

from ik.index import IkSolverType
from tests.generation.test_case import import_tests
from tests.runner import TestRunner


TEST_RESULT_DIR = "tests/results"
TEST_RESULT_FILEPATH_FORMATTER = (
    "tests/results/{solver}/res_{n}seg_{prefix}ext.json{gzip_extension}"
)
TEST_CASE_FORMATTER = "tests/export/tests_{n}seg_{prefix}ext.json{gzip_extension}"


def get_solver_test_filepaths(solver_type: IkSolverType, compress_out=True):
    """
    returns the filepaths for the test cases that are applicable to the given solver,
    and the destination file paths
    """
    for robot_type in solver_type.applicable_robots():
        n, ext = robot_type.as_filepath_params()
        gzip_ext = ".gz" if compress_out else ""
        yield (
            TEST_CASE_FORMATTER.format(
                n=n, prefix="" if ext else "in", gzip_extension=gzip_ext
            ),
            TEST_RESULT_FILEPATH_FORMATTER.format(
                solver=solver_type.value,
                n=n,
                prefix="" if ext else "in",
                gzip_extension=gzip_ext,
            ),
        )


def run_json_tests(
    filepath: str, solver_type: IkSolverType, outfile: str, compress_out=True
):
    tests = import_tests(filepath, decompress=compress_out)
    solver_class = solver_type.solver_class()

    print(f"Running {filepath} tests for {solver_type} solver")

    start = time.time()
    runner = TestRunner(solver_class, tests)

    runner.run()
    duration = time.time() - start
    runner.save_results(outfile, compress=compress_out)

    print(f"Results saved to {outfile} after {duration:.2f} seconds")


def run_all_for_solver(solver_type: IkSolverType, compress_out=True):
    print(f"Running all test files for {solver_type} solver")
    start = time.time()
    for test_filepath, results_filepath in get_solver_test_filepaths(
        solver_type, compress_out=compress_out
    ):
        run_json_tests(
            test_filepath, solver_type, results_filepath, compress_out=compress_out
        )
    duration = time.time() - start
    print(
        f"Finished running all tests for {solver_type} after {duration:.2f} seconds\n"
    )


def run_all(compressed_run=True):
    print("Running all tests for all solvers")
    start = time.time()
    for solver_type in IkSolverType:
        if solver_type != IkSolverType.Mics:
            continue
        run_all_for_solver(solver_type, compress_out=compressed_run)
    duration = time.time() - start
    print(f"Finished running all tests for all solvers after {duration:.2f} seconds")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tool for running json file tests")
    parser.add_argument(
        "-f", "--file", required=False, type=str, help="Task to run [json]"
    )
    parser.add_argument(
        "-s", "--solver", type=str, required=False, help="Solver to test [solver]"
    )
    parser.add_argument("-a", "--all", required=False, action="store_true")
    parser.add_argument("-c", "--compressed", required=False, action="store_true")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=False,
        help="Output file for test results [output]",
    )

    args = parser.parse_args()

    if args.file is None and not args.all:
        print("Please provide a file to run or use the --all flag")

    if args.all:
        run_all(args.compressed)
        exit()

    if args.solver:
        solver = IkSolverType(args.solver)
    else:
        solver = IkSolverType.NewtonRaphson

    if args.output:
        output = args.output
    else:
        output = "results.json"

    run_json_tests(args.file, solver, output, args.compressed)
