"""
runs all the tests to perform an evaluation of the solver.

looks through test case files (all of them or just the ones specified) and saves the
results in the appropriate filepath
"""

import time

from ik.index import IkSolverType
from tests.generation.test_case import import_tests
from tests.runner import TestRunner

from tests.generation.main import TESTCASE_TYPES

TEST_RESULT_DIR = "tests/results"
TEST_RESULT_FILEPATH_FORMATTER = "tests/results/{solver}/{type}_{n}seg_{prefix}ext.json"
TEST_CASE_FORMATTER = "tests/export/{type}_{n}seg_{prefix}ext.json"


def get_solver_test_filepaths(solver_type: IkSolverType):
    """
    returns the filepaths for the test cases that are applicable to the given solver,
    and the destination file paths
    """
    for robot_type in solver_type.applicable_robots():
        n, ext = robot_type.as_filepath_params()

        for test_type in TESTCASE_TYPES:
            yield (
                TEST_CASE_FORMATTER.format(
                    solver=solver_type.name,
                    type=test_type,
                    n=n,
                    prefix="" if ext else "in",
                ),
                TEST_RESULT_FILEPATH_FORMATTER.format(
                    solver=solver_type.value,
                    type=test_type,
                    n=n,
                    prefix="" if ext else "in",
                ),
            )


def run_json_tests(filepath: str, solver_type: IkSolverType, outfile: str):
    tests = import_tests(filepath)
    solver_class = solver_type.solver_class()

    print(f"Running {filepath} tests for {solver_type} solver")

    start = time.time()
    runner = TestRunner(solver_class, tests)

    runner.run()
    duration = time.time() - start
    runner.save_results(outfile)

    print(f"Results saved to {outfile} after {duration:.2f} seconds")


def run_all_for_solver(solver_type: IkSolverType):
    print(f"Running all test files for {solver_type} solver")
    start = time.time()
    for test_filepath, results_filepath in get_solver_test_filepaths(solver_type):
        run_json_tests(test_filepath, solver_type, results_filepath)
    duration = time.time() - start
    print(f"Finished running all tests for {solver_type} after {duration:.2f} seconds")


def run_all():
    print("Running all tests for all solvers")
    start = time.time()
    for solver_type in IkSolverType:
        run_all_for_solver(solver_type)
    duration = time.time() - start
    print(f"Finished running all tests for all solvers after {duration:.2f} seconds")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tool for running json file tests")
    parser.add_argument("file", type=str, help="Task to run [json]")
    parser.add_argument(
        "-s", "--solver", type=str, required=False, help="Solver to test [solver]"
    )
    parser.add_argument(
        "-a",
        "--all",
        required=False,
        action="store_true",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=False,
        help="Output file for test results [output]",
    )

    args = parser.parse_args()

    if args.all:
        run_all()
        exit()

    if args.solver:
        solver = IkSolverType(args.solver)
    else:
        solver = IkSolverType.NewtonRaphson

    if args.output:
        output = args.output
    else:
        output = "results.json"

    run_json_tests(args.file, solver, output)
