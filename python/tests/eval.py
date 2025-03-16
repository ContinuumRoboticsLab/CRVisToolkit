"""
runs all the tests to perform an evaluation of the solver.
"""

from ik.index import IkSolverType
from tests.generation.test_case import import_tests
from tests.runner import TestRunner


def run_json_tests(filepath: str, solver_type: IkSolverType, outfile: str):
    tests = import_tests(filepath)
    solver_class = solver_type.solver_class()

    runner = TestRunner(solver_class, tests[:1000])

    runner.run()
    runner.save_results(outfile)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parser for running json file tests")
    parser.add_argument("file", type=str, help="Task to run [json]")
    parser.add_argument(
        "-s", "--solver", type=str, required=False, help="Solver to test [solver]"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=False,
        help="Output file for test results [output]",
    )

    args = parser.parse_args()
    if args.solver:
        solver = IkSolverType(args.solver)
    else:
        solver = IkSolverType.NewtonRaphson

    if args.output:
        output = args.output
    else:
        output = "results.json"

    run_json_tests(args.file, solver, output)
