"""
the script run to generate the various forward-kinematics based inverse-kinematics test
cases. This script generates N cases for each of the following test configurations:
- 2-segment extensible robots
- 2-segment non-extensible robots
- 3-segment extensible robots
- 3-segment non-extensible robots
"""

from tests.generation.uniform import UniformDistributionGenerator
from tests.generation.perturbation import PerturbedRobotGenerator
from tests.generation.test_case import IkTestCase

from common.robot import RobotSegmentLimits

import os
import json

NUM_CASES = 10000
SEED = 0

OUTPUT_DIRECTORY = "tests/export"
NORMAL_FILENAME_FORMATTER = "normal_{n}seg_{prefix}ext.json"
PERTURBED_FILENAME_FORMATTER = "perturbed_{magnitude}_{n}seg_{prefix}ext.json"

PERTURBATION_VALUES = [0.01, 0.05, 0.1, 0.2]

TESTCASE_TYPES = ["normal"] + [
    f"perturbed_{stdev:.2f}".replace(".", "_") for stdev in PERTURBATION_VALUES
]


def generate_normal_tests(
    n: int, extensible: bool, limit_kwargs: dict, seed
) -> list[IkTestCase]:
    segment_limits = RobotSegmentLimits(**limit_kwargs, is_extensible=extensible)
    generator = UniformDistributionGenerator(n, segment_limits, seed=seed)
    return [generator.generate_case() for _ in range(NUM_CASES)]


def generate_perturbed_tests(
    n: int, extensible: bool, limit_kwargs: dict, seed, stdev_percentage: float
) -> list[IkTestCase]:
    segment_limits = RobotSegmentLimits(**limit_kwargs, is_extensible=extensible)
    generator = PerturbedRobotGenerator(
        n, segment_limits, seed=seed, stdev_percentage=stdev_percentage
    )
    return [generator.generate_case() for _ in range(NUM_CASES)]


def tests_to_str(tests: list[IkTestCase]) -> str:
    return json.dumps([test.as_dict() for test in tests], indent=4)


def generate_normal_tests_to_json(
    filepath: str, n: int, extensible: bool, limit_kwargs: dict, seed
) -> list[IkTestCase]:
    cases = generate_normal_tests(n, extensible, limit_kwargs, seed)
    serialized = tests_to_str(cases)

    with open(filepath, "w") as f:
        f.write(serialized)
    print(f"wrote {NUM_CASES} tests to {filepath}")


def generate_perturbed_tests_to_json(
    filepath: str,
    n: int,
    extensible: bool,
    limit_kwargs: dict,
    seed,
    stdev_percentage: float,
) -> list[IkTestCase]:
    cases = generate_perturbed_tests(
        n, extensible, limit_kwargs, seed, stdev_percentage
    )
    serialized = tests_to_str(cases)

    with open(filepath, "w") as f:
        f.write(serialized)
    print(f"wrote {NUM_CASES} tests to {filepath}")


if __name__ == "__main__":
    # TODO for user: add to this dict as necessary, unspecified values will be filled in with defaults
    limit_kwargs = {}
    for n in [2, 3]:
        for extensible in [True, False]:
            prefix = "" if extensible else "in"

            normal_filename = NORMAL_FILENAME_FORMATTER.format(n=n, prefix=prefix)
            normal_filepath = os.path.join(OUTPUT_DIRECTORY, normal_filename)

            generate_normal_tests_to_json(
                normal_filepath, n, extensible, limit_kwargs, seed=SEED
            )

            for stdev in PERTURBATION_VALUES:
                perturbed_filename = PERTURBED_FILENAME_FORMATTER.format(
                    n=n, prefix=prefix, magnitude=f"{stdev:.2f}".replace(".", "_")
                )
                perturbed_filepath = os.path.join(OUTPUT_DIRECTORY, perturbed_filename)

                generate_perturbed_tests_to_json(
                    perturbed_filepath,
                    n,
                    extensible,
                    limit_kwargs,
                    seed=SEED,
                    stdev_percentage=stdev,
                )
