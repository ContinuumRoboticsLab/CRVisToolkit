"""
the script run to generate the various forward-kinematics based inverse-kinematics test
cases. This script generates N cases for each of the following test configurations:
- 2-segment extensible robots
- 2-segment non-extensible robots
- 3-segment extensible robots
- 3-segment non-extensible robots
"""

from tests.generation.test_case import IkTestCase
from tests.generation.base_generator import IkTestGenerator

from common.robot import RobotSegmentLimits
from tests.generation.perturbation import PERTURBATION_VALUES

import os
import json
import gzip

NUM_CASES = 10000
NUM_UNCOMPRESSED = 100
SEED = 0

OUTPUT_DIRECTORY = "tests/export"
TESTCASE_FILENAME_FORMATTER = "tests_{n}seg_{prefix}ext.json{zip_extension}"


TESTCASE_TYPES = ["normal"] + [
    f"perturbed_{stdev:.2f}".replace(".", "_") for stdev in PERTURBATION_VALUES
]


def generate_tests(
    n: int, extensible: bool, limit_kwargs: dict, seed
) -> list[IkTestCase]:
    segment_limits = RobotSegmentLimits(**limit_kwargs, is_extensible=extensible)
    generator = IkTestGenerator(n, segment_limits, seed=seed)
    return [generator.generate_test_case() for _ in range(NUM_CASES)]


def tests_to_str(tests: list[IkTestCase]) -> str:
    return json.dumps([test.as_dict() for test in tests], indent=4)


def generate_tests_to_json(
    filepath: str,
    compressed_filepath: str,
    n: int,
    extensible: bool,
    limit_kwargs: dict,
    seed,
) -> list[IkTestCase]:
    cases = generate_tests(n, extensible, limit_kwargs, seed)
    uncompressed_serialized = tests_to_str(cases[:NUM_UNCOMPRESSED])

    with open(filepath, "w") as f:
        f.write(uncompressed_serialized)
    print(f"wrote {NUM_UNCOMPRESSED} tests to {filepath}")

    # compress full set and export
    full_serialized = tests_to_str(cases)
    with gzip.open(compressed_filepath, "wt", encoding="utf-8") as f:
        f.write(full_serialized)
    print(f"wrote {NUM_CASES} tests to {compressed_filepath}")


if __name__ == "__main__":
    # TODO for user: add to this dict as necessary, unspecified values will be filled in with defaults
    limit_kwargs = {}
    for n in [2, 3]:
        for extensible in [True, False]:
            prefix = "" if extensible else "in"

            normal_filename = TESTCASE_FILENAME_FORMATTER.format(
                n=n, prefix=prefix, zip_extension=""
            )
            compressed_filename = TESTCASE_FILENAME_FORMATTER.format(
                n=n, prefix=prefix, zip_extension=".gz"
            )
            normal_filepath = os.path.join(OUTPUT_DIRECTORY, normal_filename)
            compressed_filepath = os.path.join(OUTPUT_DIRECTORY, compressed_filename)

            generate_tests_to_json(
                normal_filepath,
                compressed_filepath,
                n,
                extensible,
                limit_kwargs,
                seed=SEED,
            )
