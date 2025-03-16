"""
the script run to generate the various forward-kinematics based inverse-kinematics test
cases. This script generates N cases for each of the following test configurations:
- 2-segment extensible robots
- 2-segment non-extensible robots
- 3-segment extensible robots
- 3-segment non-extensible robots
"""

from tests.generation.uniform import UniformDistributionGenerator
from tests.generation.test_case import IkTestCase

from common.robot import RobotSegmentLimits

import os
import json

NUM_CASES = 10000
SEED = 0

OUTPUT_DIRECTORY = "tests/export"
FILENAME_FORMATTER = "tests_{n}seg_{prefix}extensible.json"


def generate_tests(
    n: int, extensible: bool, limit_kwargs: dict, seed
) -> list[IkTestCase]:
    segment_limits = RobotSegmentLimits(**limit_kwargs, is_extensible=extensible)
    generator = UniformDistributionGenerator(n, segment_limits, seed=seed)
    return [generator.generate_case() for _ in range(NUM_CASES)]


def tests_to_str(tests: list[IkTestCase]) -> str:
    return json.dumps([test.as_dict() for test in tests], indent=4)


if __name__ == "__main__":
    # TODO for user: add to this dict as necessary, unspecified values will be filled in with defaults
    limit_kwargs = {}
    for n in [2, 3]:
        for extensible in [True, False]:
            tests = generate_tests(n, extensible, limit_kwargs, SEED)

            prefix = "" if extensible else "in"
            filename = FILENAME_FORMATTER.format(n=n, prefix=prefix)
            path = os.path.join(OUTPUT_DIRECTORY, filename)
            with open(path, "w") as f:
                f.write(tests_to_str(tests))
                print(f"wrote {NUM_CASES} tests to {filename}")
