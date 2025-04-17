# Overview
The full Python toolkit for CR, including visualization, forward kineamatics and multiple inverse kinematics solvers, and a testing pipeline for inverse kinematics algorithms.

## Dependencies
The library has minimal dependencies and can be seen in the `pyproject.toml`. Dependence on external libraries that do not contribute to performance were kept to a minimum. Installation with poetry is recommended.

## Testing Pipeline
The entire testing pipeline can be run in steps by running the correct python files with the correct arguments. The pipeline works as follows: first, a number of test cases are randomly generated and saved as zipped JSON files. Then, the test cases in these JSON files are parsed, and the implemented sovlers can be tested across the parsed test cases. The results of each test case are collected and saved in a file. Then, the results file can be parsed to evaluate solver metrics.

To generate the tests, you can run the following command from within the `python` directory:

```
python -m tests.generation.main
```

This will generate 10000 test cases for each of the four "types" of test case: 2-segment inextensible, 2-segment extensible, 3-segment inextensible, and 3-segment extensible. The number and type of test cases generated can easily be modified in the source code found at `tests/generation/main.py`.

To evaluate each the generated test cases with the appropriate solvers:

```
python -m tests.eval -a
```

The `-a` option will run all the unzipped JSON files for all five inverse kinematics solvers, and save their results in the appropriate location in the `tests/results` directory. The results included as part of this repository were evaluated using these same scripts on an M2 Mac Studio.

To then compute the performance metrics from any set of test results, run the following:

```
python -m tests.tesults.parse_results tests/results/nr/res_2seg_inext.json
```

The file path provided can be replaced by any path to a test result JSON file, even if the file is gzip compressed. The output should look as follows:

```
Results for tests/results/nr/res_2seg_inext.json
Ran 600 tests

Success rate: 93.67%
Mean execution time: 8.927E-03 ± 2.570E-02
Mean iteration count: 1.533E+01 ± 4.830E+01
Mean position error: 1.721E-03 ± 1.792E-03
Mean orientation error: 2.198E-04 ± 2.695E-04
Mean execution time (succeeded): 2.261E-03 ± 9.980E-04
Mean iteration count (succeeded): 2.778E+00 ± 1.406E+00
Mean position error (succeeded): 7.937E-04 ± 1.792E-03
Mean orientation error (succeeded): 2.164E-04 ± 2.613E-04
```

## Implementing a Solver
The toolkit was designed with an architecture that allows for easy implementation of other inverse kinematics algorithms. To implement an inverse kinematics algorithm, the following steps must be taken:

1. A subclass of the `IkTarget` class found in `ik/target` must be defined to encapsulate the mathematical definition of the desired end effector state. If one of the provided target classes suits your needs, then an existing target can also be used.

2. A subclass of the `CcIkSolver` class found in `ik/base_solver` must also be implemented. The appropriate methods should be implemented specifically for the inverse kinematics solution you would like to implement. If you are implementing a numerical solver, the subclass `IterativeIkSolver` provides some additional logic which may be helpful in implementing your own algorithm.