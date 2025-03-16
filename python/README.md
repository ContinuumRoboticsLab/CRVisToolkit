# Overview
The full Python toolkit for CR, including visualization, forward kineamatics and multiple inverse kinematics solvers.

## Dependencies
The library has minimal dependencies and can be seen in the `pyproject.toml`. Dependence on external libraries that do not contribute to performance were kept to a minimum. Installation with poetry is recommended.

## Module organization

### `common`
Includes structures and utility functions that are used by multiple Inverse-Kinematics solvers and/or are common functions that should be accesible within other modules of code. Most notably, the `robot.py` file defines the classes used to describe a Constant Curvature Robot.

### `examples`
Sample usage of code in the repository (to be added to later).

### `ik`
The inverse kinematics module. Contains generic classes for inverse kinematics solvers and implementations of specific solvers. To date, the Neppalli, Garrigas-Casanovas (abbreviated to GCRB in code), MICS, and FABRIKc solvers have been implemented. Additionally, a numerical Newton-Raphson solver for the inverse kinematics has been implemented.

### `plotter`
Utilities for plotting continuum robots. Can be useful for debugging purposes.

### `tests`
Code for generation and running of tests. Includes ways of generating random tests of different styles
and evaluating/banchmarking solver performance across a large set of test cases.