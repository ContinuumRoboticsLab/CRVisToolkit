from tests.generation.base_generator import IkTestGenerator


class PerturbedRobotGenerator(IkTestGenerator):
    """
    The perturbed robot generator generates test cases by generating a starter robot,
    then perturbing all arc parameters of all segments using a normalized sum of normals
    distribution.

    The two gaussian peaks of the probability distribution are equidistant from zero and
    have the same stdev.

    The aim of this test case generator is to generate test cases that have target robots
    defined by some perturbation to the starting robot in all of it's configuration parameters.
    This helps provide additional information, especially where numerical solvers are concerned,
    providing more extensive data about the conditions under which a numerical solver can be
    expected to converge.
    """
