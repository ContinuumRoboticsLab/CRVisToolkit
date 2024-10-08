"""
Implementation of the Newton-Rhapson method for inverse kinematics.

this implementation uses a discrete constant curvate representation of a continuum robot
and uses the Newton-Rhapson method to solve for the curvature parameters that constitute
a solution to the IK problem.

The Jacobian is computed using the finite differences method.
"""

import numpy as np

from common.coordinates import CrConfigurationType
from common.robot import ConstantCurvatureCR, ConstantCurvatureSegment
from common.jacobian import jacobian
from ik.solvers.base_solver import IterativeIkSolver, CcIkSettings, IkResult
from plotter.tdcr import draw_tdcr


class NewtonRhapsonIkSettings(CcIkSettings):
    pass


class NewtonRhapsonIkSolver(IterativeIkSolver):
    """
    Implementation of the Newton-Rhapson method for inverse kinematics.

    NOTE: this implementation assumes the robot is in KPL representation

    Notation/Convention
    --------
    theta_i: the solution to the IK problem at current iteration


    Parameters
    ----------
    cr: ContinuousCurvatureCR
        the CR object to solve the IK for.
    settings: CcIkSettings
        the settings for the solver
    initial_condition: np.array[float]
        the initial condition theta_0 for the solver.
        The input should be a 3nx1 array as follows:
        [kappa_i, phi_i, length_i] for each segment i in the CR object
    """

    def __init__(
        self,
        cr: ConstantCurvatureCR,
        settings: NewtonRhapsonIkSettings,
        initial_condition: np.ndarray[float] | list[float],
        target_pose: np.ndarray[float],
        **kwargs,
    ):
        if isinstance(initial_condition, list):
            initial_condition = np.array(initial_condition)

        self.total_dof = sum([seg.n for seg in cr.segments])

        # check if the initial condition is of valid dimensionality
        if initial_condition.shape != (self.total_dof, 1):
            try:
                initial_condition = initial_condition.reshape((self.total_dof, 1))
            except Exception:
                raise ValueError(
                    f"Invalid initial condition. Expected shape: {(self.total_dof, 1)}"
                )

        # make sure we're using the right representation
        if not cr.repr_type == CrConfigurationType.KPL:
            raise ValueError("Only KPL representation is supported for the NR solver")

        self.theta_i = np.reshape(initial_condition, (initial_condition.size, 1))

        super().__init__(
            cr,
            settings,
            initial_condition=initial_condition,
            target_pose=target_pose,
            **kwargs,
        )

    def _prepare_solver(self, *args, **kwargs):
        # prepare method-specific attributes

        # dimensionality of the solution
        self.n = self.total_dof
        # dimensionality of relevant task space
        self.m = 6

    def __compute_jacobian(self):
        """
        compute the Jacobian matrix at the current solution
        returns an (m x n) matrix
        """
        return jacobian(self.cr.pose_vector, self.cr.state_vector())

    def _perform_iteration(self, *args, **kwargs):
        """
        performs a single iteration of:
        theta_(i+1) = theta_i + pinv(J(theta_i)) * twist(theta_i)

        calculation of both the jacobian and twist are done in other
        object methods
        """

        # update the CR internal state
        self._update_cr_configuration()

        j = self.__compute_jacobian()

        diff = self.target_pose - self.cr.pose_vector()
        d_theta = np.linalg.pinv(j) @ diff
        self.theta_i += np.reshape(d_theta, (d_theta.size, 1))

        self.iter_count += 1

    def _update_cr_configuration(self):
        """
        updates the CR object using the current solution
        necessary for performing forward kinematics step
        """

        # validate the parameters
        if not self.theta_i.shape == (self.n, 1):
            try:
                theta_i = self.theta_i.reshape((self.n, 1))
            except Exception:
                raise ValueError(f"Invalid theta_i shape: {self.theta_i.shape}")
        else:
            theta_i = self.theta_i

        # theta_i_per_segment = np.split(theta_i, self.cr.num_segments)
        self.cr.set_config(theta_i)

    def _check_error_in_bounds(self, *args, **kwargs):
        """
        compute the error for the current solution, return True if the error
        is within the acceptable bounds set by the settings object

        the error is computed as the norm of the difference between the current
        pose and the target pose, and tolerances are set in the settings object
        for position and orientation separately
        """

        error = self.cr.pose_vector() - self.target_pose

        # of form (check result, (position error, orientation error))
        error_res = self.settings.check_error_bounds(error[:3], error[3:])
        return error_res

    @property
    def stopping_condition(self) -> tuple[bool, IkResult | None]:
        if self.solved:
            # possible that other parts of algorithm set solved to True
            return (True, IkResult.SUCCESS)
        (error_in_bounds, error) = self._check_error_in_bounds()
        if error_in_bounds:
            return (True, IkResult.SUCCESS)
        elif self.iter_count > self.settings.max_iter:
            return (True, IkResult.MAX_ITER)
        else:
            print(f"Iteration {self.iter_count} - Error: {error}")
            return (False, None)


if __name__ == "__main__":
    """
    if the module is run as main, a couple examples of the NR solver will be run
    """
    from math import pi

    # Test Case 1: Single segment CR
    print("**** Test Case 1: single-segment CR ****")
    seg1 = ConstantCurvatureSegment(1 / 0.14, -0.8 * pi, 0.05)
    robot = ConstantCurvatureCR([seg1])

    target_robot = ConstantCurvatureCR([ConstantCurvatureSegment(1 / 0.11, -pi, 0.05)])
    print(
        f"starting curvature is: {robot.state_vector()} yielding state\n {robot.pose_vector(robot.state_vector())}"
    )

    # draw_tdcr(robot.as_discrete_curve(pts_per_seg=10))
    settings = NewtonRhapsonIkSettings()

    target_pose = target_robot.pose_vector()

    solver = NewtonRhapsonIkSolver(robot, settings, robot.state_vector(), target_pose)
    solver.solve()

    print(f"Solution at: {solver.theta_i} after {solver.iter_count} iterations")
    print(f"yields position: {solver.cr.pose_vector(solver.cr.state_vector())}")
    print(f"\ntarget: {target_pose}")
    print(f"Error: {solver.cr.pose_vector() - target_pose}")

    draw_tdcr(solver.cr.as_discrete_curve(pts_per_seg=10))
    draw_tdcr(target_robot.as_discrete_curve(pts_per_seg=10))

    # Test Case 2: Multi-segment CR
    print("\n**** Test Case 2: multiple-segment CR ****")

    seg1 = ConstantCurvatureSegment(1 / 0.1, pi / 4, 0.05)
    seg2 = ConstantCurvatureSegment(1 / 0.05, 0, 0.03)
    robot = ConstantCurvatureCR([seg1, seg2])

    target_robot = ConstantCurvatureCR(
        [
            ConstantCurvatureSegment(1 / 0.11, pi / 3, 0.05),
            ConstantCurvatureSegment(1 / 0.1, pi / 10, 0.03),
        ]
    )

    print(
        f"starting curvature is: {robot.state_vector()} yielding state\n {robot.pose_vector(robot.state_vector())}"
    )

    target_pose = target_robot.pose_vector()

    solver = NewtonRhapsonIkSolver(robot, settings, robot.state_vector(), target_pose)
    solver.solve()

    print(f"Solution at: {solver.theta_i} after {solver.iter_count} iterations")
    print(f"yields position: {solver.cr.pose_vector(solver.cr.state_vector())}")
    print(f"\ntarget: {target_pose}")
    print(f"Error: {solver.cr.pose_vector() - target_pose}")

    draw_tdcr(solver.cr.as_discrete_curve(pts_per_seg=10))
    draw_tdcr(target_robot.as_discrete_curve(pts_per_seg=10))
