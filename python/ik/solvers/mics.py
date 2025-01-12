import numpy as np
from scipy.linalg import logm
from spatialmath import SE3

from common.robot import ConstantCurvatureCR
from common.utils import se3_to_uq, se3_to_pose

from ik.solvers.base_solver import CcIkSettings, CcIkSolver, IkResult
from ik.solvers.nr import NewtonRhapsonIkSettings, NewtonRhapsonIkSolver
from ik.target import IkTarget, SE3IkTarget

from copy import deepcopy


GAMMA_VAL = 0.5 + 1 / np.pi


def arc_params_from_quaternion(q, length):
    a, b, c = q
    kappa = 2 / length * np.arccos(a)
    phi = np.arctan2(-b, c)
    return kappa, phi


class MicsSolverSettings(CcIkSettings):
    num_t_steps = 100
    max_numerical_solver_iterations = 100
    zero_tolerance = 1e-6
    numerical_solver_settings = NewtonRhapsonIkSettings()


class MicsSolver(CcIkSolver):
    settings: MicsSolverSettings
    mics_starting_points: list

    class NoLocalMinimaFound(Exception):
        pass

    def __init__(
        self,
        cr: ConstantCurvatureCR,
        settings: MicsSolverSettings,
        ik_target: IkTarget,
        **kwargs,
    ):
        super().__init__(cr, settings, ik_target, **kwargs)

        if self.cr.num_segments != 3:
            raise ValueError("MICS solver only works for 3 segment chains")
        if any(seg.is_extensible for seg in self.cr.segments):
            raise ValueError("MICS solver does not support extensible segments")

        # problem definition - MICS paper uses r, q to denote position, orientation respectively
        se3: np.ndarray = SE3(ik_target.as_array()).A
        self.target_pose = se3
        self.r = se3[:3, 3]
        self.q = se3_to_uq(se3)

        # constant robot segment lengths
        self.l1 = self.cr.segments[0].length
        self.l2 = self.cr.segments[1].length
        self.l3 = self.cr.segments[2].length

        # the following parameters can be determined solely from the problem defn
        a, b, c, d = self.q
        self.A = np.array([[-a, -d, c], [d, -a, -b], [c, -b, a]])
        self.B = np.array([[d, a, b], [-a, d, c], [-b, -c, d]])

        n0 = self.B.T @ self.r
        n = n0 / np.linalg.norm(n0)
        l3 = self.cr.segments[2].length
        self.r0 = (
            GAMMA_VAL * d * l3 / np.linalg.norm(n0) * n
        )  # this is how it's presented in the code?

        self.norm_r01 = np.sqrt(1 - np.inner(self.r0, self.r0))

        # the following are all normalized
        self.n0 = n0 / np.linalg.norm(n0)
        self.n1 = np.linalg.cross(self.n0, np.array([0, 0, 1]))
        self.n2 = np.linalg.cross(self.n0, self.n1)

        self.u = np.array([n[1], n[0], 0])
        self.v = np.linalg.cross(n, self.u)
        self.P = np.column_stack([self.u, self.v, n])
        # self.P = np.column_stack([self.n1, self.n2, self.n0])
        # search settings
        self.t_step = 1 / settings.num_t_steps

        self.mics_starting_points = []

    def _get_r3_approx(self, t):
        """
        implements eqn (40) in the MICS paper, term by term
        """

        r3t = self.r0 + self.norm_r01 * self.P @ np.array(
            [np.sin(2 * np.pi * t), np.cos(2 * np.pi * t), 0]
        )

        return r3t

    def _get_r1_approx(self):
        """
        implements equation (37) in the MICS paper

        two equalities that must be satisfied. Using just one
        may not provide sufficient information if some components
        of n0 are zero components, so we use both
        """

        r1 = np.array([None, None, None])

        for i, n_i in enumerate(self.n0):
            if n_i < self.settings.zero_tolerance:
                continue
            r1[i] = GAMMA_VAL * self.l1 * self.r0[i]

        # some components of r1 can be None because n0 can have zero components
        # we can solve for these components using the other equation

        unsolved_indices = [i for i, val in enumerate(r1) if val is None]
        ne = self.B @ self.r3

        if len(unsolved_indices) == 1:
            # can determine the last element of r1 using the other equation
            i = unsolved_indices[0]
            r1[i] = sum([-r1[j] * ne[j] for j in range(3) if j != i])

        if len(unsolved_indices) == 2:
            # too many degrees of freedom, multiple solutions
            # set first unsolved to keep magnitude of 1, other to zero

            # TODO: this involves solving a quadratic equation that will yield two solutions.
            # shouldn't occur often in practice...
            raise NotImplementedError("Multiple solutions for r1")

        return r1

    def _get_r2_values(self):
        """
        implements equations (18), (24) from the MICS paper
        and provides the two candidate values for r2
        """

        # necessary that r1 is set by previous steps in the iteration
        r1 = self.r1

        r2_cand_1 = self.A @ self.r1

        flip_matrix = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        v = self.r - np.linalg.norm(r1) * r1

        r2_cand_2 = flip_matrix @ (2 * r1 @ r1.T - np.eye(3)) @ v / np.linalg.norm(v)

        return r2_cand_1, r2_cand_2

    def _get_error(self, r1, r2, r3):
        """
        determines the error of the solution as defined in (42)

        the candidate unit sphere points r1, r2, r3 are translated into
        twist coordinates which can then be used to determine the resultant
        end effector pose (41) that is used to determine the error.

        The vectors r_i each map to a unit quaternion denoting the rotational
        transformation of the segment i, which in turn provide the arc parameters
        kappa and phi required to determine the twist coordinates.
        """

        unit_sphere_vectors = [r1, r2, r3]

        # stored as q = a + bi + cj + dk quaternions, with d=0 omitted
        rotation_quaternions = [(ri[2], -ri[1], ri[0]) for ri in unit_sphere_vectors]

        # arc parameters for each segment
        kappa1, phi1 = arc_params_from_quaternion(rotation_quaternions[0], self.l1)
        kappa2, phi2 = arc_params_from_quaternion(rotation_quaternions[1], self.l2)
        kappa3, phi3 = arc_params_from_quaternion(rotation_quaternions[2], self.l3)

        # twist coordinates for each segment
        xi1 = self.l1 * np.array(
            [-kappa1 * np.sin(phi1), kappa1 * np.cos(phi1), 0, 0, 0, 1]
        )
        xi2 = self.l2 * np.array(
            [-kappa2 * np.sin(phi2), kappa2 * np.cos(phi2), 0, 0, 0, 1]
        )
        xi3 = self.l3 * np.array(
            [-kappa3 * np.sin(phi3), kappa3 * np.cos(phi3), 0, 0, 0, 1]
        )

        # convert to 4x4 SE3
        T1 = SE3.Exp(xi1)
        T2 = SE3.Exp(xi2)
        T3 = SE3.Exp(xi3)

        T = SE3(T1 * T2 * T3)

        # error as defined in (42)
        e = np.linalg.norm(se3_to_pose(logm(T.inv().A @ self.target_pose)))
        return e

    def _set_state_from_r(self, r1, r2, r3):
        """
        set self.cr state from the r1, r2, r3 values

        transforms the r1, r2, r3 unit sphere points into arc parameters,
        then sets the robot state accordingly.

        Note: this function should not be called to evaluate errors
        (_get_error has a more efficient implementation already).
        This should only be used to set the robot state after a solution
        has been found.
        """

        unit_sphere_vectors = [r1, r2, r3]

        # stored as q = a + bi + cj + dk quaternions, with d=0 omitted
        rotation_quaternions = [(ri[2], -ri[1], ri[0]) for ri in unit_sphere_vectors]

        # arc parameters for each segment
        kappa1, phi1 = arc_params_from_quaternion(rotation_quaternions[0], self.l1)
        kappa2, phi2 = arc_params_from_quaternion(rotation_quaternions[1], self.l2)
        kappa3, phi3 = arc_params_from_quaternion(rotation_quaternions[2], self.l3)

        print(kappa1, phi1, kappa2, phi2, kappa3, phi3)

        self.cr.set_config(np.array([[kappa1, phi1], [kappa2, phi2], [kappa3, phi3]]))

    def _numerical_correction(self):
        """
        assumes own robot state has already been set, and attempts to perform
        NR minimization from this point
        """
        robot_copy = deepcopy(self.cr)

        numerical_solver = NewtonRhapsonIkSolver(
            robot_copy,
            self.settings.numerical_solver_settings,
            robot_copy.state_vector(),
            SE3IkTarget(se3_to_pose(self.target_pose)),
        )

        numerical_solver.solve()

        _, (pos_error, ori_error) = numerical_solver._check_error_in_bounds()
        error = np.linalg.norm(np.vstack([pos_error, ori_error]))

        return (
            error < self.settings.zero_tolerance,
            numerical_solver.cr.state_vector(),
            error,
        )

    def solve(self, *args, **kwargs):
        """
        uses internally developed methods for determining r1, r2, r3 and the error
        (these functions in the source code are not publicly available) but uses the same
        logic as in the source MATLAB code for local error minimum detection
        """

        t = 0
        i = 0
        num_points = 0

        errors = np.array(
            [np.nan for _ in range(self.settings.max_numerical_solver_iterations)]
        )
        min_iter_nums = np.array(
            [np.nan for _ in range(self.settings.max_numerical_solver_iterations)]
        )

        local_min = []

        # find all errors
        while i < self.settings.max_numerical_solver_iterations:
            self.r3 = self._get_r3_approx(t)
            self.r1 = self._get_r1_approx()

            r2_cand_1, r2_cand_2 = self._get_r2_values()

            e1 = self._get_error(self.r1, r2_cand_1, self.r3)
            e2 = self._get_error(self.r1, r2_cand_2, self.r3)

            if e1 > e2:
                self.r2 = r2_cand_2
                err = e2
            else:
                self.r2 = r2_cand_1
                err = e1

            # TODO: fill out the error checking to mirror the MATLAB logic line for line
            if i == 0:
                pass  # noqa
            elif i == 1:
                # always add first point
                num_points += 1
                local_min.append((self.r1, self.r2, self.r3))
                min_iter_nums[num_points] = 1
            else:
                if err > errors[i - 1] and errors[i - 1] <= errors[i - 2]:
                    num_points += 1
                    local_min.append((self.r1, self.r2, self.r3))
                    min_iter_nums[num_points] = i - 1

            errors[i] = err
            t += self.t_step
            i += 1

        # check limits of search space
        if t == 1:  # first, last point will coincide
            if errors[1] > errors[0] and errors[0] <= errors[2]:
                # first/last point is a local min
                num_points += 1
                local_min.append((self.r1, self.r2, self.r3))
                min_iter_nums[num_points] = 1
            else:
                local_min = local_min[1:]
                min_iter_nums = min_iter_nums[1:]

        else:  # step size not divisor of 1, check both ends
            if errors[0] > errors[-1] and errors[-1] <= errors[-2]:
                num_points += 1
                local_min.append((self.r1, self.r2, self.r3))
                min_iter_nums[num_points] = i - 1
            if errors[1] > errors[0] and errors[0] <= errors[2]:
                num_points += 1
                local_min.append((self.r1, self.r2, self.r3))
                min_iter_nums[num_points] = 1

        # try numerical correction (NR) for all candidate local minima
        if len(local_min) == 0:
            raise self.NoLocalMinimaFound("No local minima found")

        # cache all local minima found - we will later start from these points for numerical convergence
        self.mics_starting_points = local_min

        post_correction_errors = []
        for i, (r1, r2, r3) in enumerate(local_min):
            try:
                # set robot state from the local minima
                self._set_state_from_r(r1, r2, r3)
                # use the set robot state to start the numerical correction
                converged, robot_state, error = self._numerical_correction()
                post_correction_errors.append(error)

                if converged:
                    self.cr.set_config(robot_state)
                    return IkResult.SUCCESS
            except Exception as e:
                print(f"Unable to perform numerical convergence for local min {i}: {e}")

        if len(post_correction_errors) == 0:
            return IkResult.DIVERGED

        ind = np.argmin(post_correction_errors)
        r1, r2, r3 = local_min[ind]

        return IkResult.DIVERGED
