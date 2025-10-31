import time
import numpy as np
from scipy.linalg import logm
from spatialmath import SE3

from common.robot import ConstantCurvatureCR
from common.utils import (
    se3_to_uq,
    se3_to_pose,
    up_star,
    up_plus,
    up_oplus,
    invert_transformation,
)

from ik.solvers.base_solver import CcIkSettings, CcIkSolver, IkResult
from ik.solvers.nr import NewtonRaphsonIkSettings, NewtonRaphsonIkSolver
from ik.target import IkTarget, IkTargetType

from copy import deepcopy


GAMMA_VAL = 0.5 + 1 / np.pi


def arc_params_from_quaternion(q, length):
    a, b, c = q
    if abs(a) > 1 and abs(a) < 1 + 1e-5:
        a = np.clip(a, -1, 1)
    elif abs(a) > 1 + 1e-4:
        raise ValueError(f"a is out of bounds (got value of {a})")
    kappa = 2 / length * np.arccos(a)
    phi = np.arctan2(-b, c)
    return kappa, phi


def _rho(a, length) -> int:
    # linear distance between two ends of a circular arc
    if a == 1:
        return length
    else:
        return length * np.sqrt(1 - a**2) / np.acos(a)


class MicsSolverSettings(CcIkSettings):
    t_search_resolutions = [0.03, 0.01, 0.005]
    zero_tolerance = 1e-4
    numerical_solver_settings = NewtonRaphsonIkSettings(max_iter=30)
    num_r1_corrections = 2
    num_r3_corrrections = 1


class MicsSolver(CcIkSolver):
    target_type = IkTargetType.SE3
    settings_class = MicsSolverSettings
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
        self.r0 = GAMMA_VAL * d * l3 / np.linalg.norm(n0) * n

        self.norm_r01 = np.sqrt(1 - np.inner(self.r0, self.r0))

        # the following are all normalized
        self.n0 = n0 / np.linalg.norm(n0)
        self.n1 = np.linalg.cross(self.n0, np.array([0, 0, 1]))
        self.n2 = np.linalg.cross(self.n0, self.n1)

        self.u = np.array([n[1], n[0], 0])
        self.v = np.linalg.cross(n, self.u)
        self.P = np.column_stack([self.u, self.v, n])

        self.mics_starting_points = []
        self.mics_starting_point_configurations = []
        self.converged_starting_point = None
        self.iter_count = 0

    def _get_r3_approx(self, t):
        """
        implements eqn (40) in the MICS paper, term by term
        """

        r3t = self.r0 + self.norm_r01 * self.P @ np.array(
            [np.sin(2 * np.pi * t), np.cos(2 * np.pi * t), 0]
        )

        return r3t

    def _spp(self, n1, d, n2, rn):
        """
        mirrors the spp function in function `spp` in solve_r1.m in
        the MICS source code
        """
        n11, n12, n13 = n1
        n21, n22, n23 = n2

        det0 = n11 * n22 - n12 * n21
        det1 = n12 * n23 - n13 * n22
        det2 = n11 * n23 - n13 * n21

        a = det0**2 + det1**2 + det2**2

        if a < self.settings.zero_tolerance:
            out = np.array(
                [-rn[0] * rn[2], -rn[1] * rn[2], rn[0] ** 2 + rn[1] ** 2]
            ) / np.sqrt(rn[0] ** 2 + rn[1] ** 2)
        else:
            b = 2 * d * (n22 * det1 + n21 * det2)
            c = d**2 * (n22**2 + n21**2) - det0**2
            delta = b**2 - 4 * a * c

            if delta < 0:
                r3 = -b / (2 * a)
                r1 = (det1 * r3 + n22 * d) / det0
                r2 = -(det2 * r3 + n21 * d) / det0
                out = np.array([r1, r2, r3])
                out = out / np.linalg.norm(out)

            else:
                r3 = (-b + np.sqrt(delta)) / (2 * a)
                r1 = (det1 * r3 + n22 * d) / det0
                r2 = -(det2 * r3 + n21 * d) / det0
                out = np.array([r1, r2, r3])

        return out / np.linalg.norm(out)

    def _get_r1_approx(self):
        """
        implements equation (37) in the MICS paper.
        Both equalities are satisfied by the following.
        """

        d = self.q[3]

        r0 = self.l1 * d / np.inner(self.n0, self.n0) * self.n0
        ne = self.B @ self.r3

        r1 = self._spp(self.n0, GAMMA_VAL * self.l1 * d, ne, self.r3)

        if d != 0:  # perform correction steps
            for _ in range(self.settings.num_r1_corrections):
                M1 = np.vstack(
                    [
                        self.n0
                        + np.array(
                            [
                                0,
                                0,
                                self.l1
                                * d
                                * (1 / np.acos(r1[2]) - 1 / np.sqrt(1 - r1[2] ** 2)),
                            ]
                        ),
                        ne,
                    ]
                )

                x1 = np.hstack([np.reshape(r0, (3, 1)), np.reshape(ne, (3, 1))])
                M = M1 @ x1

                correction = x1 @ np.linalg.solve(
                    M,
                    np.array(
                        [np.dot(self.n0, r1) - _rho(r1[2], self.l1) * d, np.dot(ne, r1)]
                    ),
                )

                tmp = r1 - correction
                r1 = tmp / np.linalg.norm(tmp)

        return r1

    def _rho(self, a, length):
        if a == 1:
            return length
        else:
            return length * np.sqrt(1 - a**2) / np.arccos(a)

    def _get_r2_values(self):
        """
        implements equations (18), (24) from the MICS paper
        and provides the two candidate values for r2
        """

        a, b, c, _ = self.q
        B = self.B
        m = np.array([c, -b, a])
        qe = np.concat([np.array([m @ self.r3]), B @ self.r3])
        re = np.concat([np.array([0]), self.r])
        re -= (
            _rho(self.r3[2], self.l3)
            * up_plus(qe)
            @ up_oplus(up_star(qe))
            @ np.concat([np.array([0]), self.r3])
        )
        re = re[1:]

        # applying rotational constraint
        ae, be, ce, de = qe
        Ae = np.array([[-ae, -de, ce], [de, -ae, -be], [ce, -be, ae]])
        r2r = Ae @ self.r1
        rv = re - _rho(self.r1[2], self.l1) * self.r1
        w2 = (
            -(
                2 * np.dot(np.reshape(self.r1, (3, 1)), np.reshape(self.r1, (1, 3)))
                - np.eye(3)
            )
            @ rv
            / np.linalg.norm(rv)
        )
        r2t = np.array([w2[0], w2[1], -w2[2]])

        return r2r, r2t

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
        t_inverse = invert_transformation(T.A)
        e = np.linalg.norm(se3_to_pose(logm(t_inverse) @ self.target_pose))
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

        config = np.array([[kappa1, phi1], [kappa2, phi2], [kappa3, phi3]])

        self.cr.set_config(config)
        self.mics_starting_point_configurations.append(config)

    def _numerical_correction(self):
        """
        assumes own robot state has already been set, and attempts to perform
        NR minimization from this point
        """
        robot_copy = deepcopy(self.cr)

        numerical_solver = NewtonRaphsonIkSolver(
            robot_copy,
            self.settings.numerical_solver_settings,
            self.ik_target,
        )

        numerical_result = numerical_solver.solve()

        pos_error, ori_error = numerical_solver.get_errors()
        error = np.linalg.norm(np.array([pos_error, ori_error]))

        return (
            numerical_result.is_success,
            numerical_solver.cr.state_vector(),
            error,
        )

    def solve(self, *args, **kwargs):
        for t_resolution in self.settings.t_search_resolutions:
            try:
                result = self._solve_with_resolution(t_resolution)
                if result == IkResult.SUCCESS:
                    return result
            except self.NoLocalMinimaFound:
                continue
        return IkResult.DIVERGED

    def _solve_with_resolution(self, t_resolution: float):
        """
        uses internally developed methods for determining r1, r2, r3 and the error
        (these functions in the source code are not publicly available) but uses the same
        logic as in the source MATLAB code for local error minimum detection
        """

        start = time.time()

        t = 0
        i = 0
        num_points = 0
        num_t_steps = int(1 / t_resolution) + 1

        errors = np.array([np.nan for _ in range(num_t_steps)])

        local_min = []
        local_min_indices = []

        # find all errors
        while t <= 1:
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

            if i == 0:
                pass  # noqa
            elif i == 1:
                # always add first point
                num_points += 1
                local_min.append((self.r1, self.r2, self.r3))
                local_min_indices.append(i)
            else:
                # local minimum case:
                if err > errors[i - 1] and errors[i - 1] <= errors[i - 2]:
                    num_points += 1
                    local_min.append((self.r1, self.r2, self.r3))
                    local_min_indices.append(i)

            errors[i] = err
            t += t_resolution
            i += 1

        # check limits of search space
        if t == 1:  # first, last point will coincide
            if not (errors[1] > errors[0] and errors[0] <= errors[-1]):
                local_min = local_min[1:]
                local_min_indices = local_min_indices[1:]

        else:  # step size not divisor of 1, check both ends
            if errors[0] > errors[-1] and errors[-1] <= errors[-2]:
                num_points += 1
                local_min.append((self.r1, self.r2, self.r3))
                local_min_indices.append(i)
            if not (errors[1] > errors[0] and errors[0] <= errors[-1]):
                local_min = local_min[1:]
                local_min_indices = local_min_indices[1:]

        # try numerical correction (NR) for all candidate local minima
        if len(local_min) == 0:
            raise self.NoLocalMinimaFound("No local minima found")

        # sort local minima by how close the t value was to 0.5
        # in our code, we achieve the same by sorting by the index of the local minima
        mid_index = num_t_steps / 2
        distances = [abs(index - mid_index) for index in local_min_indices]
        sorting_indices = np.argsort(distances)
        local_min = [local_min[i] for i in sorting_indices]

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
                    self.converged_starting_point = i
                    self.cr.set_config(robot_state)
                    self.exec_time = time.time() - start
                    # NOTE iter_count is interpreted as the number of times the MICS
                    # solver attempted numerical convergence. The number of iterations
                    # performed in the previous iteration is set as a hyperparameter
                    self.iter_count = i
                    return IkResult.SUCCESS
            except Exception as e:
                self.exec_time = time.time() - start
                print(f"Unable to perform numerical convergence for local min {i}: {e}")
                raise e

        self.exec_time = time.time() - start
        if len(post_correction_errors) == 0:
            return IkResult.DIVERGED

        ind = np.argmin(post_correction_errors)
        r1, r2, r3 = local_min[ind]

        return IkResult.DIVERGED
