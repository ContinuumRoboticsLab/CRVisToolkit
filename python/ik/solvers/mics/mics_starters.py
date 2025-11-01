import numpy as np
from scipy.linalg import norm

from ik.solvers.mics import mics_utils as utils


def spp(n1, d, n2, rn):
    """
    Solves for r that satisfies:
        n1 · r - d = 0
        n2 · r = 0
        |r| = 1

    Geometrically, r is the intersection of a sphere and two planes.

    Parameters:
        n1: normal vector 1 (3D)
        d: scalar offset
        n2: normal vector 2 (3D)
        rn: reference vector (3D)

    Returns:
        soln: solution vector (3D, unit norm)
    """
    n1 = np.asarray(n1).flatten()
    n2 = np.asarray(n2).flatten()
    rn = np.asarray(rn).flatten()

    n11, n12, n13 = n1[0], n1[1], n1[2]
    n21, n22, n23 = n2[0], n2[1], n2[2]

    det0 = n11 * n22 - n12 * n21
    det1 = n12 * n23 - n13 * n22
    det2 = n11 * n23 - n13 * n21

    a = det1**2 + det2**2 + det0**2

    if a < np.finfo(float).eps:
        # Degenerate case
        soln = np.array(
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
            soln = np.array([r1, r2, r3]) / norm([r1, r2, r3])
        else:
            r3 = (-b + np.sqrt(delta)) / (2 * a)
            r1 = (det1 * r3 + n22 * d) / det0
            r2 = -(det2 * r3 + n21 * d) / det0
            soln = np.array([r1, r2, r3])

    return soln


def solve_r1(L1, q, r, r3, noc):
    """
    Computes the model parameter of the 1st section.

    Parameters:
        L1: length of section 1
        q: desired quaternion (4D)
        r: desired translation (3D)
        r3: model parameter of section 3 (3D)
        noc: number of corrections

    Returns:
        r1: model parameter of section 1 (3D, unit norm)
    """
    q = np.asarray(q).flatten()
    r = np.asarray(r).flatten()
    r3 = np.asarray(r3).flatten()

    a, b, c, d = q[0], q[1], q[2], q[3]

    B = np.array([[d, a, b], [-a, d, c], [-b, -c, d]])

    n0 = B.T @ r
    r0 = (L1 * d) / norm(n0) ** 2 * n0
    ne = B @ r3

    r1 = spp(n0, (1 / 2 + 1 / np.pi) * L1 * d, ne, r3)

    if d != 0:
        for cor_idx in range(noc):
            # One-step correction
            n0_extended = n0 + np.array(
                [0, 0, L1 * d * (1 / np.arccos(r1[2]) - 1 / np.sqrt(1 - r1[2] ** 2))]
            )
            A_mat = np.column_stack([r0, ne])
            b_vec = np.array([n0_extended @ r1 - utils.rho(r1[2], L1) * d, ne @ r1])

            # Solve the linear system
            correction = A_mat @ np.linalg.solve(
                np.vstack([n0_extended.reshape(1, -1), ne.reshape(1, -1)]) @ A_mat,
                b_vec,
            )

            tmp = r1 - correction
            r1 = tmp / norm(tmp)

    return r1


def solve_r2(L1, L3, q, r, r3, r1):
    """
    Computes the model parameter of the 2nd section using rotational
    and translational constraints.

    Parameters:
        L1, L3: section lengths
        q: desired quaternion (4D)
        r: desired translation (3D)
        r3, r1: model parameters of sections 3 and 1 (3D each)

    Returns:
        r2r: solution using rotational constraint (3D)
        r2t: solution using translational constraint (3D)
    """
    q = np.asarray(q).flatten()
    r = np.asarray(r).flatten()
    r3 = np.asarray(r3).flatten()
    r1 = np.asarray(r1).flatten()

    a, b, c, d = q[0], q[1], q[2], q[3]

    B = np.array([[d, a, b], [-a, d, c], [-b, -c, d]])

    m = np.array([c, -b, a])
    qe = np.concatenate([[m @ r3], B @ r3])

    # Compute re
    q_ext = np.concatenate([[0], r])
    qe_transform = (
        utils.up_plus(qe)
        @ utils.up_oplus(utils.up_star(qe))
        @ np.concatenate([[0], r3])
    )
    re = q_ext - utils.rho(r3[2], L3) * qe_transform
    re = re[1:4]

    # Use rotational constraint
    ae, be, ce, de = qe[0], qe[1], qe[2], qe[3]
    Ae = np.array([[-ae, -de, ce], [de, -ae, -be], [ce, -be, ae]])
    r2r = Ae @ r1

    # Use translational constraint
    rv = re - utils.rho(r1[2], L1) * r1
    negative_w2 = (2 * np.outer(r1, r1) - np.eye(3)) @ rv / norm(rv)
    r2t = np.array([-negative_w2[0], -negative_w2[1], negative_w2[2]])

    return r2r, r2t


def get_err(r1, r2, r3, L1, L2, L3, q, r):
    """
    Computes the error between desired and current end pose.

    Parameters:
        r1, r2, r3: model parameters (3D each)
        L1, L2, L3: section lengths
        q: desired quaternion (4D)
        r: desired translation (3D)

    Returns:
        err: scalar error (norm of the pose difference)
    """
    r1 = np.asarray(r1).flatten()
    r2 = np.asarray(r2).flatten()
    r3 = np.asarray(r3).flatten()
    q = np.asarray(q).flatten()
    r = np.asarray(r).flatten()

    # Desired transformation
    Td = np.eye(4)
    Td[0:3, 0:3] = utils.q2rot(q)
    Td[0:3, 3] = r

    # Section transformations
    T1 = np.eye(4)
    T1[0:3, 0:3] = utils.q2rot(np.array([r1[2], -r1[1], r1[0], 0]))
    T1[0:3, 3] = utils.rho(r1[2], L1) * r1

    T2 = np.eye(4)
    T2[0:3, 0:3] = utils.q2rot(np.array([r2[2], -r2[1], r2[0], 0]))
    T2[0:3, 3] = utils.rho(r2[2], L2) * r2

    T3 = np.eye(4)
    T3[0:3, 0:3] = utils.q2rot(np.array([r3[2], -r3[1], r3[0], 0]))
    T3[0:3, 3] = utils.rho(r3[2], L3) * r3

    # Total transformation
    Tt = T1 @ T2 @ T3

    # Compute error
    V = utils.veelog(np.linalg.inv(Tt) @ Td)
    err = norm(V)

    return err


# ============================================================================
# MAIN SOLVER FUNCTION
# ============================================================================


def find_mics_starters(L1, L2, L3, q, r, par, noc):
    """
    Multi-solution solver core function for inverse kinematics of
    3-section constant-curvature robots.

    Parameters:
        L1, L2, L3: section lengths
        q: desired end rotation as quaternion (4D array)
        r: desired end translation (3D array)
        par: partition length (scalar, e.g., 0.01 or 0.03)
        noc: number of corrections [noc_r3, noc_r1] (2-element array)

    Returns:
        solns: 9 x N array of candidate solutions, each column is [r1; r2; r3]
    """
    q = np.asarray(q).flatten()
    r = np.asarray(r).flatten()
    noc = np.asarray(noc).flatten().astype(int)

    # Step 1: Initialize parameters
    a, b, c, d = q[0], q[1], q[2], q[3]

    B = np.array([[d, a, b], [-a, d, c], [-b, -c, d]])

    n0 = B.T @ r
    n = n0 / norm(n0)
    r0 = (1 / 2 + 1 / np.pi) * (L3 * d) / norm(n0) * n

    norm_r0 = norm(r0)
    norm_r01 = np.sqrt(1 - norm_r0**2)

    # Create orthonormal basis
    u = np.array([n[1], -n[0], 0]) / norm([n[1], -n[0], 0])
    v = np.cross(n, u)
    P = np.column_stack([u, v, n])

    # Step 2: Traverse parameter space
    zeta = np.arange(0, 1 + par, par)
    npar = len(zeta)

    err = np.full(npar, np.nan)
    err_r = np.full(npar, np.nan)
    err_t = np.full(npar, np.nan)
    solns = np.full((9, npar), np.nan)
    is_indices = []
    nts = 0

    for i in range(npar):
        t = zeta[i]

        # Step 2a: Compute r3
        circle_point = np.array([np.sin(2 * np.pi * t), np.cos(2 * np.pi * t), 0])
        r3 = r0 + norm_r01 * (P @ circle_point)

        # Step 2b: Apply corrections to r3
        if d != 0:
            for cor_idx in range(noc[0]):
                # One-step correction
                denominator = n0 @ r0 + np.array(
                    [
                        0,
                        0,
                        L3 * d * (1 / np.arccos(r3[2]) - 1 / np.sqrt(1 - r3[2] ** 2)),
                    ]
                )
                numerator = n0 @ r3 - utils.rho(r3[2], L3) * d
                tmp = r3 - (numerator / denominator) * r0
                r3 = tmp / norm(tmp)

        # Step 2c: Solve for r1
        r1 = solve_r1(L1, q, r, r3, noc[1])

        # Step 2d: Solve for r2 (two candidates)
        r2r, r2t = solve_r2(L1, L3, q, r, r3, r1)

        # Step 2e: Evaluate errors
        e_r = get_err(r1, r2r, r3, L1, L2, L3, q, r)
        e_t = get_err(r1, r2t, r3, L1, L2, L3, q, r)

        # Step 2f: Select r2 with minimum error
        if e_r < e_t:
            r2 = r2r
            e = e_r
        else:
            r2 = r2t
            e = e_t

        # Step 2g: Detect local minima
        soln = None  # shouldn't ever raise issue in iteration, but stop LSP complaints
        if i == 0:
            # No operation
            pass
        elif i == 1:
            nts += 1
            solns[:, nts - 1] = soln
            is_indices.append(1)
        else:  # i > 1
            if e > err[i - 1] and err[i - 1] <= err[i - 2]:
                # Local minimum at i-1
                nts += 1
                solns[:, nts - 1] = soln
                is_indices.append(i - 1)

        # Update for next iteration
        soln = np.concatenate([r1, r2, r3])
        err[i] = e
        err_r[i] = e_r
        err_t[i] = e_t

    # Step 3: Handle boundary conditions
    if zeta[-1] != 1:
        # Check if last point is a local minimum
        if err[0] > err[-1] and err[-1] <= err[-2]:
            nts += 1
            solns[:, nts - 1] = soln
            is_indices.append(len(zeta) - 1)

        # Check if first point is a local minimum
        if err[1] > err[0] and err[0] <= err[-1]:
            solns = solns[:, :nts]
            # is_indices already correct
        else:
            # Remove first solution
            if nts > 0:
                solns = solns[:, 1:nts]
                is_indices = is_indices[1:]
                nts -= 1
    else:  # Last point coincides with first
        # Check if first (also last) point is a local minimum
        if err[1] > err[0] and err[0] <= err[-2]:
            solns = solns[:, :nts]
            # is_indices already correct
        else:
            # Remove first solution
            if nts > 0:
                solns = solns[:, 1:nts]
                is_indices = is_indices[1:]
                nts -= 1

    # Step 4: Post-process solutions
    if nts > 0:
        solns = solns[:, :nts]
        ts = zeta[is_indices]

        # Sort by distance from center (t=0.5)
        sort_idx = np.argsort(np.abs(ts - 0.5))
        solns = solns[:, sort_idx]
    else:
        solns = np.empty((9, 0))

    return solns
