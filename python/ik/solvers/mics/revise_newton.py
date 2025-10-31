"""
Newton-Raphson method for correcting initial values in robot kinematics.

This module implements the Newton-Raphson iterative correction method for
3-link constant-curvature robots using the product of exponentials formula.
"""

import numpy as np
from scipy.linalg import logm
from scipy.spatial.transform import Rotation
from ik.solvers.mics.jacobian3cc import up_hat, up_vee, jacobian3cc


def q2rot(q):
    """
    Converts a quaternion to a rotation matrix.

    Parameters
    ----------
    q : array_like
        Quaternion [a, b, c, d] where a is the scalar part

    Returns
    -------
    R : ndarray
        3x3 rotation matrix
    """
    q = np.asarray(q).flatten()
    a, b, c, d = q[0], q[1], q[2], q[3]

    R = np.array(
        [
            [1 - 2 * c**2 - 2 * d**2, 2 * b * c - 2 * a * d, 2 * a * c + 2 * b * d],
            [2 * b * c + 2 * a * d, 1 - 2 * b**2 - 2 * d**2, 2 * c * d - 2 * a * b],
            [2 * b * d - 2 * a * c, 2 * a * b + 2 * c * d, 1 - 2 * b**2 - 2 * c**2],
        ]
    )

    return R


def exphat(V):
    """
    Composition of the hat map and the matrix exponential (Rodrigues' formula).

    Computes the matrix exponential of the hat map of V, mapping a vector in
    R^6 to a transformation matrix in SE(3).

    Parameters
    ----------
    V : array_like
        6-element vector [omega; v] where omega is angular and v is linear

    Returns
    -------
    M : ndarray
        4x4 transformation matrix in SE(3)
    """
    V = np.asarray(V).flatten()
    theta = np.linalg.norm(V[:3])

    if theta < 2e-8:
        # Small angle approximation
        M = np.eye(4)
        M[:3, 3] = V[3:6]
    else:
        omega = V[:3] / theta
        v = V[3:6] / theta
        omega_hat = up_hat(omega)
        omega_hat_sq = omega_hat @ omega_hat

        # Rotation part using Rodrigues' formula
        R = np.eye(3) + np.sin(theta) * omega_hat + (1 - np.cos(theta)) * omega_hat_sq

        # Translation part
        p = (
            np.eye(3) * theta
            + (1 - np.cos(theta)) * omega_hat
            + (theta - np.sin(theta)) * omega_hat_sq
        ) @ v

        M = np.vstack([np.hstack([R, p.reshape(-1, 1)]), np.array([[0, 0, 0, 1]])])

    return M


def get_end(L1, L2, L3, xi):
    """
    Computes the end pose of a 3-section constant-curvature robot.

    Parameters
    ----------
    L1 : float
        Length of link 1
    L2 : float
        Length of link 2
    L3 : float
        Length of link 3
    xi : array_like
        6-element array of joint parameters [xi1, xi2, xi3, xi4, xi5, xi6]

    Returns
    -------
    T : ndarray
        4x4 transformation matrix representing the end effector pose
    """
    xi = np.asarray(xi).flatten()

    T1 = exphat(np.array([xi[0], xi[1], 0, 0, 0, L1]))
    T2 = exphat(np.array([xi[2], xi[3], 0, 0, 0, L2]))
    T3 = exphat(np.array([xi[4], xi[5], 0, 0, 0, L3]))

    T = T1 @ T2 @ T3

    return T


def xi2arc(L1, L2, L3, xi):
    """
    Converts the overall exponential coordinate to the arc parameters.

    Parameters
    ----------
    L1, L2, L3 : float
        Section lengths
    xi : array_like
        6-element exponential coordinate array

    Returns
    -------
    arc : ndarray
        6-element array [k1, p1, k2, p2, k3, p3] where k is curvature
        and p is bending angle for each section
    """
    xi = np.asarray(xi).flatten()

    k1 = np.mod(np.sqrt(xi[0] ** 2 + xi[1] ** 2), 2 * np.pi) / L1
    p1 = np.arctan2(-xi[0], xi[1])

    k2 = np.mod(np.sqrt(xi[2] ** 2 + xi[3] ** 2), 2 * np.pi) / L2
    p2 = np.arctan2(-xi[2], xi[3])

    k3 = np.mod(np.sqrt(xi[4] ** 2 + xi[5] ** 2), 2 * np.pi) / L3
    p3 = np.arctan2(-xi[4], xi[5])

    arc = np.array([k1, p1, k2, p2, k3, p3])

    return arc


def arc2xi(L1, L2, L3, arc):
    """
    Converts the arc parameters to the exponential coordinate.

    Parameters
    ----------
    L1, L2, L3 : float
        Section lengths
    arc : array_like
        6-element array [k1, p1, k2, p2, k3, p3] containing curvatures
        and bending angles of each section

    Returns
    -------
    xi : ndarray
        6-element exponential coordinate array
    """
    arc = np.asarray(arc).flatten()

    xi = np.array(
        [
            -L1 * arc[0] * np.sin(arc[1]),
            L1 * arc[0] * np.cos(arc[1]),
            -L2 * arc[2] * np.sin(arc[3]),
            L2 * arc[2] * np.cos(arc[3]),
            -L3 * arc[4] * np.sin(arc[5]),
            L3 * arc[4] * np.cos(arc[5]),
        ]
    )

    return xi


def revise_newton(L1, L2, L3, q, r, xi, mstep, tol, plot=False):
    """
    Correct the initial value with the Newton-Raphson method.

    Iteratively corrects the robot configuration using Newton-Raphson method
    to achieve the desired end effector pose.

    Parameters
    ----------
    L1, L2, L3 : float
        Section lengths of the 3-link robot
    q : array_like
        Desired end rotation as quaternion [a, b, c, d]
    r : array_like
        Desired end translation [x, y, z]
    xi : array_like
        Initial value of exponential coordinates (6 elements)
    mstep : int
        Maximum allowed steps of iterations
    tol : float
        Error tolerance for convergence
    plot : bool, optional
        If True, plot the error convergence (default: False)
        Note: plotting functionality not yet implemented

    Returns
    -------
    xi_star : ndarray
        Final corrected exponential coordinates
    err : float
        Final error norm
    k : int
        Number of iterations performed

    Example
    -------
    >>> L1, L2, L3 = 1.0, 1.0, 1.0
    >>> alpha = 15*np.pi/16
    >>> omega = np.array([0.48, np.sqrt(3)/10, -0.86])
    >>> q = np.array([np.cos(alpha/2), *(np.sin(alpha/2)*omega)])
    >>> r = np.array([-0.4, 1.1, 0.8])
    >>> xi_0 = np.random.rand(6) * 2 * np.pi
    >>> xi, err, noi = revise_newton(L1, L2, L3, q, r, xi_0, 200, 1e-2)
    """
    xi = np.asarray(xi).flatten()
    q = np.asarray(q).flatten()
    r = np.asarray(r).flatten()

    # Desired end effector configuration
    Td = np.vstack([np.hstack([q2rot(q), r.reshape(-1, 1)]), np.array([[0, 0, 0, 1]])])

    # Storage for error metrics
    omg_e = np.full(mstep + 1, np.nan)
    v_e = np.full(mstep + 1, np.nan)
    e = np.full(mstep + 1, np.nan)

    k = 0
    while k < mstep:
        # Check error condition
        Tt = get_end(L1, L2, L3, xi)
        V = up_vee(logm(np.linalg.inv(Tt) @ Td))

        omg_e[k] = np.linalg.norm(V[:3])
        v_e[k] = np.linalg.norm(V[3:6])
        e[k] = np.linalg.norm(V)

        if e[k] < tol:
            break
        else:
            # Update step using Newton-Raphson
            J = jacobian3cc(L1, L2, L3, xi)
            # Solve (J^T * J) * delta_xi = J^T * V
            xi = xi + np.linalg.solve(J.T @ J, J.T @ V)

            # Convert to arc and back to xi (for normalization/wrapping)
            xi = arc2xi(L1, L2, L3, xi2arc(L1, L2, L3, xi))
            k = k + 1

    # Compute final error if max steps reached
    if k == mstep:
        Tt = get_end(L1, L2, L3, xi)
        V = up_vee(logm(np.linalg.inv(Tt) @ Td))
        omg_e[k] = np.linalg.norm(V[:3])
        v_e[k] = np.linalg.norm(V[3:6])
        e[k] = np.linalg.norm(V)

    xi_star = xi
    err = e[k]

    if plot:
        # Plotting functionality could be added here
        # For now, just print a warning
        print("Warning: Plotting functionality not yet implemented")
        print(f"Iterations: {k}, Final error: {err:.6e}")
        print(f"Angular error: {omg_e[k]:.6e}, Linear error: {v_e[k]:.6e}")

    return xi_star, err, k


if __name__ == "__main__":
    # Example usage
    L1, L2, L3 = 1.0, 1.0, 1.0

    # Define desired pose
    alpha = 15 * np.pi / 16
    omega = np.array([0.48, np.sqrt(3) / 10, -0.86])
    omega = omega / np.linalg.norm(omega)  # Normalize
    q = np.array([np.cos(alpha / 2), *(np.sin(alpha / 2) * omega)])
    r = np.array([-0.4, 1.1, 0.8])
    print(Rotation.from_quat((q[1], q[2], q[3], q[0])).as_matrix())

    # Random initial guess
    xi_0 = np.random.rand(6) * 2 * np.pi

    print("Initial xi:", xi_0)
    print("\nRunning Newton-Raphson correction...")

    xi_star, err, k = revise_newton(L1, L2, L3, q, r, xi_0, 200, 1e-2, plot=True)

    print(f"\nFinal xi: {xi_star}")
    print(f"Final error: {err:.6e}")
    print(f"Iterations: {k}")

    # Verify the result
    T_final = get_end(L1, L2, L3, xi_star)
    print("\nFinal end effector position:")
    print(T_final[:3, 3])
    print("\nDesired end effector position:")
    print(r)
