"""
Newton-Raphson method for correcting initial values in robot kinematics.

This module implements the Newton-Raphson iterative correction method for
3-link constant-curvature robots using the product of exponentials formula.
"""

import numpy as np
from scipy.linalg import logm
from scipy.spatial.transform import Rotation
from ik.solvers.mics.mics_utils import q2rot, get_end, xi2arc, arc2xi, up_vee
from ik.solvers.mics.jacobian3cc import jacobian3cc


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
        try:
            V = up_vee(logm(np.linalg.inv(Tt) @ Td))
        except Exception as _:
            return xi, np.inf, k

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
