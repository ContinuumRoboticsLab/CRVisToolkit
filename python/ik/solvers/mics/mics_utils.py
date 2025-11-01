import numpy as np
from scipy.linalg import norm

from common.robot import ConstantCurvatureCR


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


def xi2arc_robot(xi: np.ndarray, cr: ConstantCurvatureCR):
    arc = []

    offset = 0
    for seg in cr.segments:
        kappa = np.sqrt(xi[offset] ** 2 + xi[offset + 1] ** 2)
        kappa = (kappa % np.pi) / seg.length
        phi = np.arctan2(-xi[offset], xi[offset + 1])
        arc.extend([kappa, phi])
        offset += 2

    return np.array(arc)


def up_hat(v):
    """
    Computes the Lie algebra of a vector (hat map).

    Converts a vector to its corresponding skew-symmetric matrix representation.

    Parameters
    ----------
    v : array_like
        Vector in R^3 or R^6

    Returns
    -------
    M : ndarray
        Element of so(3) (3x3) if v is in R^3, or se(3) (4x4) if v is in R^6

    Raises
    ------
    ValueError
        If input vector is not of length 3 or 6
    """
    v = np.asarray(v).flatten()

    if len(v) == 3:
        M = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    elif len(v) == 6:
        M = np.array(
            [
                [0, -v[2], v[1], v[3]],
                [v[2], 0, -v[0], v[4]],
                [-v[1], v[0], 0, v[5]],
                [0, 0, 0, 0],
            ]
        )
    else:
        raise ValueError("Input must be in R^3 or R^6")

    return M


def up_vee(M):
    """
    Computes the inverse of the hat map (vee map).

    Converts a skew-symmetric matrix back to its vector representation.

    Parameters
    ----------
    M : array_like
        Element of so(3) (3x3) or se(3) (4x4)

    Returns
    -------
    v : ndarray
        Vector in R^3 or R^6

    Raises
    ------
    ValueError
        If input matrix is not 3x3 or 4x4
    """
    M = np.asarray(M)

    if M.shape[0] == 3:
        v = np.array([-M[1, 2], M[0, 2], -M[0, 1]])
    elif M.shape[0] == 4:
        v = np.array([-M[1, 2], M[0, 2], -M[0, 1], M[0, 3], M[1, 3], M[2, 3]])
    else:
        raise ValueError("Input must be in so(3) or se(3)")

    return v


def rho(a, L):
    """
    Computes the linear distance between two ends of a circular arc.

    Parameters:
        a: scalar value
        L: length of the arc

    Returns:
        d: linear distance
    """
    if a == 1:
        d = L
    else:
        d = L * np.sqrt(1 - a**2) / np.arccos(a)
    return d


def up_plus(q):
    """
    Computes the matrix for left multiplication of quaternions.

    Parameters:
        q: quaternion as numpy array [delta, epsilon1, epsilon2, epsilon3]

    Returns:
        q_up_plus: 4x4 left multiplication matrix
    """
    q = np.asarray(q).flatten()
    delta = q[0]
    epsilon = q[1:4]

    # q^+ = [delta,        -epsilon^T
    #        epsilon,   delta*I + epsilon^×]
    q_up_plus = np.zeros((4, 4))
    q_up_plus[0, 0] = delta
    q_up_plus[0, 1:4] = -epsilon
    q_up_plus[1:4, 0] = epsilon
    q_up_plus[1:4, 1:4] = delta * np.eye(3) + up_hat(epsilon)

    return q_up_plus


def up_oplus(q):
    """
    Computes the matrix for right multiplication of quaternions.

    Parameters:
        q: quaternion as numpy array [delta, epsilon1, epsilon2, epsilon3]

    Returns:
        q_up_oplus: 4x4 right multiplication matrix
    """
    q = np.asarray(q).flatten()
    delta = q[0]
    epsilon = q[1:4]

    # q^⊕ = [delta,        -epsilon^T
    #        epsilon,   delta*I - epsilon^×]
    q_up_oplus = np.zeros((4, 4))
    q_up_oplus[0, 0] = delta
    q_up_oplus[0, 1:4] = -epsilon
    q_up_oplus[1:4, 0] = epsilon
    q_up_oplus[1:4, 1:4] = delta * np.eye(3) - up_hat(epsilon)

    return q_up_oplus


def up_star(q):
    """
    Computes the quaternion conjugation.

    Parameters:
        q: quaternion as numpy array [delta, epsilon1, epsilon2, epsilon3]

    Returns:
        q_up_star: conjugated quaternion [delta, -epsilon]
    """
    q = np.asarray(q).flatten()
    delta = q[0]
    epsilon = q[1:4]

    q_up_star = np.concatenate([[delta], -epsilon])

    return q_up_star


def veelog(M):
    """
    Composition of the matrix logarithm and the vee map.
    Inverse of exphat using Rodrigues' formula.

    Parameters:
        M: matrix in SO(3) (3x3) or SE(3) (4x4)

    Returns:
        V: vector in R3 or R6
    """
    M = np.asarray(M)

    R = M[0:3, 0:3]

    if norm(R - np.eye(3)) < 2e-8:
        # Near identity
        if M.shape == (3, 3):
            V = np.zeros(3)
        else:
            V = np.concatenate([np.zeros(3), M[0:3, 3]])
    else:
        theta = np.arccos((np.trace(R) - 1) / 2)
        omega_hat = 1 / (2 * np.sin(theta)) * (R - R.T)
        omega = up_vee(omega_hat)

        if M.shape == (3, 3):
            V = omega * theta
        else:
            # SE(3) case
            A_inv = (
                np.eye(3)
                - theta / 2 * omega_hat
                + (1 - theta / (2 * np.tan(theta / 2))) * omega_hat @ omega_hat
            )
            v = A_inv @ M[0:3, 3]
            V = np.concatenate([omega * theta, v])

    return V
