import numpy as np

def robotindependentmapping(kappa: np.ndarray[np.double], phi: np.ndarray[np.double], ell: np.ndarray[np.double], ptsperseg: np.ndarray[np.uint8]) -> np.ndarray[np.double]:
    """
    ROBOTINDEPENDENTMAPPING creates a framed curve for given configuration parameters

    Example
    -------
    g = robotindependentmapping([1/40e-3;1/10e-3],[0,pi],[25e-3,20e-3],10)
    creates a 2-segment curve with radius of curvatures 1/40 and 1/10
    and segment lengths 25 and 20, where the second segment is rotated by pi rad.

    Parameters
    ------
    kappa: ndarray
        (nx1) segment curvatures
    phi: ndarray
        (nx1) segment bending plane angles
    l: ndarray
        (nx1) segment lengths
    ptsperseg: ndarray
        (nx1) number of points per segment
            if n=1 all segments with equal number of points

    Returns
    -------
    g : ndarray
        (mx16) backbone curve with m 4x4 transformation matrices, where m is
            total number of points, reshaped into 1x16 vector (columnwise)

    Author: Jessica Burgner-Kahrs <jbk@cs.toronto.edu>
    Date: 2022/02/16
    Version: 0.2
    Copyright: 2023 Continuum Robotics Laboratory, University of Toronto
    """

    if kappa.shape != phi.shape or kappa.shape != ell.shape:
        raise ValueError("Dimension mismatch.")
    
    numseg = kappa.shape[0]
    if ptsperseg.size == 1 and numseg > 1: # If same number of points per segment
        ptsperseg = np.tile(ptsperseg, numseg)  # Create an array that is numseg long with the num points repeated

    g = np.zeros((np.sum(ptsperseg), 16))  # Stores the transformation matrices of all the points in all the segments as rows

    T_base = np.eye(4)  # base starts off as identity
    for i in range(numseg):
        c_p = np.cos(phi[i])
        s_p = np.sin(phi[i])

        for j in range(ptsperseg[i]):
            c_ks = np.cos(kappa[i] * j * (ell[i]/ptsperseg[i]))
            s_ks = np.sin(kappa[i] * j * (ell[i]/ptsperseg[i]))

            T_temp = np.array([
                [c_p*c_p*(c_ks-1) + 1, s_p*c_p*(c_ks-1), -c_p*s_ks, 0],
                [s_p*c_p*(c_ks-1), c_p*c_p*(1-c_ks) + c_ks, -s_p*s_ks, 0],
                [c_p*s_ks, s_p*s_ks, c_ks, 0],
                [0, 0, j*(ell[i]/ptsperseg[i]), 1]
            ])

            if kappa[i] != 0:  # No division by zero possible
                T_temp[3, :] = [(c_p*(1-c_ks))/kappa[i], (s_p*(1-c_ks))/kappa[i], s_ks/kappa[i], 1]

            g[i*numseg + j, :] = (T_base @ T_temp).reshape(1, 16)  # populate with transformation matrix for that point

        T_base = g[i*numseg + ptsperseg[i] - 1, :].reshape(4, 4)  # lastmost point's transformation matrix is the new base

    return g
