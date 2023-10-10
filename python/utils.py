import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def setupfigure(g: np.ndarray[float], seg_end: np.ndarray[int], tipframe: bool, segframe: bool, baseframe: bool, projections: bool, baseplate: bool):
    """
    Sets up the matplotlib figure of the model along with the different visuals

    Parameters
    ------
    g: ndarray
        Backbone curve with n 4x4 transformation matrices reshaped into 1x16 vector (columnwise)
    seg_end: ndarray
        Indices of g where tdcr segments terminate
    tipframe: bool
        Shows tip frame
    segframe: bool
        Shows segment end frames
    baseframe: bool
        Shows robot base frame
    projections: bool
        Shows projections of backbone curve onto coordinate axes
    baseplate: bool
        Shows robot base plate

    Returns
    -------
    ax : Axis3D
        Matplotlib 3D figure
    """

    # Setup figures
    fig = plt.figure()
    fig.set_size_inches(1280/fig.dpi, 1024/fig.dpi) # converting pixel dimensions to inches
    ax = fig.add_subplot(projection='3d', computed_zorder=False)

    # Axes, Labels
    clearance = 0.03
    curvelength = np.sum(np.linalg.norm(g[1:, 12:15] - g[:-1, 12:15], axis=1))
    max_val_x = np.max(np.abs(g[:, 12])) + clearance
    max_val_y = np.max(np.abs(g[:, 13])) + clearance
    ax.set_box_aspect([1, 1, 1]) # set aspect ratio of the plot
    ax.set_xlim(-max_val_x, max_val_x)
    ax.set_ylim(-max_val_y, max_val_y)
    ax.set_zlim(0, curvelength + clearance)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.grid(True, alpha=0.3)
    ax.view_init(azim=45, elev=30)

    # Projections
    if projections:
        ax.plot(np.ones(g.shape[0])*ax.get_xlim()[0], g[:, 13], g[:, 14], linewidth=2, color='r') # project in y-z axis
        ax.plot(g[:, 12], np.ones(g.shape[0])*ax.get_ylim()[0], g[:, 14], linewidth=2, color='g') # project in x-z axis
        ax.plot(g[:, 12], g[:, 13], np.zeros(g.shape[0]), linewidth=2, color='b') # project in x-y axis

    # Base Plate
    if baseplate:
        color = [0.9, 0.9, 0.9]
        squaresize = 0.02
        thickness = 0.001
        
        x = np.array([-1, 1, 1, -1]) * squaresize
        y = np.array([-1, -1, 1, 1]) * squaresize
        z = np.array([-1, -1, -1, -1]) * thickness
        verts = [list(zip(x, y, z))]
        ax.add_collection3d(Poly3DCollection(verts, color=color, rasterized=True, zorder=-1), zdir='z')

        x = np.array([-1, 1, 1, -1]) * squaresize
        y = np.array([-1, -1, 1, 1]) * squaresize
        z = np.array([0, 0, 0, 0])
        verts = [list(zip(x, y, z))]
        ax.add_collection3d(Poly3DCollection(verts, color=color, rasterized=True, zorder=-1), zdir='z')

        x = np.array([1, 1, 1, 1]) * squaresize
        y = np.array([-1, -1, 1, 1]) * squaresize
        z = np.array([-1, 0, 0, -1]) * thickness
        verts = [list(zip(x, y, z))]
        ax.add_collection3d(Poly3DCollection(verts, color=color, rasterized=True, zorder=-1), zdir='z')

        x = np.array([-1, -1, -1, -1]) * squaresize
        y = np.array([-1, -1, 1, 1]) * squaresize
        z = np.array([-1, 0, 0, -1]) * thickness
        verts = [list(zip(x, y, z))]
        ax.add_collection3d(Poly3DCollection(verts, color=color, rasterized=True, zorder=-1), zdir='z')

        x = np.array([-1, 1, 1, -1]) * squaresize
        y = np.array([-1, -1, -1, -1]) * squaresize
        z = np.array([-1, -1, 0, 0]) * thickness
        verts = [list(zip(x, y, z))]
        ax.add_collection3d(Poly3DCollection(verts, color=color, rasterized=True, zorder=-1), zdir='z')

        x = np.array([-1, 1, 1, -1]) * squaresize
        y = np.array([1, 1, 1, 1]) * squaresize
        z = np.array([-1, -1, 0, 0]) * thickness
        verts = [list(zip(x, y, z))]
        ax.add_collection3d(Poly3DCollection(verts, color=color, rasterized=True, zorder=-1), zdir='z')

    # Coordinate Frames
    if tipframe and not segframe:
        ax.quiver(g[-1,12], g[-1,13], g[-1,14], g[-1,0], g[-1,1], g[-1,2], length=0.01, linewidth=3, color='r')
        ax.quiver(g[-1,12], g[-1,13], g[-1,14], g[-1,4], g[-1,5], g[-1,6], length=0.01, linewidth=3, color='g')
        ax.quiver(g[-1,12], g[-1,13], g[-1,14], g[-1,8], g[-1,9], g[-1,10], length=0.01, linewidth=3, color='b')

    if segframe:
        for i in range(seg_end.size):
            seg_end_idx = seg_end[i] - 1
            ax.quiver(g[seg_end_idx, 12], g[seg_end_idx, 13], g[seg_end_idx, 14], g[seg_end_idx, 0], g[seg_end_idx, 1], g[seg_end_idx, 2], length=0.01, linewidth=3, color='r')
            ax.quiver(g[seg_end_idx, 12], g[seg_end_idx, 13], g[seg_end_idx, 14], g[seg_end_idx, 4], g[seg_end_idx, 5], g[seg_end_idx, 6], length=0.01, linewidth=3, color='g')
            ax.quiver(g[seg_end_idx, 12], g[seg_end_idx, 13], g[seg_end_idx, 14], g[seg_end_idx, 8], g[seg_end_idx, 9], g[seg_end_idx, 10], length=0.01, linewidth=3, color='b')

    # Base Frame
    if baseframe:
        ax.quiver(0, 0, 0, 1, 0, 0, length=0.01, linewidth=3, color='r')
        ax.quiver(0, 0, 0, 0, 1, 0, length=0.01, linewidth=3, color='g')
        ax.quiver(0, 0, 0, 0, 0, 1, length=0.01, linewidth=3, color='b')

    return ax


def nullspace(A: np.ndarray[float], atol: float=1e-13, rtol: float=0):
    """
    Compute an approximate basis for the nullspace of A.

    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length k will be treated
        as a 2-D with shape (1, k)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    Return value
    ------------
    ns : ndarray
        If `A` is an array with shape (m, k), then `ns` will be an array
        with shape (k, n), where n is the estimated dimension of the
        nullspace of `A`.  The columns of `ns` are a basis for the
        nullspace; each element in numpy.dot(A, ns) will be approximately
        zero.
    """

    # Source: https://scipy-cookbook.readthedocs.io/items/RankNullspace.html

    A = np.atleast_2d(A)
    u, s, vh = np.linalg.svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns


def robotindependentmapping(kappa: np.ndarray[float], phi: np.ndarray[float], ell: np.ndarray[float], pts_per_seg: np.ndarray[int]) -> np.ndarray[float]:
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
    pts_per_seg: ndarray
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

    numseg = kappa.size
    if pts_per_seg.size == 1 and numseg > 1: # If same number of points per segment
        pts_per_seg = np.tile(pts_per_seg, numseg)  # Create an array that is numseg long with the num points repeated

    g = np.zeros((np.sum(pts_per_seg), 16))  # Stores the transformation matrices of all the points in all the segments as rows

    p_count = 0  # Points counter
    T_base = np.eye(4)  # base starts off as identity
    for i in range(numseg):
        c_p = np.cos(phi[i])
        s_p = np.sin(phi[i])

        for j in range(pts_per_seg[i]):
            c_ks = np.cos(kappa[i] * j * (ell[i]/pts_per_seg[i]))
            s_ks = np.sin(kappa[i] * j * (ell[i]/pts_per_seg[i]))

            T_temp = np.array([
                [ c_p*c_p*(c_ks-1) + 1, s_p*c_p*(c_ks-1),        c_p*s_ks, 0],
                [ s_p*c_p*(c_ks-1),     c_p*c_p*(1-c_ks) + c_ks, s_p*s_ks, 0],
                [-c_p*s_ks,            -s_p*s_ks,                c_ks,     0],
                [ 0,                    0,                       0,        0]
            ])

            if kappa[i] != 0:
                T_temp[:, 3] = [(c_p*(1-c_ks))/kappa[i], (s_p*(1-c_ks))/kappa[i], s_ks/kappa[i], 1]
            else:  # To avoid division by zero
                T_temp[:, 3] = [0, 0, j*(ell[i]/pts_per_seg[i]), 1]

            g[p_count, :] = (T_base @ T_temp).T.reshape((1, 16))  # A matlab reshape is column-wise and not row-wise
            p_count += 1

        T_base = g[p_count - 1, :].reshape(4, 4).T  # lastmost point's transformation matrix is the new base

    return g
