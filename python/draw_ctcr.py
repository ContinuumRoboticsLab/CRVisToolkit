import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from utils import robotindependentmapping, setupfigure


def draw_ctcr(g: np.ndarray[np.ndarray[float]], tube_end: np.ndarray[int], r_tube: np.ndarray[float], tipframe: bool=True, segframe: bool=False, baseframe: bool=False, projections: bool=False, baseplate: bool=True) -> None:
    '''
    DRAW_CTCR Creates a figure of a concentric tube continuum robot (ctcr)

    Takes a matrix with nx16 entries, where n is the number
    of points on the backbone curve. For each point on the curve, the 4x4
    transformation matrix is stored columnwise (16 entries). The x- and
    y-axis span the material orientation and the z-axis is tangent to the
    curve.

    Parameters
    ----------
    g: ndarray
        Backbone curve with n 4x4 transformation matrices reshaped into 1x16 vector (columnwise)
    tube_end: ndarray
        Indices of g where ctcr tubes terminate
    r_tube: ndarray
        Radii of tubes
    tipframe: bool, default=True
        Shows tip frame
    segframe: bool, default=False
        Shows segment end frames
    baseframe: bool, default=False
        Shows robot base frame
    projections: bool, default=False
        Shows projections of backbone curve onto coordinate axes
    baseplate: bool, default=True
        Shows robot base plate

    Outputs
    -------
    A matplotlib figure object
        Figure object of the generated plot

    Author: Jessica Burgner-Kahrs <jbk@cs.toronto.edu>
    Date: 2023/02/16
    Version: 0.2
    Copyright: 2023 Continuum Robotics Laboratory, University of Toronto
    '''

    if np.max(tube_end) > g.shape[0] or tube_end.size != r_tube.size:
        raise ValueError("Dimension mismatch")

    # Setup figures
    ax = setupfigure(g=g, seg_end=tube_end, tipframe=tipframe, segframe=segframe, baseframe=baseframe, projections=projections, baseplate=baseplate)

    numtubes = tube_end.size
    radial_pts = 16  # resolution (num point on circular cross section, increase for high detail level)
    tcirc = np.linspace(0, 2*np.pi, radial_pts)
    alpha = 1  # 0 = transparent

    if numtubes == 1:
        col = np.array([0.8])
    else:
        col = np.linspace(0.2, 0.8, numtubes)

    ## draw tubes
    start_tube = 0
    for j in range(numtubes):
        end_tube = tube_end[j]
        color = [col[j], col[j], col[j]]

        # points on a circle in the local x-y plane
        basecirc = np.vstack([r_tube[j] * np.sin(tcirc), r_tube[j] * np.cos(tcirc), np.zeros(radial_pts), np.ones(radial_pts)])

        # Loop to draw each cylindrical segment for tube
        for i in range(start_tube, end_tube - 1):
            # Get the coordinates of the current segment
            T_point = g[i, :].reshape((4, 4)).T
            basecirc_trans = T_point @ basecirc

            # Get the coordinates of the next segment
            T_point = g[i + 1, :].reshape((4, 4)).T
            basecirc_trans_ahead = T_point @ basecirc

            # Loop to draw each radial point for this entire segment
            for k in range(radial_pts - 1):
                xedge = np.array([basecirc_trans[0, k], basecirc_trans[0, k + 1], basecirc_trans_ahead[0, k + 1], basecirc_trans_ahead[0, k]])
                yedge = np.array([basecirc_trans[1, k], basecirc_trans[1, k + 1], basecirc_trans_ahead[1, k + 1], basecirc_trans_ahead[1, k]])
                zedge = np.array([basecirc_trans[2, k], basecirc_trans[2, k + 1], basecirc_trans_ahead[2, k + 1], basecirc_trans_ahead[2, k]])

                verts = [list(zip(xedge, yedge, zedge))]
                ax.add_collection3d(Poly3DCollection(verts, alpha=alpha, color=color, rasterized=True, zorder=10), zdir='z')

        start_tube = end_tube

    plt.show()


if "__main__" == __name__:
    kappa = np.array([1/30e-3, 1/40e-3, 1/15e-3])
    phi = np.array([0, np.deg2rad(160), np.deg2rad(30)])
    ell = np.array([50e-3, 70e-3, 25e-3])
    pts_per_seg = np.array([30])

    g = robotindependentmapping(kappa, phi, ell, pts_per_seg)
    draw_ctcr(g, np.array([30, 60, 90]), np.array([2e-3, 1.5e-3, 1e-3]), tipframe=True, segframe=True, baseframe=True, projections=True, baseplate=True)
