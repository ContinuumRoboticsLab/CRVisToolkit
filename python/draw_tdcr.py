import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from utils import nullspace, setupfigure


def draw_tdcr(g: np.ndarray[float], seg_end: np.ndarray[int], r_disk: float=2.5*1e-3, r_height: float=1.5*1e-3, tipframe: bool=True, segframe: bool=False, baseframe: bool=False, projections: bool=False, baseplate: bool=True):
    '''
    DRAW_TDCR Creates a figure of a tendon-driven continuum robot (tdcr)

    Takes a matrix with nx16 entries, where n is the number
    of points on the backbone curve. For each point on the curve, the 4x4
    transformation matrix is stored columnwise (16 entries). The x- and
    y-axis span the material orientation and the z-axis is tangent to the
    curve.

    Parameters
    ----------
    g: ndarray
        Backbone curve with n 4x4 transformation matrices reshaped into 1x16 vector (columnwise)
    seg_end: ndarray
        Indices of g where tdcr segments terminate
    r_disk: double
        Radius of spacer disks
    r_height: double
        height of spacer disks
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
    Date: 2023/01/04
    Version: 0.1
    Copyright: 2023 Continuum Robotics Laboratory, University of Toronto
    '''

    # Argument validation
    if g.shape[0] < len(seg_end) or max(seg_end) > g.shape[0]:
        raise ValueError("Dimension mismatch")

    # Setup figure
    ax = setupfigure(g=g, seg_end=seg_end, tipframe=tipframe, segframe=segframe, baseframe=baseframe, projections=projections, baseplate=baseplate)

    numseg = seg_end.size

    if numseg == 1:
        col = np.array([0.8])
    else:
        col = np.linspace(0.2, 0.8, numseg)

    # Backbone
    start = 0
    for i in range(numseg):
        ax.plot(g[start:seg_end[i], 12], g[start:seg_end[i], 13], g[start:seg_end[i], 14], linewidth=5, color=col[i]*np.ones(3))
        start = seg_end[i]

    # Tendons
    tendon1 = np.zeros((seg_end[numseg - 1], 3))
    tendon2 = np.zeros((seg_end[numseg - 1], 3))
    tendon3 = np.zeros((seg_end[numseg - 1], 3))

    # Tendon locations on disk
    r1 = np.array([0, r_disk, 0])
    r2 = np.array([np.cos(30*np.pi/180)*r_disk, -np.sin(30*np.pi/180)*r_disk, 0])
    r3 = np.array([-np.cos(30*np.pi/180)*r_disk, -np.sin(30*np.pi/180)*r_disk, 0])

    for i in range(seg_end[numseg - 1]):
        RotMat = np.array([g[i, 0:3], g[i, 4:7], g[i, 8:11]]).T
        tendon1[i, 0:3] = RotMat@r1 + g[i, 12:15]
        tendon2[i, 0:3] = RotMat@r2 + g[i, 12:15]
        tendon3[i, 0:3] = RotMat@r3 + g[i, 12:15]

    ax.plot(tendon1[:, 0], tendon1[:, 1], tendon1[:, 2], color='k')
    ax.plot(tendon2[:, 0], tendon2[:, 1], tendon2[:, 2], color='k')
    ax.plot(tendon3[:, 0], tendon3[:, 1], tendon3[:, 2], color='k')

    # draw spheres to represent tendon location at end disks
    radius = 0.75e-3

    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]  # Get sphere coordinates
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)

    for i in range(numseg):
        ax.plot_surface(x*radius+tendon1[seg_end[i] - 1, 0], y*radius+tendon1[seg_end[i] - 1, 1], z*radius+tendon1[seg_end[i] - 1, 2], color='k')
        ax.plot_surface(x*radius+tendon2[seg_end[i] - 1, 0], y*radius+tendon2[seg_end[i] - 1, 1], z*radius+tendon2[seg_end[i] - 1, 2], color='k')
        ax.plot_surface(x*radius+tendon3[seg_end[i] - 1, 0], y*radius+tendon3[seg_end[i] - 1, 1], z*radius+tendon3[seg_end[i] - 1, 2], color='k')

    # spacer disks
    seg_idx = 0
    theta = np.arange(0, 2 * np.pi, 0.05)
    for i in range(g.shape[0]):
        if seg_end[seg_idx] < i:
            seg_idx += 1

        color = col[seg_idx]*np.ones(3)

        RotMat = np.array([g[i, 0:3], g[i, 4:7], g[i, 8:11]]).T
        normal = RotMat[:3, 2].T
        v = nullspace(normal)
        v_theta = v[:, 0].reshape((-1, 1)) * np.cos(theta) + v[:, 1].reshape((-1, 1)) * np.sin(theta)

        # Draw the lower circular surface of the disk
        pos = g[i, 12:15].T - RotMat @ np.array([0, 0, r_height/2])
        lowercirc = np.tile(pos.reshape((-1, 1)), theta.size) + r_disk * v_theta
        x, y, z = lowercirc[0, :], lowercirc[1, :], lowercirc[2, :]

        verts = [list(zip(x, y, z))]
        ax.add_collection3d(Poly3DCollection(verts, color=color, edgecolor='k', rasterized=True, zorder=10), zdir='z')

        # Draw the upper circular surface of the disk
        pos = g[i, 12:15].T + RotMat @ np.array([0, 0, r_height/2])
        uppercirc = np.tile(pos.reshape((-1, 1)), theta.size) + r_disk * v_theta
        x, y, z = uppercirc[0, :], uppercirc[1, :], uppercirc[2, :]

        verts = [list(zip(x, y, z))]
        ax.add_collection3d(Poly3DCollection(verts, color=color, edgecolor='k', rasterized=True, zorder=10), zdir='z')

        # Draw the in-between surface of the disk
        x = np.vstack((lowercirc[0, :], uppercirc[0, :]))
        y = np.vstack((lowercirc[1, :], uppercirc[1, :]))
        z = np.vstack((lowercirc[2, :], uppercirc[2, :]))

        ax.plot_surface(x, y, z, color=color, shade=False, zorder=10)

    plt.show()


if "__main__" == __name__:
    import json

    data = {}
    with open("./tdcr_curve_examples.json", "r") as f:
        data = json.load(f)

    # draw_tdcr(np.array(data["onesegtdcr"]), np.array([10]), tipframe=True, segframe=True, baseframe=True, projections=True, baseplate=True)
    # draw_tdcr(np.array(data["threesegtdcr"]), np.array([10, 20, 30]), tipframe=True, segframe=True, baseframe=True, projections=True, baseplate=True)
    draw_tdcr(np.array(data["foursegtdcr"]), np.array([15, 30, 45, 60]), tipframe=True, segframe=True, baseframe=True, projections=True, baseplate=True)
