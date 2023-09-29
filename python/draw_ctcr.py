import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from utils import robotindependentmapping


# TODO: Fix the triangular patches issue
def draw_ctcr(g: np.ndarray[np.ndarray[float]], tube_end: np.ndarray[int], r_tube: np.ndarray[float], tipframe: bool=False, baseframe: bool=False, projections:bool=False, baseplate:bool=False) -> None:
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
    baseframe: bool, default=False
        Shows robot base frame
    projections: bool, default=False
        Shows projections of backbone curve onto coordinate axes
    baseplate: bool, default=False
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

    curvelength = np.sum(np.linalg.norm(g[1:, 12:15] - g[:-1, 12:15], axis=1))
    numtubes = tube_end.size

    ## Setup figure
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.set_box_aspect([1,1,1])

    # # Axes, Labels

    fig = plt.figure()
    fig.set_size_inches(1280/100, 1024/100) # converting pixel dimensions to inches
    ax = fig.add_subplot(111, projection='3d')

    radial_pts = 16  # resolution (num point on circular cross section, increase for high detail level)
    tcirc = np.linspace(0, 2*np.pi, radial_pts)
    col = np.linspace(0.2, 0.8, numtubes)
    alpha = 1  # 0 = transparent

    ## draw tubes
    start_tube = 0
    for j in range(numtubes):
        end_tube = tube_end[j]
        color = [col[j], col[j], col[j]]

        # points on a circle in the local x-y plane
        basecirc = np.vstack([r_tube[j] * np.sin(tcirc), r_tube[j] * np.cos(tcirc), np.zeros(radial_pts), np.ones(radial_pts)])

        # transform circle points into the tube's base frame
        # T_point = g[start_tube, :].reshape((4, 4))
        # basecirc_trans = T_point @ basecirc

        # draw patches to fill in the circle
        # ax.plot_trisurf(basecirc_trans[0, :], basecirc_trans[1, :], basecirc_trans[2, :], color=color, edgecolor='none', alpha=alpha, linewidth=0, antialiased=True, shade=True)
        # patch(ax, basecirc_trans[0, :], basecirc_trans[1, :], basecirc_trans[2, :], color)

        # Loop to draw each cylindrical segment for tube
        seg_x, seg_y, seg_z = [], [], []
        for i in range(start_tube, end_tube):
            T_point = g[i, :].reshape((4, 4)).T
            basecirc_trans = T_point @ basecirc
            seg_x.extend(basecirc_trans[0, :])
            seg_y.extend(basecirc_trans[1, :])
            seg_z.extend(basecirc_trans[2, :])

            # basecirc_trans = g[i, :].reshape(4, 4) @ basecirc  # current frame circle points
            # basecirc_trans_ahead = g[i + 1, :].reshape(4, 4) @ basecirc  # next frame circle points

            # loop to draw each square patch for this segment
            # for k in range(radial_pts - 1):
            #     xedge = np.array([basecirc_trans[0, k], basecirc_trans[0, k + 1], basecirc_trans_ahead[0, k + 1], basecirc_trans_ahead[0, k]])
            #     yedge = np.array([basecirc_trans[1, k], basecirc_trans[1, k + 1], basecirc_trans_ahead[1, k + 1], basecirc_trans_ahead[1, k]])
            #     zedge = np.array([basecirc_trans[2, k], basecirc_trans[2, k + 1], basecirc_trans_ahead[2, k + 1], basecirc_trans_ahead[2, k]])

            #     print(k)
            #     print(xedge)
            #     print(yedge)
            #     print(zedge)
            #     print()

            #     patch(ax, xedge, yedge, zedge, color)

                # ax.plot_trisurf(xedge, yedge, zedge, color=color, edgecolor='none', alpha=alpha, linewidth=0)
                # verts = [list(zip(xedge, yedge, zedge))]
                # ax.add_collection(Poly3DCollection(verts, alpha=alpha, facecolor=color, edgecolor='none', linewidth=0))

        ax.plot_trisurf(seg_x, seg_y, seg_z)
        start_tube = end_tube

    # Projections
    if projections:
        ax.plot(g[:,12], np.ones(g.shape)*ax.get_ylim()[0], g[:,14], color='k', linewidth=2) # project in x-z axis
        ax.plot(np.ones(g.shape)*ax.get_xlim()[0], g[:,13], g[:,14], color='k', linewidth=2) # project in y-z axis
        ax.plot(g[:,12], g[:,13], np.zeros(g.shape), color='k', linewidth=2) # project in x-y axis

    # Base Plate
    if baseplate:
        color = [0.9, 0.9, 0.9]
        squaresize = 0.02
        thickness = 0.001
        
        x = np.array([-1, 1, 1, -1]) * squaresize
        y = np.array([-1, -1, 1, 1]) * squaresize
        z = np.array([-1, -1, -1, -1]) * thickness
        ax.plot_trisurf(x, y, z, color=color)

        x = np.array([-1, 1, 1, -1]) * squaresize
        y = np.array([-1, -1, 1, 1]) * squaresize
        z = np.array([0, 0, 0, 0])
        ax.plot_trisurf(x, y, z, color=color)

        x = np.array([1, 1, 1, 1]) * squaresize
        y = np.array([-1, -1, 1, 1]) * squaresize
        z = np.array([-1, 0, 0, -1]) * thickness
        ax.plot_trisurf(x, y, z, color=color)

        x = np.array([-1, -1, -1, -1]) * squaresize
        y = np.array([-1, -1, 1, 1]) * squaresize
        z = np.array([-1, 0, 0, -1]) * thickness
        ax.plot_trisurf(x, y, z, color=color)

        x = np.array([-1, 1, 1, -1]) * squaresize
        y = np.array([-1, -1, -1, -1]) * squaresize
        z = np.array([-1, -1, 0, 0]) * thickness
        ax.plot_trisurf(x, y, z, color=color)

        x = np.array([-1, 1, 1, -1]) * squaresize
        y = np.array([1, 1, 1, 1]) * squaresize
        z = np.array([-1, -1, 0, 0]) * thickness
        ax.plot_trisurf(x, y, z, color=color)

    # Frames
    if tipframe:
        ax.quiver(g[-1,12], g[-1,13], g[-1,14], g[-1,0], g[-1,1], g[-1,2], length=0.01, linewidth=3, color='r')
        ax.quiver(g[-1,12], g[-1,13], g[-1,14], g[-1,4], g[-1,5], g[-1,6], length=0.01, linewidth=3, color='g')
        ax.quiver(g[-1,12], g[-1,13], g[-1,14], g[-1,8], g[-1,9], g[-1,10], length=0.01, linewidth=3, color='b')

    # Base Frame
    if baseframe:
        ax.quiver(0, 0, 0, 1, 0, 0, length=0.01, linewidth=3, color='r')
        ax.quiver(0, 0, 0, 0, 1, 0, length=0.01, linewidth=3, color='g')
        ax.quiver(0, 0, 0, 0, 0, 1, length=0.01, linewidth=3, color='b')

    clearance = 0.03
    max_val_x = np.max(np.abs(g[:, 12])) + clearance
    max_val_y = np.max(np.abs(g[:, 13])) + clearance
    ax.set_box_aspect([1, 1, 1]) # set aspect ratio of the plot
    ax.set_xlim(-max_val_x, max_val_x)
    ax.set_ylim(-max_val_y, max_val_y)
    ax.set_zlim(0, curvelength + clearance)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.grid(alpha=0.3)
    ax.view_init(elev=30, azim=-45)

    plt.show()


if "__main__" == __name__:
    kappa = np.array([1/30e-3, 1/40e-3, 1/15e-3])
    phi = np.array([0, np.deg2rad(160), np.deg2rad(30)])
    ell = np.array([50e-3, 70e-3, 25e-3])
    pts_per_seg = np.array([30])

    g = robotindependentmapping(kappa, phi, ell, pts_per_seg)
    draw_ctcr(g, np.array([30, 60, 90]), np.array([2e-3, 1.5e-3, 1e-3]))

    # fig = plt.figure()
    # # fig.set_size_inches(1280/100, 1024/100) # converting pixel dimensions to inches
    # ax = fig.add_subplot(111, projection='3d')

    # x = np.linspace(-10, 10, 100)
    # x, y = np.meshgrid(x, x)
    # z = x*y

    # # ax.plot_trisurf(x, y, x**2)
    # ax.plot(x.flatten(), y.flatten(), z.flatten())
    # plt.show()
