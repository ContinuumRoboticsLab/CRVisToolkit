import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def null_space(A, rcond=None):
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    M, N = u.shape[0], vh.shape[1]
    if rcond is None:
        rcond = np.finfo(s.dtype).eps * max(M, N)
    tol = np.amax(s) * rcond
    num = np.sum(s > tol, dtype=int)
    Q = vh[num:,:].T.conj()
    return Q


def draw_tdcr(g: np.ndarray[np.double], seg_end: np.ndarray[np.uint8], r_disk: np.double=2.5*1e-3, r_height: np.double=1.5*1e-3, tipframe: bool=True, segframe: bool=False, baseframe: bool=False, projections: bool=False, baseplate: bool=False):
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
    baseplate: bool, default=False
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

    numseg = len(seg_end)
    curvelength = np.sum(np.linalg.norm(g[1:, 12:15].T - g[:-1, 12:15].T))
    
    # Setup figure
    fig = plt.figure()
    fig.set_size_inches(1280/fig.dpi, 1024/fig.dpi)
    ax = fig.add_subplot(projection='3d')
    
    col = np.linspace(0.2, 0.8, numseg)

    # Backbone
    ax.plot3D(g[:seg_end[0], 12], g[:seg_end[0], 13], g[:seg_end[0], 14], linewidth=5, color=[col[0], col[0], col[0]])
    for i in range(numseg):
        start = 0 if i == 0 else seg_end[i], seg_end[i, 0]
        ax.plot3D(g[seg_end[i - 1]:seg_end[i], 12], g[seg_end[i]:seg_end[i + 1] + 1, 13], g[seg_end[i]:seg_end[i + 1], 14], linewidth=5, color=col[i]*np.ones((3,)))
    
    # Projections
    if projections:
        ax.plot3D(g[:, 12], np.full(g.shape[0], ax.get_ylim()[0]), g[:, 14], linewidth=2, color=[0, 1, 0]) # project in x-z axis
        ax.plot3D(np.full(g.shape[0], ax.get_xlim()[0]), g[:, 13], g[:, 14], linewidth=2, color=[1, 0, 0]) # project in y-z axis
        ax.plot3D(g[:, 12], g[:, 13], np.zeros(g.shape[0]), linewidth=2, color=[0, 0, 1]) # project in x-y axis
    
    # Tendons
    tendon1 = np.zeros((seg_end[numseg - 1, 0], 3))
    tendon2 = np.zeros((seg_end[numseg - 1, 0], 3))
    tendon3 = np.zeros((seg_end[numseg - 1, 0], 3))
    
    # Tendon locations on disk
    r1 = np.array([0, r_disk, 0]).T
    r2 = np.array([np.cos(30*np.pi/180)*r_disk, -np.sin(30*np.pi/180)*r_disk, 0]).T
    r3 = np.array([-np.cos(30*np.pi/180)*r_disk, -np.sin(30*np.pi/180)*r_disk, 0]).T

    for i in range(int(seg_end[numseg - 1])):
        RotMat = np.reshape([g[i, 0:3], g[i, 4:7], g[i, 8:11]], (3, 3))
        tendon1[i, 0:3] = np.dot(RotMat, r1) + g[i, 12:15]
        tendon2[i, 0:3] = np.dot(RotMat, r2) + g[i, 12:15]
        tendon3[i, 0:3] = np.dot(RotMat, r3) + g[i, 12:15]

    ax.plot3D(tendon1[:, 0], tendon1[:, 1], tendon1[:, 2], color='black')
    ax.plot3D(tendon2[:, 0], tendon2[:, 1], tendon2[:, 2], color='black')
    ax.plot3D(tendon3[:, 0], tendon3[:, 1], tendon3[:, 2], color='black')

    # draw spheres to represent tendon location at end disks
    radius = 0.75e-3

    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]  # Get sphere coordinates
    x=np.cos(u)*np.sin(v)
    y=np.sin(u)*np.sin(v)
    z=np.cos(v)

    for i in range(numseg):
        ax.plot_surface(x*radius+tendon1[seg_end[i, 0] - 1, 0], y*radius+tendon1[seg_end[i, 0] - 1, 1], z*radius+tendon1[seg_end[i, 0] - 1, 2], color='black')
        ax.plot_surface(x*radius+tendon2[seg_end[i, 0] - 1, 0], y*radius+tendon2[seg_end[i, 0] - 1, 1], z*radius+tendon2[seg_end[i, 0] - 1, 2], color='black')
        ax.plot_surface(x*radius+tendon3[seg_end[i, 0] - 1, 0], y*radius+tendon3[seg_end[i, 0] - 1, 1], z*radius+tendon3[seg_end[i, 0] - 1, 2], color='black')

    # spacer disks
    for i in range(g.shape[0]):
        seg = np.where(seg_end >= i)[0]
        if seg.size == 0:
            color = col[0] * np.array([1, 1, 1])
        else:
            color = col[seg[0]] * np.array([1, 1, 1])

        RotMat = g[i, [0, 1, 2, 4, 5, 6, 8, 9, 10]].reshape(3, 3)
        normal = RotMat[:3, 2].T
        pos = g[i, 12:15].T - RotMat @ np.array([0, 0, r_height/2])

        theta = np.arange(0, 2 * np.pi + 0.05, 0.05)
        v = null_space(normal)
        lowercirc = np.tile(pos.reshape(-1, 1), (1, theta.size)) + r_disk * (v[:, 0].reshape(-1, 1) * np.cos(theta) + v[:, 1].reshape(-1, 1) * np.sin(theta))
        ax.plot_surface(lowercirc[0, :], lowercirc[1, :], lowercirc[2, :], facecolors=color, shade=False)

        pos = g[i, 12:15] + RotMat @ np.array([0, 0, r_height/2])
        uppercirc = np.tile(pos.reshape(-1, 1), (1, theta.size)) + r_disk * (v[:, 0].reshape(-1, 1) * np.cos(theta) + v[:, 1].reshape(-1, 1) * np.sin(theta))
        ax.plot_surface(uppercirc[0, :], uppercirc[1, :], uppercirc[2, :], facecolors=color, shade=False)

        x = np.vstack((lowercirc[0, :], uppercirc[0, :]))
        y = np.vstack((lowercirc[1, :], uppercirc[1, :]))
        z = np.vstack((lowercirc[2, :], uppercirc[2, :]))

        ax.plot_surface(x, y, z, facecolors=color, shade=False)

    # base plate
    if baseplate:
        color = [1, 1, 1] * 0.9
        squaresize = 0.02
        thickness = 0.001
        ax.gca().add_patch(plt.Polygon([[-squaresize, -squaresize, squaresize, squaresize], [-squaresize, squaresize, squaresize, -squaresize]], [-thickness, -thickness, -thickness, -thickness], color=color))
        ax.gca().add_patch(plt.Polygon([[-squaresize, -squaresize, squaresize, squaresize], [-squaresize, squaresize, squaresize, -squaresize]], [0, 0, 0, 0], color=color))
        ax.gca().add_patch(plt.Polygon([[squaresize, squaresize, squaresize, squaresize], [-squaresize, squaresize, squaresize, -squaresize]], [-thickness, 0, 0, -thickness], color=color))
        ax.gca().add_patch(plt.Polygon([[-squaresize, -squaresize, -squaresize, -squaresize], [-squaresize, squaresize, squaresize, -squaresize]], [-thickness, 0, 0, -thickness], color=color))
        ax.gca().add_patch(plt.Polygon([[-squaresize, -squaresize, squaresize, squaresize], [-squaresize, -squaresize, -squaresize, -squaresize]], [-thickness, -thickness, 0, 0], color=color))
        ax.gca().add_patch(plt.Polygon([[-squaresize, -squaresize, squaresize, squaresize], [squaresize, squaresize, squaresize, squaresize]], [-thickness, -thickness, 0, 0], color=color))

    # Coordinate Frames
    if tipframe and not segframe:
        ax.quiver(g[-1, 12], g[-1, 13], g[-1, 14], g[-1, 0], g[-1, 1], g[-1, 2], color=[1, 0, 0], linewidth=3, length=0.01)
        ax.quiver(g[-1, 12], g[-1, 13], g[-1, 14], g[-1, 4], g[-1, 5], g[-1, 6], color=[0, 1, 0], linewidth=3, length=0.01)
        ax.quiver(g[-1, 12], g[-1, 13], g[-1, 14], g[-1, 8], g[-1, 9], g[-1, 10], color=[0, 0, 1], linewidth=3, length=0.01)

    if segframe:
        for i in range(numseg):
            ax.quiver(g[seg_end[i], 12], g[seg_end[i], 13], g[seg_end[i], 14], g[seg_end[i], 0], g[seg_end[i], 1], g[seg_end[i], 2], color=[1, 0, 0], linewidth=3, length=0.01)
            ax.quiver(g[seg_end[i], 12], g[seg_end[i], 13], g[seg_end[i], 14], g[seg_end[i], 4], g[seg_end[i], 5], g[seg_end[i], 6], color=[0, 1, 0], linewidth=3, length=0.01)
            ax.quiver(g[seg_end[i], 12], g[seg_end[i], 13], g[seg_end[i], 14], g[seg_end[i], 8], g[seg_end[i], 9], g[seg_end[i], 10], color=[0, 0, 1], linewidth=3, length=0.01)

    if baseframe:
        ax.quiver(0, 0, 0, 1, 0, 0, color=[1, 0, 0], linewidth=3, length=0.01)
        ax.quiver(0, 0, 0, 0, 1, 0, color=[0, 1, 0], linewidth=3, length=0.01)
        ax.quiver(0, 0, 0, 0, 0, 1, color=[0, 0, 1], linewidth=3, length=0.01)

    # Axes, Labels
    clearance = 0.03
    ax.set_xlim(-(np.max(np.abs(g[:, 12])) + clearance), np.max(np.abs(g[:, 12])) + clearance)
    ax.set_ylim(-(np.max(np.abs(g[:, 13])) + clearance), np.max(np.abs(g[:, 13])) + clearance)
    ax.set_zlim(0, curvelength + clearance)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.grid(True, alpha=0.3)
    ax.view_init(azim=45, elev=30)
    ax.set_box_aspect([1, 1, 1])

    plt.show()


if "__main__" == __name__:
    import json

    with open("example_data.json", "r") as f:
        data = json.load(f)

    draw_tdcr(np.array(data["onesegtdcr"]), np.array([10]).reshape((1, 1)))
    draw_tdcr(np.array(data["threesegtdcr"]), np.array([10, 20, 30]).reshape((3, 1)), projections=True)
    draw_tdcr(np.array(data["foursegtdcr"]), np.array([15, 30, 45, 60]).reshape((4, 1)), segframe=True, baseframe=True)
