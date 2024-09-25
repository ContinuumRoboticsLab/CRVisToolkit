import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from common.utils import robotindependentmapping, setupfigure
from common.types import CTCRPlotterSettings, CRDiscreteCurve


def draw_ctcr(
    curve: CRDiscreteCurve,
    plotter_settings: CTCRPlotterSettings = CTCRPlotterSettings(),
) -> None:
    """
    DRAW_CTCR Creates a figure of a concentric tube continuum robot (ctcr)

    Takes a matrix with nx16 entries, where n is the number
    of points on the backbone curve. For each point on the curve, the 4x4
    transformation matrix is stored columnwise (16 entries). The x- and
    y-axis span the material orientation and the z-axis is tangent to the
    curve.

    Parameters
    ----------
    curve: CRDiscreteCurve
        a discrete point representation of a CR curve with multiple segments
    plotter_settings: CTCRPlotterSettings
        The paramters required to specify how a CTCR should be plotted. The
        defaults are as specified in the class definition.

    Outputs
    -------
    A matplotlib figure object
        Figure object of the generated plot

    Author: Jessica Burgner-Kahrs <jbk@cs.toronto.edu>
    Date: 2023/02/16
    Version: 0.2
    Copyright: 2023 Continuum Robotics Laboratory, University of Toronto
    """

    g = curve.g

    if (
        np.max(curve.seg_end) > curve.g.shape[0]
        or curve.seg_end.size != plotter_settings.r_tube.size
    ):
        raise ValueError("Dimension mismatch")

    # Setup figures
    ax = setupfigure(curve, plotter_settings)

    numtubes = curve.seg_end.size
    radial_pts = 16  # resolution (num point on circular cross section, increase for high detail level)
    tcirc = np.linspace(0, 2 * np.pi, radial_pts)
    alpha = 1  # 0 = transparent

    if numtubes == 1:
        col = np.array([0.8])
    else:
        col = np.linspace(0.2, 0.8, numtubes)

    ## draw tubes
    start_tube = 0
    for j in range(numtubes):
        end_tube = curve.seg_end[j]
        color = [col[j], col[j], col[j]]

        # points on a circle in the local x-y plane
        basecirc = np.vstack(
            [
                plotter_settings.r_tube[j] * np.sin(tcirc),
                plotter_settings.r_tube[j] * np.cos(tcirc),
                np.zeros(radial_pts),
                np.ones(radial_pts),
            ]
        )

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
                xedge = np.array(
                    [
                        basecirc_trans[0, k],
                        basecirc_trans[0, k + 1],
                        basecirc_trans_ahead[0, k + 1],
                        basecirc_trans_ahead[0, k],
                    ]
                )
                yedge = np.array(
                    [
                        basecirc_trans[1, k],
                        basecirc_trans[1, k + 1],
                        basecirc_trans_ahead[1, k + 1],
                        basecirc_trans_ahead[1, k],
                    ]
                )
                zedge = np.array(
                    [
                        basecirc_trans[2, k],
                        basecirc_trans[2, k + 1],
                        basecirc_trans_ahead[2, k + 1],
                        basecirc_trans_ahead[2, k],
                    ]
                )

                verts = [list(zip(xedge, yedge, zedge))]
                ax.add_collection3d(
                    Poly3DCollection(
                        verts, alpha=alpha, color=color, rasterized=True, zorder=10
                    ),
                    zdir="z",
                )

        start_tube = end_tube

    plt.show()


def plot_from_file(json_file_path):
    with open(json_file_path, "r") as file:
        curve_data = json.load(file)

    curve = CRDiscreteCurve.from_json(curve_data)
    draw_ctcr(curve)