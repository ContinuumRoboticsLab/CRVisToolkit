import numpy as np
import matplotlib.pyplot as plt

from common.types import CrBackbone


def draw_backbone(
    backbone: CrBackbone,
    axes,
    start_ind: int = None,
    end_ind: int = None,
    color=None,
    label=None,
):
    points = backbone.points

    if start_ind is not None:
        points = points[start_ind:]
    if end_ind is not None:
        points = points[:end_ind]

    x_vals = [point[0] for point in points]
    y_vals = [point[1] for point in points]
    z_vals = [point[2] for point in points]

    axes.plot(x_vals, y_vals, z_vals, color=color, label=label)


if __name__ == "__main__":
    """
    sample usage - get a robot, call as_cr_backbone, and draw it
    """
    from common.robot import ConstantCurvatureCR, ConstantCurvatureSegment

    robot = ConstantCurvatureCR(
        [
            ConstantCurvatureSegment(1 / 0.09, np.pi / 4, 0.05),
            ConstantCurvatureSegment(1 / 0.01, 0, 0.03),
        ]
    )

    backbone = robot.as_cr_backbone(10)

    draw_backbone(backbone)
    plt.show()
