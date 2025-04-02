"""
script for visualization of a large number of tests in a single plot
"""

from tests.generation.test_case import import_tests
from plotter.backbone import draw_backbone

MAX_PLOTTED_CURVES = 300
PTS_PER_SEG = 5
KEEP_LAST_N_POINTS = 15

JSON_FILENAME = "./tests/export/tests_3seg_inext.json"


def import_json_test_target_robots(filepath: str):
    """
    reads the json test file, and returns the constant curvature robots that are
    defined in the file tests.

    Returns two lists, one for the starting robots and one for the target robots
    """

    test_cases = import_tests(filepath)
    targets = [case.target_robot for case in test_cases]

    return targets


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    targets = import_json_test_target_robots(JSON_FILENAME)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for target in targets[:MAX_PLOTTED_CURVES]:
        target_plot = target.as_cr_backbone(pts_per_seg=PTS_PER_SEG)

        draw_backbone(target_plot, ax, start_ind=-KEEP_LAST_N_POINTS)

    ax.set_aspect("equal")
    ax.view_init(elev=45, azim=30)
    plt.draw()
    plt.show()
