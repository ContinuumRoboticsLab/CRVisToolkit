"""
script for visualization of a large number of tests in a single plot
"""

from tests.generation.test_case import import_tests
from plotter.backbone import draw_backbone

MAX_PLOTTED_CURVES = 100
PTS_PER_SEG = 15

JSON_FILENAME = "./tests/export/tests_2seg_extensible.json"


def import_json_test_robots(filepath: str):
    """
    reads the json test file, and returns the constant curvature robots that are
    defined in the file tests.

    Returns two lists, one for the starting robots and one for the target robots
    """

    test_cases = import_tests(filepath)
    starters = [case.starting_robot for case in test_cases]
    targets = [case.target_robot for case in test_cases]

    return starters, targets


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    starters, targets = import_json_test_robots(JSON_FILENAME)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for start, target in zip(
        starters[:MAX_PLOTTED_CURVES], targets[:MAX_PLOTTED_CURVES]
    ):
        start_plot = start.as_cr_backbone(pts_per_seg=PTS_PER_SEG)
        target_plot = target.as_cr_backbone(pts_per_seg=PTS_PER_SEG)

        draw_backbone(start_plot, ax)
        draw_backbone(target_plot, ax)

    plt.show()
