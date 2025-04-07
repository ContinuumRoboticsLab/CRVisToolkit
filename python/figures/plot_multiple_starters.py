from tests.generation.test_case import import_tests, STARTING_POSITION_VARS
from tests.generation.perturbation import PERTURBATION_VALUES
from common.robot import ConstantCurvatureCR
from plotter.backbone import draw_backbone

import matplotlib.pyplot as plt

FILEPATH = "tests/export/tests_3seg_inext.json"

test_case = import_tests(FILEPATH)[1]
starters = STARTING_POSITION_VARS[1:]

# get the target and five starters
target: ConstantCurvatureCR = test_case.target_robot
starters: list[ConstantCurvatureCR] = [
    getattr(test_case, starter) for starter in starters if getattr(test_case, starter)
]

target_plot = target.as_cr_backbone(20)
starter_plots = [starter.as_cr_backbone(20) for starter in starters]

colors = ["#FF0000", "#BB0044", "#880088", "#4400BB", "#0000FF"]

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

draw_backbone(target_plot, ax, color="black", label="Target Robot")
for i, starter_plot in enumerate(starter_plots):
    draw_backbone(
        starter_plot,
        ax,
        color=colors[i],
        label=f" Preturbed at {PERTURBATION_VALUES[i] * 100:.0f}%",
    )


plt.legend(loc="lower right", framealpha=1, fancybox=True)
plt.title("Perturbed Starter Robots for 3-segment Inextensible CR")
plt.show()
