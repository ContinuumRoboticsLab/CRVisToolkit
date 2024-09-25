import json

from common.types import CRDiscreteCurve
from plotter.tdcr import draw_tdcr


def plot_from_file(json_file_path):
    with open(json_file_path, "r") as f:
        curve_data = json.load(f)

    curve = CRDiscreteCurve.from_json(curve_data)

    draw_tdcr(curve)
