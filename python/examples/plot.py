import numpy as np
from common.robot import ConstantCurvatureCR, ConstantCurvatureSegment
from plotter.tdcr import draw_tdcr
from plotter.ctcr import draw_ctcr


def plott():
    seg1 = ConstantCurvatureSegment(1 / 30e-3, np.deg2rad(0), 50e-3)
    seg2 = ConstantCurvatureSegment(1 / 40e-3, np.deg2rad(160), 70e-3)
    seg3 = ConstantCurvatureSegment(1 / 15e-3, np.deg2rad(30), 25e-3)

    cr = ConstantCurvatureCR([seg1, seg2, seg3])
    draw_tdcr(cr.as_discrete_curve(pts_per_seg=10))


def plotc():
    seg1 = ConstantCurvatureSegment(1 / 30e-3, np.deg2rad(0), 50e-3)
    seg2 = ConstantCurvatureSegment(1 / 40e-3, np.deg2rad(160), 70e-3)
    seg3 = ConstantCurvatureSegment(1 / 15e-3, np.deg2rad(30), 25e-3)

    cr = ConstantCurvatureCR([seg1, seg2, seg3])
    draw_ctcr(cr.as_discrete_curve(pts_per_seg=10))


def main(task: str):
    match task:
        case "plotc":
            plotc()
        case "plott":
            plott()
        case _:
            print("Invalid task. Please specify a valid task.")
            exit(1)
