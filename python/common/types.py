from dataclasses import dataclass
import numpy as np


@dataclass
class PlotterSettings:
    """
    the base class including plotter settings common to both kinds of CRs

    Parameters
    ----------
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
    """
    tipframe: bool = True
    segframe: bool = False
    baseframe: bool = False
    projections: bool = False
    baseplate: bool = True

@dataclass
class TDCRPlotterSettings(PlotterSettings):
    """
    The paramters required to specify how a TDCR should be plotted

    Parameters
    ----------
    r_disk: double
        Radius of spacer disks
    r_height: double
        height of spacer disks
    """
    r_disk: float = 2.5 * 1e-3
    r_height: float = 1.5 * 1e-3


class CTCRPlotterSettings(PlotterSettings):
    """
    The paramters required to specify how a CTCR should be plotted

    Parameters
    ----------
    r_tube: ndarray
        Radii of tubes
    """
    r_tube: np.ndarray[float] = np.array([2.5, 2.0, 1.5]) * 1e-3


class CRDiscreteCurve:
    """
    a discrete point representation of a CR curve with multiple segments

    this class can be used to described both TDCRs and CTCRs - 
    for TDCRs, the seg_end indicates where the segments terminate,
    and similarly, indicate where the tubes terminate for CTCRs

    Parameters
    ----------
    g: ndarray | list[ndarray]
        Backbone curve with n 4x4 transformation matrices reshaped into 1x16 vector (columnwise)
    seg_end: ndarray | list
        Indices of g where tdcr segments terminate
    """
    def __init__(self, g: np.ndarray[float], seg_end: np.ndarray[int]):
        # Argument validation
        if g.shape[0] < len(seg_end) or max(seg_end) > g.shape[0]:
            raise ValueError("Dimension mismatch")
        
        if isinstance(g, list):
            g = np.array(g)
        
        # try reshaping all elements of g to (1, 16)
        for i in range(len(g)):
            if g[i].shape != (16,):
                g[i] = g[i].reshape(16)

        self.g = g
        self.seg_end = seg_end
    
    @classmethod
    def from_json(cls, data: dict):
        g = np.array(data["g"])
        seg_end = np.array(data["seg_end"])
        return cls(g=g, seg_end=seg_end)