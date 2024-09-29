import numpy as np
from common.utils import robotindependentmapping
from common.types import CRDiscreteCurve
from math import sin as s
from math import cos as c
from math import sqrt


class ContinuousCurvatureSegment:
    """
    a parametric representation of a Continuum robot segment.

    for now, this implementation parametrizes a continuum robot's configuration
    using the curvature kappa, the angle of the plane of curvature phi, and the segment length
    """

    def __init__(
        self,
        kappa: float | None = None,
        phi: float | None = None,
        length: float | None = None,
        len_limits: float | tuple[float, float] | None = None,
        max_curvature: float | None = None,
    ):
        self.max_curvature = max_curvature
        self.len_limits = len_limits

        self.kappa = kappa
        self.phi = phi
        self.length = length

    def is_valid(self):
        """
        checks if the segment is valid - all three of kappa, phi, and theta must be set

        TODO: in the future, multiple combinations of different parameters
        that help specify the robot's configuration will be considered valid
        """

        return (
            self.kappa is not None and self.phi is not None and self.length is not None
        )

    def set_config(
        self,
        kappa: float | None = None,
        phi: float | None = None,
        length: float | None = None,
    ):
        """
        updates/sets the configuration of the segment
        """
        self.kappa = kappa if kappa else self.kappa
        self.phi = phi if phi else self.phi
        self.length = length if length else self.length

    @property
    def sigma(self):
        """
        returns the euclidean distance between the segment endpoints
        """
        r = 1 / self.kappa  # arc radius
        theta = self.length * self.kappa  # arc angle

        sigma = r * sqrt((1 - c(theta) ** 2) + s(theta) ** 2)
        return sigma

    def array_repr(
        self, num_pts: int | None = None, max_len: float | None = None
    ) -> np.ndarray[float]:
        """
        returns a series of poses in a 4x4 matrix format
        """

        assert num_pts is None or num_pts > 0, "num_pts must be a positive integer"
        assert max_len is None or max_len > 0, "max_len must be a positive value"
        assert (
            num_pts is None or max_len is None
        ), "only one of num_pts or max_len can be specified"
        assert num_pts or max_len, "either num_pts or max_len must be specified"

        if max_len:
            assert self.length, "segment length must be specified"
            num_pts = int(self.length / max_len)

        return robotindependentmapping(
            np.array([self.kappa]),
            np.array([self.phi]),
            np.array([self.length]),
            np.array([num_pts]),
        )


class ContinuousCurvatureCR:
    """
    a piecewise continuous curvature representation of a continuum robot

    defined by a series of independent segments
    """

    def __init__(self, segments: list[ContinuousCurvatureSegment]):
        self.segments = segments

    def is_valid(self):
        """
        checks if the entire curve is valid - all segments must be valid
        """
        if not all([seg.is_valid() for seg in self.segments]):
            raise ValueError("All segments must be valid")

    def as_discrete_curve(
        self, pts_per_seg: int | None = None, max_len: float | None = None
    ) -> np.ndarray[float]:
        """
        exports the entire backbone curve as a CRDiscreteCurve object,
        so that it can be plotted
        """

        assert (
            pts_per_seg is None or pts_per_seg > 0
        ), "pts_per_seg must be a positive integer"
        assert max_len is None or max_len > 0, "max_len must be a positive value"
        assert (
            pts_per_seg is None or max_len is None
        ), "only one of pts_per_seg or max_len can be specified"
        assert pts_per_seg or max_len, "either pts_per_seg or max_len must be specified"

        self.is_valid()

        pose_n = np.eye(4)
        coords = []

        for seg in self.segments:
            new_coords = seg.array_repr(pts_per_seg, max_len)
            for i in range(len(new_coords)):
                transformed = pose_n.dot(new_coords[i])

                # change back into columnwise
                coords.append(transformed.T.reshape(1, 16))

            pose_n = transformed
            print("pose_n:\n", pose_n)
            breakpoint()

        seg_end = np.cumsum([seg.shape[0] for seg in coords])
        coords = np.vstack(coords)

        return CRDiscreteCurve(coords, seg_end)
