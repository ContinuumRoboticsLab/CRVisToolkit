import numpy as np
from numpy.typing import ArrayLike
from math import sin as s
from math import cos as c
from math import sqrt
from spatialmath import SE3

from common.utils import robotindependentmapping, se3_to_pose
from common.types import CRDiscreteCurve
from common.coordinates import CrConfigurationType
from ik.target import IkTargetType


class ConstantCurvatureSegment:
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
        is_extensible: bool = False,
        len_limits: tuple[float, float] | None = None,
        max_curvature: float | None = None,
        repr_type: CrConfigurationType = CrConfigurationType.KPL,
    ):
        # representation metadata
        self.repr_type = repr_type

        # robot actuation limits
        self.max_curvature = max_curvature
        self.len_limits = len_limits
        self.is_extensible = is_extensible

        # robot actuation state
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

    @property
    def n(self):
        """
        returns the degrees of freedom in the configuration space
        """
        if self.is_extensible:
            return 3
        else:
            return 2

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

    def state_vector(
        self, repr_type: CrConfigurationType | None = None
    ) -> np.ndarray[float]:
        if repr_type is None:
            repr_type = self.repr_type

        if repr_type != CrConfigurationType.KPL:
            # TODO: implement more representations using coordinates.py
            raise NotImplementedError("Only KPL representation is supported for now")

        # in an inextensible segment, the length is not considered a degree of freedom
        if self.is_extensible:
            return np.array([self.kappa, self.phi, self.length])
        else:
            return np.array([self.kappa, self.phi])

    def t_matrix(self):
        """
        returns the transformation matrix of the segment
        """
        s_p = s(self.phi)
        c_p = c(self.phi)
        s_ks = s(self.kappa * self.length)
        c_ks = c(self.kappa * self.length)

        t_matrix = np.array(
            [
                [c_p * c_p * (c_ks - 1) + 1, s_p * c_p * (c_ks - 1), c_p * s_ks, 0],
                [
                    s_p * c_p * (c_ks - 1),
                    c_p * c_p * (1 - c_ks) + c_ks,
                    s_p * s_ks,
                    0,
                ],
                [-c_p * s_ks, -s_p * s_ks, c_ks, 0],
                [0, 0, 0, 1],
            ]
        )

        if self.kappa != 0:
            t_matrix[:, 3] = [
                (c_p * (1 - c_ks)) / self.kappa,
                (s_p * (1 - c_ks)) / self.kappa,
                s_ks / self.kappa,
                1,
            ]
        else:
            t_matrix[:, 3] = [0, 0, self.length, 1]

        return SE3(t_matrix)


class ConstantCurvatureCR:
    """
    a piecewise continuous curvature representation of a continuum robot

    defined by a series of independent segments
    """

    def __init__(self, segments: list[ConstantCurvatureSegment]):
        self.segments = segments
        self.num_segments = len(segments)

        # n is degrees of freedom in configuration space across all segments
        self.n = sum([seg.n for seg in segments])

        self.repr_type = segments[0].repr_type
        for seg in segments:
            if seg.repr_type != self.repr_type:
                raise ValueError("All segments must have the same representation type")

    def is_valid(self):
        """
        checks if the entire curve is valid - all segments must be valid
        """
        if not all([seg.is_valid() for seg in self.segments]):
            raise ValueError("All segments must be valid")

    def as_discrete_curve(
        self, pts_per_seg: int | None = None, max_len: float | None = None
    ) -> CRDiscreteCurve:
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

            pose_n = np.dot(pose_n, seg.t_matrix())

        seg_end = np.cumsum([seg.shape[0] for seg in coords])
        coords = np.vstack(coords)

        return CRDiscreteCurve(coords, seg_end)

    def state_vector(self) -> np.ndarray[float]:
        """
        returns the configuration space state of the robot as
        as a single (n x 1) column vector

        the state is often referred to in literature as theta
        """

        return np.hstack([seg.state_vector() for seg in self.segments])

    def set_config(self, theta: list[np.ndarray[float]]):
        """
        updates the configuration of the robot to the given state
        """

        if isinstance(theta, list):
            assert len(theta) == self.num_segments, "Invalid number of segments"

        elif isinstance(theta, ArrayLike):
            assert (
                theta.size == self.n
            ), f"Invalid number of degrees of freedom, expected {self.n}, got {theta.size}"
            theta = theta.reshape(self.num_segments, -1)
        else:
            raise ValueError("Invalid theta type")

        # update each segment's configuration
        for i, seg in enumerate(self.segments):
            assert (
                theta[i].size == seg.n
            ), f"Invalid theta shape for segment {i} (got {theta[i].shape}, expected {(seg.n, 1)})"

            as_dict = {
                "kappa": theta[i][0],
                "phi": theta[i][1],
            }

            if seg.is_extensible:
                as_dict["length"] = theta[i][2]
            seg.set_config(**as_dict)

    def get_config(self):
        """
        returns the configuration of the robot as a list of dictionaries
        """
        return [seg.__dict__ for seg in self.segments]

    def t_matrix(self):
        """
        returns the transformation matrix of the entire robot
        """
        t_matrix = SE3(np.eye(4))
        for seg in self.segments:
            t_matrix = t_matrix * seg.t_matrix()
        return SE3(t_matrix)

    def pose_vector(self, theta: np.ndarray[float] | None = None) -> np.ndarray[float]:
        """
        returns the pose vector (m x 1) of the robot end-effector at the given state

        theta is provided as an argument and not taken from the object
        so that the method can be used in the numdifftools Jacobian,
        but can still be determined using the "state_vector" method

        the position is determined from the last column of the transformation matrix,
        and the rotation axis/angle is determined using the matrix trace

        ref: https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation#
        """
        old_theta = self.state_vector()

        if theta is not None:
            self.set_config(theta)

        # get the end-effector transformation matrix
        t_matrix = self.t_matrix()

        self.set_config(old_theta)

        return se3_to_pose(t_matrix.A)

    def pose_for_target(
        self, target_type: IkTargetType, theta: np.ndarray[float] | None = None
    ):
        match target_type:
            case IkTargetType.SE3 | IkTargetType.POSITION_POINTING:
                # return full (6x1)
                return self.pose_vector(theta)
            case IkTargetType.P3:
                # return only the position part of the pose vector
                return self.pose_vector(theta)[:3]
            case IkTargetType.SO3:
                # return only the orientation part of the pose vector
                return self.pose_vector(theta)[3:]
            case _:
                raise ValueError("Invalid target type")
