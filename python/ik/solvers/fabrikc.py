from common.robot import ConstantCurvatureCR, ConstantCurvatureSegment

from ik.solvers.base_solver import CcIkSettings, CcIkSolver, IkResult
from ik.solvers.neppalli import NeppalliIkSolver, NeppalliIkTarget, NeppalliIkSettings
from ik.target import IkTargetType, P3Direction

from dataclasses import dataclass
from typing import Optional
import numpy as np
import time

from structlog import get_logger

logger = get_logger()


def _get_joint_angle(zb, ze):
    dp = zb @ ze

    if abs(dp) > 1:
        diff = abs(dp) - 1
        if diff > 1e-6:
            logger.error("Joint angle is not valid")
            raise ValueError("Joint angle is not valid")
        else:
            dp = np.sign(dp)

    return np.arccos(dp)


def _get_link_length(seg_length, joint_angle):
    if abs(joint_angle) < 1e-6:
        return seg_length / 2
    return (seg_length / joint_angle) * np.tan(joint_angle / 2)


def _get_phi(joint_disp):
    """
    computes the bending plane angle (phi) from the joint displacement (from base frame)
    """
    return -np.arctan2(-joint_disp[1], joint_disp[0])


@dataclass
class ConstantCurvatureJoint:
    """
    represents constant curvature segment as a spherical joint with two links of equal
    length on either side.

    NOTE: all internal vector values are representend in the world coordiante frame.
    Utilities are provided for getting segment rotations/displacements from the base
    frame to the end frame.

    pb: position of the base of the joint
    pj: position of the joint (point of bending)
    pe: position of the end of the joint

    zb: direction of the base link
    ze: direction of the end link

    link_length: length of the link
    joint_angle: angle between the two links

    segment_length: length of the constant curvature segment
    """

    pb: np.ndarray
    pj: Optional[np.ndarray]
    pe: np.ndarray

    zb: np.ndarray
    ze: np.ndarray

    link_length: Optional[float]
    joint_angle: Optional[float]

    segment_length: float

    @classmethod
    def from_cc_segment(
        cls, segment: ConstantCurvatureSegment, pre_transform=np.eye(4)
    ):
        """
        constructs the joint representation from a constant curvature segment and the
        SE(3) transformation of the segments precending it, to ensure that all values
        are represented in the world coordinate frame
        """
        pre_rotation = pre_transform[:3, :3]

        seg_transformation = segment.t_matrix().A
        seg_rotation = seg_transformation[:3, :3]
        seg_endpoint = seg_transformation @ np.array([0, 0, 0, 1])

        pb = pre_transform @ np.array([0, 0, 0, 1])
        pe = pre_transform @ seg_endpoint

        zb = pre_rotation @ np.array([0, 0, 1])
        ze = seg_rotation[:, 2]
        ze = pre_rotation @ ze
        joint_angle = _get_joint_angle(zb, ze)
        link_length = _get_link_length(segment.length, joint_angle)

        pj = pb[:3] + zb * link_length

        return cls(
            pb=pb[:3],
            pj=pj,
            pe=pe[:3],
            zb=zb,
            ze=ze,
            segment_length=segment.length,
            link_length=link_length,
            joint_angle=joint_angle,
        )

    def as_cc_segment(self, base_rotation, base_displacement):
        """
        converts the joint representation back into a constant curvature segment.
        """
        theta = self.joint_angle

        joint_displacement = self._get_joint_displacement(
            base_rotation, base_displacement
        )

        phi = _get_phi(joint_displacement)

        kappa = theta / self.segment_length

        return ConstantCurvatureSegment(
            length=self.segment_length,
            kappa=kappa,
            phi=phi,
            is_extensible=False,
        )

    def _reeval_pb(self):
        """
        called from the backward reaching phase - re-evaluates the base position using
        pj, zb, and link length. Used to determine the new base position after the joint
        has been updated, which becomes the immediate proximal segment's end position.
        """

        self.pb = self.pj - self.link_length * self.zb

    def _reeval_pe(self):
        """
        called from the forward reaching phase - re-evaluates the end position using
        pj, zb, and link length. Used to determine the new end position after the joint
        has been updated, which becomes the immediate distal segment's base position.
        """

        self.pe = self.pj + self.link_length * self.ze

    def reevaluate_after_change(self, forward_reaching=True):
        """
        re-evaluate theta, pj, and link length after changes zb to other parameters

        if the forward_reaching parameter is True, then this function will determine
        pj by applying the negative translation from the end of the link (pe). If the
        value is false, this function is being called form the backward reaching
        algorithm, and will determine pj by applying the positive translation from the
        base of the link (pb).
        """

        self.joint_angle = _get_joint_angle(self.zb, self.ze)
        self.link_length = _get_link_length(self.segment_length, self.joint_angle)
        if forward_reaching:
            # using (5) from the FABRIKc paper
            self.pj = self.pe - self.link_length * self.ze
            self._reeval_pb()
        else:
            # using (4) from the FABRIKc paper
            self.pj = self.pb + self.link_length * self.zb
            self._reeval_pe()

    def _get_joint_rotation(self, base_rotation, base_displacement):
        """
        computes the joint bending plane angle (phi) and uses the current bending plane
        angle to return the rotation matrix that rotates from the base frame to the end
        """
        joint_displacement = self._get_joint_displacement(
            base_rotation, base_displacement
        )
        phi = _get_phi(joint_displacement)

        R_t1 = np.array(
            [[0, np.cos(phi), np.sin(phi)], [0, -np.sin(phi), np.cos(phi)], [1, 0, 0]]
        )

        R_t2 = np.array(
            [
                [np.cos(self.joint_angle), -np.sin(self.joint_angle), 0],
                [np.sin(self.joint_angle), np.cos(self.joint_angle), 0],
                [0, 0, 1],
            ]
        )

        return R_t1 @ R_t2 @ R_t1.T

    def _get_joint_displacement(self, base_rotation, base_displacement):
        """
        returns the displacement of the joint from the base frame to the end frame,
        using the provided rotoation and displacement to get from the world frame to the
        joint base frame.
        """
        return base_rotation.T @ (self.pe - base_displacement)


@dataclass
class FabrikcIkSettings(CcIkSettings):
    position_tolerance: float = 1e-7
    max_iter = 200


class FabrikcIkSolver(CcIkSolver):
    target_type = IkTargetType.POSITION_POINTING
    requires_init_guess = True
    settings_class = FabrikcIkSettings

    def __init__(
        self,
        robot: ConstantCurvatureCR,
        settings: CcIkSettings,
        ik_target_pose: P3Direction,
        **kwargs,
    ):
        super().__init__(robot, settings, ik_target_pose, **kwargs)

        self.segment_joints: list[ConstantCurvatureJoint] = []
        pre_transform = np.eye(4)

        for seg in robot.segments:
            self.segment_joints.append(
                ConstantCurvatureJoint.from_cc_segment(seg, pre_transform)
            )
            pre_transform = pre_transform @ seg.t_matrix().A

        self.p_star = ik_target_pose.position
        self.z_hat_star = ik_target_pose.pointing_direction
        self.exec_time = None
        self.iter_count = 0

    def __perform_forward_reaching(self):
        # set the starting conditions

        distal_joint_zb = self.z_hat_star
        distal_joint_pb = self.p_star

        # iterating over all segments from distal to proximal
        for i, segment in enumerate(reversed(self.segment_joints)):
            # calculate the new joint position
            segment.pe = distal_joint_pb
            segment.ze = distal_joint_zb
            segment.pj = segment.pe - segment.link_length * segment.ze  # (5)

            # most proximal segment
            if i == len(self.segment_joints) - 1:
                zb = np.array([0, 0, 1])
            else:
                # next segment in reversed ordering is proximal
                proximal_joint_pj = self.segment_joints[-(i + 2)].pj  # p_(t-1)j
                link_direction = segment.pj - proximal_joint_pj
                zb = link_direction / np.linalg.norm(link_direction)  # (9)

            segment.zb = zb

            # reevaluate the joint parameters after pj, pe, ze, zb have been set
            segment.reevaluate_after_change(forward_reaching=True)

            # update for the next joint
            distal_joint_zb = segment.zb
            distal_joint_pb = segment.pb

    def __perform_backward_reaching(self):
        proximal_joint_pe = np.array([0, 0, 0])
        # the base segments zb should always be [0, 0, 1] after forward reaching
        assert all(self.segment_joints[0].zb == np.array([0, 0, 1]))
        proximal_joint_ze = np.array([0, 0, 1])

        # iterating over all segments from proximal to distal
        for i, joint in enumerate(self.segment_joints):
            joint.pb = proximal_joint_pe
            joint.zb = proximal_joint_ze

            joint.pj = joint.pb + joint.link_length * joint.zb

            if i == len(self.segment_joints) - 1:
                # the part of FABRIKc that guarantees ee orientation
                joint.ze = self.z_hat_star
            else:
                next_joint = self.segment_joints[i + 1]
                joint_direction = next_joint.pj - joint.pj
                joint.ze = joint_direction / np.linalg.norm(joint_direction)

            joint.reevaluate_after_change(forward_reaching=False)

            proximal_joint_ze = joint.ze
            proximal_joint_pe = joint.pe

    def _get_cc_solution(self):
        """
        convert our current segment joints state into the constant curvature segment
        representation
        """

        prev_rotation = np.eye(3)
        prev_displacement = np.array([0, 0, 0])

        cc_segments = []

        for joint in self.segment_joints:
            cc_segments.append(joint.as_cc_segment(prev_rotation, prev_displacement))
            prev_rotation = prev_rotation @ joint._get_joint_rotation(
                prev_rotation, prev_displacement
            )
            prev_displacement = joint.pe

        return ConstantCurvatureCR(cc_segments)

    def _get_cc_neppalli(self):
        """
        use neppalli solver to map from endpoints to constant curvature segments
        """
        endpoints = [joint.pe for joint in self.segment_joints]
        neppalli_solver = NeppalliIkSolver(
            self.cr,
            NeppalliIkSettings(),
            NeppalliIkTarget(endpoints),
        )
        neppalli_solver.solve()
        return neppalli_solver.cr

    def _get_error(self):
        p_ne = self.segment_joints[-1].pe
        error = np.linalg.norm(p_ne - self.p_star)
        return error

    def __check_nan(self):
        for joint in self.segment_joints:
            if any(np.isnan([joint.joint_angle, joint.link_length])):
                logger.error("NaN detected in joint parameters")

    def solve(self):
        start_time = time.time()
        while (
            self._get_error() > self.settings.position_tolerance
            and self.iter_count < self.settings.max_iter
        ):
            self.__perform_forward_reaching()
            self.__perform_backward_reaching()
            self.iter_count += 1

        self.exec_time = time.time() - start_time

        # use joint representation to get arc parameters

        robot = self._get_cc_neppalli()

        self.cr = robot

        print(self._get_cc_neppalli().t_matrix())
        print(self.segment_joints[-1].ze)
        print(self.z_hat_star)

        res = (
            IkResult.SUCCESS
            if self.iter_count < self.settings.max_iter
            else IkResult.MAX_ITER
        )
        return res

    def get_errors(self):
        """
        fabrikc does not use the typical SE(3) error class. Orientation is specified
        using a unit vector that indicated the desired pointing direction of the end
        effector and is guaranteed by the algorithm. Thus, only positional error is
        considered.
        """

        ee_pose = self.cr.t_matrix().A
        ee_position = ee_pose[:3, 3]
        ee_orientation = ee_pose[:3, :3] @ np.array([0, 0, 1])

        target_position = self.p_star
        pos_error = np.linalg.norm(ee_position - target_position)

        target_orientation = self.z_hat_star
        orientation_error = np.linalg.norm(ee_orientation - target_orientation)

        return pos_error, orientation_error


if __name__ == "__main__":
    from math import pi
    from plotter.tdcr import draw_tdcr, TDCRPlotterSettings
    from matplotlib import pyplot as plt

    seg1 = ConstantCurvatureSegment(1 / 0.09, pi / 4, 0.05)
    seg2 = ConstantCurvatureSegment(1 / 0.07, pi / 10, 0.03)
    seg3 = ConstantCurvatureSegment(1 / 0.09, pi / 3, 0.05)
    robot = ConstantCurvatureCR([seg1, seg2, seg3])
    starter_plot = robot.as_discrete_curve(pts_per_seg=5)

    target_robot = ConstantCurvatureCR(
        [
            ConstantCurvatureSegment(1 / 0.08, pi / 3, 0.05),
            ConstantCurvatureSegment(1 / 0.05, pi / 11, 0.03),
            ConstantCurvatureSegment(1 / 0.076, pi / 4, 0.05),
        ]
    )

    target_pose = P3Direction.from_target_robot(target_robot)
    solver = FabrikcIkSolver(
        robot,
        FabrikcIkSettings(),
        target_pose,
    )
    draw_tdcr(starter_plot, plotter_settings=TDCRPlotterSettings(plot_title="Starter"))

    solver.solve()
    print(f"errors are {solver.get_errors()} after {solver.iter_count} iterations")
    draw_tdcr(
        solver.cr.as_discrete_curve(pts_per_seg=5),
        plotter_settings=TDCRPlotterSettings(plot_title="Solution"),
    )
    plt.show()
