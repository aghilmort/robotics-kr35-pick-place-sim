"""Small pose/quaternion helpers shared by scene_builder, ik, and trajectory.

Quaternions are always (w, x, y, z), matching MuJoCo's convention.
Scenario YAML rpy_deg fields use the same fixed-axis (extrinsic) X-Y-Z
convention as the robot's own URDF-derived joints, for consistency.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def euler_deg_to_quat(rpy_deg):
    r, p, y = np.radians(rpy_deg)
    rot = Rotation.from_euler("XYZ", [r, p, y])
    x, y_, z, w = rot.as_quat()
    return np.array([w, x, y_, z])


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_normalize(q):
    return q / np.linalg.norm(q)


def quat_rotate_vec(q, v):
    qv = np.array([0.0, *v])
    r = quat_mul(quat_mul(q, qv), quat_conj(q))
    return r[1:]


def quat_from_z_align(target_axis):
    """Minimal rotation quat mapping world/local +Z onto target_axis (unit vector).

    Used to orient the suction cup so its face-normal (+Z in its own local
    frame) points along a task-specified approach_axis. Roll about the
    approach axis is left at the minimal-rotation solution since a suction
    cup is rotationally symmetric about that axis.
    """
    target_axis = np.asarray(target_axis, dtype=float)
    target_axis = target_axis / np.linalg.norm(target_axis)
    rot, _ = Rotation.align_vectors([target_axis], [[0.0, 0.0, 1.0]])
    x, y, z, w = rot.as_quat()
    return np.array([w, x, y, z])


def compose(pos1, quat1, pos2, quat2):
    """Compose two poses: result = pose1 ∘ pose2 (apply pose2 in pose1's frame)."""
    pos = np.asarray(pos1) + quat_rotate_vec(quat1, pos2)
    quat = quat_mul(quat1, quat2)
    return pos, quat_normalize(quat)


def invert(pos, quat):
    qc = quat_conj(quat)
    return quat_rotate_vec(qc, -np.asarray(pos)), quat_normalize(qc)


def relative_pose(pos1, quat1, pos2, quat2):
    """Pose of frame 2 expressed in frame 1 (i.e. frame1^-1 ∘ frame2)."""
    inv_pos, inv_quat = invert(pos1, quat1)
    return compose(inv_pos, inv_quat, pos2, quat2)


def quat_angle_deg(q1, q2):
    """Smallest rotation angle (deg) between two orientations."""
    q1 = quat_normalize(np.asarray(q1))
    q2 = quat_normalize(np.asarray(q2))
    dot = np.clip(abs(np.dot(q1, q2)), -1.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))
