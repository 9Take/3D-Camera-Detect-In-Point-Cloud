"""Pure geometry / matrix math for hand-eye calibration.

No hardware, no OpenCV — just numpy. Safe to import and unit-test anywhere.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def rotation_matrix_from_abc(A, B, C):
    """KUKA Euler angles (deg) -> rotation matrix.

    KUKA ABC is intrinsic Z-Y-X (A about Z, B about Y, C about X), i.e.
    R = Rz(A) @ Ry(B) @ Rx(C).  scipy's intrinsic 'ZYX' matches this exactly.
    """
    return Rotation.from_euler("ZYX", [A, B, C], degrees=True).as_matrix()


def marker_positions_in_base(R_gripper2base, t_gripper2base, t_target2cam,
                             R_cam2gripper, t_cam2gripper):
    """Map the marker origin into the robot base frame for every recorded pose.

    The marker is fixed in the world, so a good calibration maps it to (nearly)
    the same base-frame point every time. Returns an (N, 3) array of points.
    """
    pts_base = []
    for Rg, tg, t_t2c in zip(R_gripper2base, t_gripper2base, t_target2cam):
        p_grip = R_cam2gripper @ t_t2c + t_cam2gripper   # marker origin in gripper frame
        p_base = Rg @ p_grip + tg                         # marker origin in base frame
        pts_base.append(p_base.ravel())
    return np.array(pts_base)


def residual_stats(pts_base):
    """Spread of base-frame marker points = hand-eye solve error (lower is better).

    Returns (mean_point[3], rms_mm, max_mm).
    """
    mean_pt = pts_base.mean(axis=0)
    resid = np.linalg.norm(pts_base - mean_pt, axis=1)
    return mean_pt, float(np.sqrt((resid ** 2).mean())), float(resid.max())
