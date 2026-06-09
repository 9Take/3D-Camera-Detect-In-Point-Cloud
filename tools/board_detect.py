"""Find the ChArUco board in an image and recover its pose.

Needs OpenCV but NO camera/PLC, so it can be tested against a rendered board.
"""
import cv2
import numpy as np


def build_charuco(squares_x, squares_y, square_mm, marker_mm,
                  dict_name="DICT_6X6_250"):
    """Construct the ChArUco board + detector used everywhere. Returns (board, detector)."""
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_mm, marker_mm, dictionary)
    detector = cv2.aruco.CharucoDetector(board)
    return board, detector


def _reprojection_error(obj_pts, img_pts, rvec, tvec, camera_matrix, dist_coeffs):
    """RMS pixel error of obj_pts projected with (rvec, tvec) vs the detected img_pts."""
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist_coeffs)
    return float(np.sqrt(np.mean((proj.reshape(-1, 2) - img_pts.reshape(-1, 2)) ** 2)))


def detect_board_pose(frame, charuco_detector, board, camera_matrix, dist_coeffs,
                      min_corners=8, z_min_mm=50.0, z_max_mm=3000.0, max_reproj_px=2.0):
    """Detect the ChArUco board and solve its pose in the camera frame.

    Uses the planar IPPE solver, which returns BOTH mirror solutions for a flat
    board; we keep the one with the lowest reprojection error. That defeats the
    ~180-deg pose flip that otherwise corrupts hand-eye calibration. Frames whose
    best fit is still worse than max_reproj_px are rejected.

    Returns (success, R_target2cam, t_target2cam, debug_frame, n_corners, reproj_px).
    """
    debug = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ch_corners, ch_ids, _, _ = charuco_detector.detectBoard(gray)

    n = 0 if ch_ids is None else len(ch_ids)
    if n < min_corners:
        cv2.putText(debug, f"ChArUco corners: {n} (need {min_corners})", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return False, None, None, debug, n, None

    cv2.aruco.drawDetectedCornersCharuco(debug, ch_corners, ch_ids)
    # NOTE: board.matchImagePoints() is broken on OpenCV 4.7.0 — it returns the
    # MARKER corners (e.g. 68 pts for 24 charuco corners) instead of matching the
    # chessboard corners, scrambling the obj<->img pairing and blowing up the
    # pose (reproj ~1400px). Build the correspondences directly instead: each
    # detected charuco corner id maps to its 3D position in getChessboardCorners().
    cb = board.getChessboardCorners()
    ids = ch_ids.flatten()
    obj_pts = cb[ids].astype(np.float32).reshape(-1, 1, 3)
    img_pts = ch_corners.reshape(-1, 1, 2).astype(np.float32)
    if len(obj_pts) < 4:
        cv2.putText(debug, f"matched pts: {len(obj_pts)} (need 4)",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return False, None, None, debug, n, None

    ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj_pts, img_pts, camera_matrix, dist_coeffs,
                                              flags=cv2.SOLVEPNP_IPPE)
    if not ok or len(rvecs) == 0:
        return False, None, None, debug, n, None

    errs = [_reprojection_error(obj_pts, img_pts, rv, tv, camera_matrix, dist_coeffs)
            for rv, tv in zip(rvecs, tvecs)]
    best = int(np.argmin(errs))
    rvec, tvec, reproj = rvecs[best], tvecs[best], errs[best]

    # Reject ambiguous/bad fits: a flip or mis-detection blows up the reprojection.
    if reproj > max_reproj_px:
        cv2.putText(debug, f"REJECTED: reproj {reproj:.2f}px (max {max_reproj_px})", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return False, None, None, debug, n, reproj

    z_distance = tvec[2][0]
    # Reject the board if it isn't in front of the camera, or the depth is
    # outside the real working range of the rig (units mm).
    if z_distance <= z_min_mm or z_distance > z_max_mm:
        cv2.putText(debug, f"REJECTED: Invalid Cam Z ({z_distance:.1f} mm)", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return False, None, None, debug, n, reproj

    cv2.drawFrameAxes(debug, camera_matrix, dist_coeffs, rvec, tvec, board.getSquareLength() * 2, 2)
    cv2.putText(debug, f"X:{tvec[0][0]:.1f} Y:{tvec[1][0]:.1f} Z:{tvec[2][0]:.1f} mm  reproj {reproj:.2f}px",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    R_target2cam, _ = cv2.Rodrigues(rvec)
    return True, R_target2cam, tvec, debug, n, reproj
