import cv2
import numpy as np


class ArucoReference:
    """Detects an ArUco marker and transforms points from the camera frame
    into the marker's local frame (X-right, Y-up, Z-out-of-marker)."""

    def __init__(self, camera_matrix, dist_coeffs, marker_length,
                 dictionary_name="DICT_6X6_250"):
        self.camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
        self.dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64)
        self.marker_length = float(marker_length)

        aruco_dict = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, dictionary_name)
        )
        self.detector = cv2.aruco.ArucoDetector(
            aruco_dict, cv2.aruco.DetectorParameters()
        )

        half = self.marker_length / 2.0
        # Marker frame: X right, Y up, Z out of marker
        self.obj_points = np.array([
            [-half,  half, 0.0],
            [ half,  half, 0.0],
            [ half, -half, 0.0],
            [-half, -half, 0.0],
        ], dtype=np.float32)

    def detect_pose(self, color_frame):
        """Returns (rvec, tvec, marker_id) for the first detected marker, or None."""
        gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            return None

        ok, rvec, tvec = cv2.solvePnP(
            self.obj_points, corners[0][0],
            self.camera_matrix, self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok:
            return None
        return rvec, tvec, int(ids[0][0])

    def draw_overlay(self, color_frame):
        """Detect markers on color_frame and draw outline + 3D axes in-place.
        Returns the pose tuple of the first marker, or None."""
        gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            return None

        cv2.aruco.drawDetectedMarkers(color_frame, corners, ids)

        first_pose = None
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            ok, rvec, tvec = cv2.solvePnP(
                self.obj_points, marker_corners[0],
                self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if not ok:
                continue
            cv2.drawFrameAxes(color_frame, self.camera_matrix, self.dist_coeffs,
                              rvec, tvec, self.marker_length * 0.75, 2)
            if first_pose is None:
                first_pose = (rvec, tvec, int(marker_id))
        return first_pose

    def transform_from_transformer_frame(self, p_transformer, rvec, tvec):
        """Convert a point stored in transformer.py's frame [x, -y_cv, -z_cv]
        into the marker's local frame."""
        # transformer.py stores [x_cv, -y_cv, -z_cv]; recover OpenCV-frame point.
        p_cv = np.array([p_transformer[0], -p_transformer[1], -p_transformer[2]],
                        dtype=np.float64)
        R, _ = cv2.Rodrigues(rvec)
        # p_cv = R * p_marker + t  =>  p_marker = R^T (p_cv - t)
        p_marker = R.T @ (p_cv - tvec.flatten())
        return p_marker
