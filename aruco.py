import cv2
import numpy as np
from communication.realsense import DepthCamera

MARKER_LENGTH = 0.055  # meters (edge length of the printed marker)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Marker corner points in the marker's own frame (origin at center, Z up out of marker)
half = MARKER_LENGTH / 2.0
obj_points = np.array([
    [-half,  half, 0.0],
    [ half,  half, 0.0],
    [ half, -half, 0.0],
    [-half, -half, 0.0],
], dtype=np.float32)

camera = DepthCamera(640, 480)
camera_matrix, dist_coeffs = camera.get_color_intrinsics()

try:
    while True:
        ok, _, color_image = camera.get_frame()
        if not ok:
            continue

        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(color_image, corners, ids)

            for marker_corners, marker_id in zip(corners, ids.flatten()):
                ok_pnp, rvec, tvec = cv2.solvePnP(
                    obj_points, marker_corners[0],
                    camera_matrix, dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE,
                )
                if not ok_pnp:
                    continue

                cv2.drawFrameAxes(color_image, camera_matrix, dist_coeffs,
                                  rvec, tvec, MARKER_LENGTH * 0.75, 2)

                x, y, z = tvec.flatten()
                cv2.putText(color_image,
                            f"ID {marker_id}: x={x:+.3f} y={y:+.3f} z={z:+.3f} m",
                            (10, 30 + 20 * int(marker_id % 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow('ArUco Reference Frame (RealSense)', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    camera.release()
    cv2.destroyAllWindows()
