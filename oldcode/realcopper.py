import numpy as np
import pyrealsense2 as rs
import cv2
import open3d as o3d
import copy  
import os
import math
from realsense_depth import DepthCamera
from utils import createPointCloudO3D

# --- Configuration ---
resolution_width, resolution_height = (640, 480)
# แก้ไข Path ให้เหลือชั้นเดียวเพื่อป้องกัน FileNotFoundError
SAVE_DIR = "test5"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)   

selected_pixel = None

def mouse_callback(event, x, y, flags, param):
    global selected_pixel
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_pixel = (x, y)

def rotation_matrix_to_euler_angles(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.rad2deg([x, y, z])

def main():
    global selected_pixel
    Realsensed435Cam = DepthCamera(resolution_width, resolution_height)
    print("Camera running... Press 'q' then click on the copper pipe.")

    while True:
        ret , depth_raw_frame, color_raw_frame = Realsensed435Cam.get_raw_frame()
        if not ret: continue
        
        color_frame = np.asanyarray(color_raw_frame.get_data())
        cv2.imshow("Frame", color_frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            cv2.setMouseCallback("Frame", mouse_callback)
            while True:
                display_frame = color_frame.copy()
                if selected_pixel is not None:
                    cv2.circle(display_frame, selected_pixel, 5, (0, 255, 0), -1)
                cv2.imshow("Frame", display_frame)
                k = cv2.waitKey(1) & 0xFF
                if k == 13 and selected_pixel is not None: break

            # 1. Create PCD and Estimate Normals
            pcd = createPointCloudO3D(color_raw_frame, depth_raw_frame)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
            
            # 2. Snap to Surface
            u, v = selected_pixel
            intrinsics = depth_raw_frame.profile.as_video_stream_profile().intrinsics
            obj_points = np.asarray(pcd.points)
            
            raw_x, raw_y, raw_z = obj_points[:, 0], -obj_points[:, 1], -obj_points[:, 2]
            u_proj = (raw_x * intrinsics.fx / raw_z) + intrinsics.ppx
            v_proj = (raw_y * intrinsics.fy / raw_z) + intrinsics.ppy
            
            dist_sq = (u_proj - u)**2 + (v_proj - v)**2
            best_idx = np.argmin(dist_sq)
            target_pos = obj_points[best_idx]
            
            # 3. Calculate Orientation
            normal = np.asarray(pcd.normals)[best_idx]
            z_axis = normal / np.linalg.norm(normal)
            x_axis = np.array([1, 0, 0]) if abs(z_axis[0]) < 0.9 else np.array([0, 1, 0])
            y_axis = np.cross(z_axis, x_axis)
            y_axis /= np.linalg.norm(y_axis)
            x_axis = np.cross(y_axis, z_axis)
            
            rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
            roll, pitch, yaw = rotation_matrix_to_euler_angles(rotation_matrix)

            # --- 4. PRINT RESULT ---
            print("\n" + "="*50)
            print("GREEN POINT 6-DOF DATA:")
            print(f"Position (X, Y, Z): {target_pos}")
            print(f"Orientation (R, P, Y): {roll:.2f}, {pitch:.2f}, {yaw:.2f}")
            print("="*50)

            # --- 5. SAVE FILES (.ply and .txt) ---
            # สร้างตัวมาร์คจุดเขียว
            point_marker_pcd = o3d.geometry.PointCloud()
            point_marker_pcd.points = o3d.utility.Vector3dVector([target_pos])
            point_marker_pcd.colors = o3d.utility.Vector3dVector([[0, 1, 0]])

            # กำหนด Path ให้ถูกต้อง (ใช้ชื่อไฟล์ตรงๆ ใน SAVE_DIR)
            pcd_path = os.path.join(SAVE_DIR, "target_object.ply")
            marker_path = os.path.join(SAVE_DIR, "green_point_marker.ply")
            txt_path = os.path.join(SAVE_DIR, "target_data.txt")

            try:
                o3d.io.write_point_cloud(pcd_path, pcd)
                o3d.io.write_point_cloud(marker_path, point_marker_pcd)
                with open(txt_path, "w") as f:
                    f.write(f"Position_X: {target_pos[0]:.6f}\n")
                    f.write(f"Position_Y: {target_pos[1]:.6f}\n")
                    f.write(f"Position_Z: {target_pos[2]:.6f}\n")
                    f.write(f"Roll: {roll:.2f}\n")
                    f.write(f"Pitch: {pitch:.2f}\n")
                    f.write(f"Yaw: {yaw:.2f}\n")
                print(f"[SUCCESS] Files saved to: {os.path.abspath(SAVE_DIR)}")
            except Exception as e:
                print(f"[ERROR] Could not save files: {e}")

            # 6. Visualization
            target_ball = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
            target_ball.paint_uniform_color([0, 1, 0])
            target_ball.translate(target_pos)
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02, origin=target_pos)
            axis.rotate(rotation_matrix, center=target_pos)

            mat_unlit = o3d.visualization.rendering.MaterialRecord()
            mat_unlit.shader = "defaultUnlit"
            
            o3d.visualization.draw([{"name": "pcd", "geometry": pcd, "material": mat_unlit},
                                   {"name": "target", "geometry": target_ball, "material": mat_unlit},
                                   {"name": "axis", "geometry": axis, "material": mat_unlit}],
                                   title="6-DOF Target Information")
            break
            
    Realsensed435Cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()