import numpy as np
import pyrealsense2 as rs
import cv2
import open3d as o3d
import os
import math
import sys
from realsense_depth import DepthCamera
from utils import createPointCloudO3D

# --- Configuration ---
resolution_width, resolution_height = (640, 480)

# โฟลเดอร์สำหรับเซฟผลลัพธ์รอบนี้
SAVE_DIR = "test9_auto_production"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# กำหนด Path ไปหาไฟล์แม่แบบที่เราสร้างไว้จากโค้ดตัวที่แล้ว (test8)
# ** ต้องมั่นใจว่ามีไฟล์นี้อยู่ในโฟลเดอร์นะครับ **
TEMPLATE_PATH = "test8_template_match/saved_template.png"

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
    # 1. โหลดไฟล์ Template อัตโนมัติตั้งแต่เริ่มโปรแกรม
    if not os.path.exists(TEMPLATE_PATH):
        print(f"[ERROR] Could not find template image at: {TEMPLATE_PATH}")
        print("Please run the previous code to save a template first.")
        sys.exit(1)
        
    template_patch = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
    th, tw = template_patch.shape
    print(f"[SUCCESS] Loaded template automatically from {TEMPLATE_PATH}")

    # 2. เปิดกล้อง
    Realsensed435Cam = DepthCamera(resolution_width, resolution_height)
    print("\n--- FULLY AUTOMATIC DETECTION RUNNING ---")
    print("Looking for the copper pipe opening...")
    print("Press 'q' when the green box is stable to extract 6-DOF and save.")

    while True:
        ret , depth_raw_frame, color_raw_frame = Realsensed435Cam.get_raw_frame()
        if not ret: continue
        
        color_frame = np.asanyarray(color_raw_frame.get_data())
        gray_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
        display_frame = color_frame.copy()

        # --- AUTO SCANNING (ไม่ต้องลากเมาส์แล้ว) ---
        res = cv2.matchTemplate(gray_frame, template_patch, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        target_pixel = None
        
        # ถ้าความมั่นใจเกิน 70% ให้ถือว่าเจอเป้าหมาย
        if max_val >= 0.70:
            top_left = max_loc
            bottom_right = (top_left[0] + tw, top_left[1] + th)
            target_pixel = (top_left[0] + tw // 2, top_left[1] + th // 2)
            
            cv2.rectangle(display_frame, top_left, bottom_right, (0, 255, 0), 2)
            cv2.circle(display_frame, target_pixel, 5, (0, 0, 255), -1)
            cv2.putText(display_frame, f"Match: {max_val*100:.1f}%", 
                        (top_left[0], top_left[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "Searching for pipe...", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(display_frame, "AUTO MODE: Press 'q' to Extract 6-DOF & Save.", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # โชว์ภาพแม่แบบเล็กๆ มุมขวาบน
        display_frame[10:10+th, 630-tw:630] = cv2.cvtColor(template_patch, cv2.COLOR_GRAY2BGR)

        cv2.imshow("Frame", display_frame)
        key = cv2.waitKey(1) & 0xFF
        
        # กดยืนยันเพื่อดึงค่า 3D (หรือคุณจะแก้โค้ดให้มันเซฟอัตโนมัติเมื่อ Match > 0.90 ติดต่อกัน 10 เฟรมก็ได้ครับ)
        if key == ord('q'):
            if target_pixel is None:
                print("\n[WARNING] Target is not locked. Cannot extract 3D data yet.")
                continue
                
            print(f"\n[PROCESSING] Extracting 3D Data at {target_pixel}...")
            
            # --- ดึงข้อมูล 3D และคำนวณ 6-DOF ---
            pcd = createPointCloudO3D(color_raw_frame, depth_raw_frame)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
            
            u, v = target_pixel
            intrinsics = depth_raw_frame.profile.as_video_stream_profile().intrinsics
            obj_points = np.asarray(pcd.points)
            
            raw_x, raw_y, raw_z = obj_points[:, 0], -obj_points[:, 1], -obj_points[:, 2]
            valid_z = raw_z != 0 
            u_proj = np.zeros_like(raw_x)
            v_proj = np.zeros_like(raw_y)
            
            u_proj[valid_z] = (raw_x[valid_z] * intrinsics.fx / raw_z[valid_z]) + intrinsics.ppx
            v_proj[valid_z] = (raw_y[valid_z] * intrinsics.fy / raw_z[valid_z]) + intrinsics.ppy
            
            dist_sq = (u_proj - u)**2 + (v_proj - v)**2
            best_idx = np.argmin(dist_sq)
            exact_target_pos = obj_points[best_idx]
            
            normal = np.asarray(pcd.normals)[best_idx]
            z_axis = normal / np.linalg.norm(normal)
            x_axis = np.array([1, 0, 0]) if abs(z_axis[0]) < 0.9 else np.array([0, 1, 0])
            y_axis = np.cross(z_axis, x_axis)
            y_axis /= np.linalg.norm(y_axis)
            x_axis = np.cross(y_axis, z_axis)
            
            rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
            roll, pitch, yaw = rotation_matrix_to_euler_angles(rotation_matrix)

            # --- พิมพ์ผลและบันทึก ---
            print("\n" + "="*50)
            print("TARGET 6-DOF DATA (FULLY AUTOMATIC):")
            print(f"Position (X, Y, Z): {exact_target_pos}")
            print(f"Orientation (R, P, Y): {roll:.2f}, {pitch:.2f}, {yaw:.2f}")
            print("="*50)

            point_marker_pcd = o3d.geometry.PointCloud()
            point_marker_pcd.points = o3d.utility.Vector3dVector([exact_target_pos])
            point_marker_pcd.colors = o3d.utility.Vector3dVector([[0, 1, 0]])

            try:
                o3d.io.write_point_cloud(os.path.join(SAVE_DIR, "target_object.ply"), pcd)
                o3d.io.write_point_cloud(os.path.join(SAVE_DIR, "green_point_marker.ply"), point_marker_pcd)
                with open(os.path.join(SAVE_DIR, "target_data.txt"), "w") as f:
                    f.write(f"Position_X: {exact_target_pos[0]:.6f}\n")
                    f.write(f"Position_Y: {exact_target_pos[1]:.6f}\n")
                    f.write(f"Position_Z: {exact_target_pos[2]:.6f}\n")
                    f.write(f"Roll: {roll:.2f}\n")
                    f.write(f"Pitch: {pitch:.2f}\n")
                    f.write(f"Yaw: {yaw:.2f}\n")
                print(f"[SUCCESS] Files saved to: {os.path.abspath(SAVE_DIR)}")
            except Exception as e:
                print(f"[ERROR] Could not save files: {e}")

            # --- แสดงผล 3D ---
            target_ball = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
            target_ball.paint_uniform_color([0, 1, 0])
            target_ball.translate(exact_target_pos)
            
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.04, origin=[0,0,0])
            axis.rotate(rotation_matrix, center=[0,0,0])
            axis.translate(exact_target_pos)

            mat_unlit = o3d.visualization.rendering.MaterialRecord()
            mat_unlit.shader = "defaultUnlit"
            
            o3d.visualization.draw([{"name": "pcd", "geometry": pcd, "material": mat_unlit},
                                   {"name": "target", "geometry": target_ball, "material": mat_unlit},
                                   {"name": "axis", "geometry": axis, "material": mat_unlit}],
                                   title="Fully Auto 6-DOF Target")
            break 

    Realsensed435Cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()