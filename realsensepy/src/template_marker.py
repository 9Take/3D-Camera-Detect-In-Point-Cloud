import numpy as np
import pyrealsense2 as rs
import cv2
import open3d as o3d
import os
import math
import argparse
from realsense_depth import DepthCamera
from utils import createPointCloudO3D

# --- Configuration ---
resolution_width, resolution_height = (1280, 720)

def get_config():
    """รับค่า configuration จาก command-line arguments หรือ user input"""
    parser = argparse.ArgumentParser(
        description='Real Sense 3D Point Cloud Detection - Manual Target Selection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python juuuuuuu.py --folder realsensepy/test1 --target target_A
  python juuuuuuu.py -f realsensepy/my_data -t target_B
  python juuuuuuu.py  (จะถามผู้ใช้ input)
        '''
    )
    
    parser.add_argument(
        '-f', '--folder',
        type=str,
        default=None,
        help='ชื่อ folder สำหรับเก็บข้อมูล (default: realsensepy/test1)'
    )
    parser.add_argument(
        '-t', '--target',
        type=str,
        default=None,
        help='ชื่อเป้าหมาย (default: target_A)'
    )
    
    args = parser.parse_args()
    
    # ถ้าไม่มี argument ให้ถามผู้ใช้
    save_dir = args.folder
    if save_dir is None:
        save_dir = input("กรุณาป้อนชื่อ folder สำหรับเก็บข้อมูล [realsensepy/test1]: ").strip()
        if not save_dir:
            save_dir = "realsensepy/test1"
    
    target_name = args.target
    if target_name is None:
        target_name = input("กรุณาป้อนชื่อเป้าหมาย [target_A]: ").strip()
        if not target_name:
            target_name = "target_A"
    
    # สร้าง folder ถ้ายังไม่มี
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"✓ สร้าง folder '{save_dir}' เรียบร้อย")
    
    print(f"\n--- ตั้งค่า Configuration ---")
    print(f"Save Folder: {os.path.abspath(save_dir)}")
    print(f"Target Name: {target_name}\n")
    
    return save_dir, target_name

SAVE_DIR, CURRENT_TARGET_NAME = get_config()

polygon_points = []
template_patch = None
shape_saved = False

# ตัวแปรใหม่สำหรับเก็บจุดเป้าหมายแบบกำหนดเอง
exact_target_pixel_manual = None
target_offset = None

def draw_shape_callback(event, x, y, flags, param):
    global polygon_points, exact_target_pixel_manual
    
    # คลิกซ้ายเพื่อวาดกรอบ Template
    if event == cv2.EVENT_LBUTTONDOWN:
        polygon_points.append((x, y))
        
    # คลิกขวาเพื่อมาร์ค "จุดสีแดง" เป้าหมายด้วยตัวเอง
    elif event == cv2.EVENT_RBUTTONDOWN:
        exact_target_pixel_manual = (x, y)
        print(f"Target Point Manually Set to: ({x}, {y})")

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
    global polygon_points, template_patch, shape_saved, exact_target_pixel_manual, target_offset
    
    print("\n[INIT] Initializing RealSense Camera...")
    Realsensed435Cam = DepthCamera(resolution_width, resolution_height)
    print("[INIT] Camera initialized successfully!")
    
    print("[INIT] Creating display window...")
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Frame", 848, 480)
    cv2.setMouseCallback("Frame", draw_shape_callback)
    print("[INIT] Window created!")

    print(f"\n--- PHASE 1: DRAW TEMPLATE FOR '{CURRENT_TARGET_NAME}' ---")
    print("1. LEFT CLICK to draw a box around the pipe opening.")
    print("2. RIGHT CLICK to manually mark the exact Target Point (Red dot).")
    print("3. Press 'c' to CLEAR.")
    print("4. Press 's' to SAVE and start Auto-Detection preview.")

    while True:
        ret , depth_raw_frame, color_raw_frame = Realsensed435Cam.get_raw_frame()
        if not ret: continue
        
        color_frame = np.asanyarray(color_raw_frame.get_data())
        gray_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
        display_frame = color_frame.copy()

        # ---------------------------------------------------------
        # PHASE 1: วาดกรอบ + กำหนดจุด Target ด้วยตัวเอง
        # ---------------------------------------------------------
        if not shape_saved:
            cv2.putText(display_frame, "LEFT: Draw Box | RIGHT: Mark Target | 's': Save", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            
            # วาดเส้นขอบ
            if len(polygon_points) > 0:
                pts = np.array(polygon_points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(display_frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                for pt in polygon_points:
                    cv2.circle(display_frame, pt, 4, (0, 255, 0), -1)

            # วาดจุด Target ที่คลิกขวาเอาไว้
            if exact_target_pixel_manual is not None:
                cv2.circle(display_frame, exact_target_pixel_manual, 5, (0, 0, 255), -1)
                cv2.putText(display_frame, "Target", (exact_target_pixel_manual[0]+10, exact_target_pixel_manual[1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            try:
                cv2.imshow("Frame", display_frame)
            except Exception as e:
                print(f"[ERROR] Failed to display frame: {e}")
            
            key = cv2.waitKey(33) & 0xFF

            if key == ord('c'):
                polygon_points.clear()
                exact_target_pixel_manual = None
                
            elif key == ord('s') and len(polygon_points) > 2:
                # หา Bounding Box จากจุดที่วาด
                x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(np.array(polygon_points))
                
                template_patch = gray_frame[y_rect:y_rect+h_rect, x_rect:x_rect+w_rect]
                
                # เซฟไฟล์รูป Template
                cv2.imwrite(os.path.join(SAVE_DIR, f"{CURRENT_TARGET_NAME}_template.png"), template_patch)
                
                # ถ้าไม่ได้คลิกขวาไว้ ให้ใช้จุดกึ่งกลางเหมือนเดิม
                if exact_target_pixel_manual is not None:
                    # คำนวณระยะห่าง (Offset) ระหว่างมุมซ้ายบนของกรอบ กับ จุดเป้าหมาย
                    target_offset = (exact_target_pixel_manual[0] - x_rect, exact_target_pixel_manual[1] - y_rect)
                else:
                    target_offset = (w_rect // 2, h_rect // 2)
                    
                # เซฟไฟล์ Text Offset ควบคู่ไปด้วย
                with open(os.path.join(SAVE_DIR, f"{CURRENT_TARGET_NAME}_offset.txt"), "w") as f:
                    f.write(f"{target_offset[0]},{target_offset[1]}")
                    
                print(f"Target Offset Saved: X={target_offset[0]}, Y={target_offset[1]}")
                print(f"\n[SUCCESS] Template '{CURRENT_TARGET_NAME}' Captured and Saved!")
                print("--- PHASE 2: AUTO DETECTION PREVIEW ---")
                shape_saved = True
                
        # ---------------------------------------------------------
        # PHASE 2: สแกนหาอัตโนมัติด้วย Template Matching (สำหรับพรีวิวทดสอบ)
        # ---------------------------------------------------------
        else:
            res = cv2.matchTemplate(gray_frame, template_patch, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
            target_pixel = None
            th, tw = template_patch.shape
            
            if max_val >= 0.60:
                top_left = max_loc
                bottom_right = (top_left[0] + tw, top_left[1] + th)
                
                # ใช้ระยะ Offset ที่คำนวณไว้ตอนวาด มาบวกเพิ่มแทนที่จะใช้จุดกึ่งกลาง
                target_pixel = (top_left[0] + target_offset[0], top_left[1] + target_offset[1])
                
                cv2.rectangle(display_frame, top_left, bottom_right, (0, 255, 0), 2)
                cv2.circle(display_frame, target_pixel, 5, (0, 0, 255), -1)
                cv2.putText(display_frame, f"Match: {max_val*100:.1f}%", 
                            (top_left[0], top_left[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "Searching...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.putText(display_frame, f"PREVIEW MODE: Press 'q' to Extract 6-DOF & Exit.", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            display_frame[10:10+th, 630-tw:630] = cv2.cvtColor(template_patch, cv2.COLOR_GRAY2BGR)

            try:
                cv2.imshow("Frame", display_frame)
            except Exception as e:
                print(f"[ERROR] Failed to display frame: {e}")
            
            key = cv2.waitKey(33) & 0xFF
            
            if key == ord('q'):
                if target_pixel is None:
                    print("\n[WARNING] Please wait until the target is locked before pressing 'q'.")
                    continue
                    
                print(f"\n[PROCESSING] Extracting 3D Data at {target_pixel}...")
                
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

                print("\n" + "="*50)
                print(f"TARGET '{CURRENT_TARGET_NAME}' 6-DOF DATA:")
                print(f"Position (X, Y, Z): {exact_target_pos}")
                print(f"Orientation (R, P, Y): {roll:.2f}, {pitch:.2f}, {yaw:.2f}")
                print("="*50)

                point_marker_pcd = o3d.geometry.PointCloud()
                point_marker_pcd.points = o3d.utility.Vector3dVector([exact_target_pos])
                point_marker_pcd.colors = o3d.utility.Vector3dVector([[0, 1, 0]])

                try:
                    o3d.io.write_point_cloud(os.path.join(SAVE_DIR, f"{CURRENT_TARGET_NAME}_object.ply"), pcd)
                    o3d.io.write_point_cloud(os.path.join(SAVE_DIR, f"{CURRENT_TARGET_NAME}_marker.ply"), point_marker_pcd)
                    with open(os.path.join(SAVE_DIR, f"{CURRENT_TARGET_NAME}_data.txt"), "w") as f:
                        f.write(f"Position_X: {exact_target_pos[0]:.6f}\n")
                        f.write(f"Position_Y: {exact_target_pos[1]:.6f}\n")
                        f.write(f"Position_Z: {exact_target_pos[2]:.6f}\n")
                        f.write(f"Roll: {roll:.2f}\n")
                        f.write(f"Pitch: {pitch:.2f}\n")
                        f.write(f"Yaw: {yaw:.2f}\n")
                    print(f"[SUCCESS] Files saved to: {os.path.abspath(SAVE_DIR)}")
                except Exception as e:
                    print(f"[ERROR] Could not save files: {e}")

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
                                       title=f"Manual Target Matched 6-DOF ({CURRENT_TARGET_NAME})")
                break 

    Realsensed435Cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()