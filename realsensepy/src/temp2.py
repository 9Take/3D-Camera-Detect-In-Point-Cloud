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
resolution_width, resolution_height = (848, 480)

def get_config():
    """รับค่า configuration จาก command-line arguments หรือ user input"""
    parser = argparse.ArgumentParser(
        description='Real Sense 3D Point Cloud Detection - SIFT Homography (Freeze Capture)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('-f', '--folder', type=str, default=None, help='ชื่อ folder สำหรับเก็บข้อมูล (default: realsensepy/test1)')
    parser.add_argument('-t', '--target', type=str, default=None, help='ชื่อเป้าหมาย (default: target_A)')
    args = parser.parse_args()
    
    save_dir = args.folder
    if save_dir is None:
        save_dir = input("กรุณาป้อนชื่อ folder สำหรับเก็บข้อมูล [realsensepy/test1]: ").strip()
        if not save_dir: save_dir = "realsensepy/test1"
    
    target_name = args.target
    if target_name is None:
        target_name = input("กรุณาป้อนชื่อเป้าหมาย [target_A]: ").strip()
        if not target_name: target_name = "target_A"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"✓ สร้าง folder '{save_dir}' เรียบร้อย")
    
    print(f"\n--- ตั้งค่า Configuration ---")
    print(f"Save Folder: {os.path.abspath(save_dir)}")
    print(f"Target Name: {target_name}\n")
    return save_dir, target_name

SAVE_DIR, CURRENT_TARGET_NAME = get_config()

# Global Variables สำหรับ State และการวาด
app_state = 0 # 0: Live View, 1: Frozen Annotation, 2: Tracking
polygon_points = []
template_patch = None
exact_target_pixel_manual = None
target_offset = None

# เก็บภาพที่ถูก Freeze
frozen_color = None
frozen_gray = None

def draw_shape_callback(event, x, y, flags, param):
    global polygon_points, exact_target_pixel_manual, app_state
    
    # อนุญาตให้คลิกวาดได้เฉพาะตอนที่ภาพถูก Freeze แล้วเท่านั้น (State 1)
    if app_state != 1: 
        return
        
    if event == cv2.EVENT_LBUTTONDOWN:
        polygon_points.append((x, y))
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
    global polygon_points, template_patch, exact_target_pixel_manual, target_offset
    global app_state, frozen_color, frozen_gray
    
    print("\n[INIT] Initializing RealSense Camera...")
    Realsensed435Cam = DepthCamera(resolution_width, resolution_height)
    print("[INIT] Camera initialized successfully!")
    
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Frame", 848, 480)
    cv2.setMouseCallback("Frame", draw_shape_callback)

    sift = cv2.SIFT_create()
    kp_template, des_template = None, None

    print(f"\n--- PHASE 1: TARGET CAPTURE ---")
    print("1. เล็งกล้องไปที่เป้าหมาย แล้วกด 'SPACEBAR' เพื่อแช่ภาพ (Capture)")
    print("2. หลังจากภาพนิ่ง ให้คลิกซ้ายวาดกรอบ และคลิกขวาเพื่อระบุจุด Target")

    while True:
        # ดึงภาพจากกล้องแบบ Live เฉพาะ State 0 (เล็ง) และ State 2 (แทร็ก)
        if app_state == 0 or app_state == 2:
            ret , depth_raw_frame, color_raw_frame = Realsensed435Cam.get_raw_frame()
            if not ret: continue
            color_frame = np.asanyarray(color_raw_frame.get_data())
            gray_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)

        # ---------------------------------------------------------
        # STATE 0: เล็งเป้าหมาย (Live View) รอถ่ายภาพนิ่ง
        # ---------------------------------------------------------
        if app_state == 0:
            display_frame = color_frame.copy()
            cv2.putText(display_frame, "Aim camera -> Press SPACEBAR to Capture Frame", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("Frame", display_frame)
            key = cv2.waitKey(33) & 0xFF
            
            if key == 32: # SPACEBAR
                frozen_color = color_frame.copy()
                frozen_gray = gray_frame.copy()
                polygon_points.clear()
                exact_target_pixel_manual = None
                app_state = 1 # เปลี่ยนไปโหมดวาดบนภาพนิ่ง
                print("\n[FROZEN] Frame Captured! Now draw the box and select target.")

        # ---------------------------------------------------------
        # STATE 1: วาดกรอบและเลือกจุดบนภาพนิ่ง (Frozen Frame)
        # ---------------------------------------------------------
        elif app_state == 1:
            display_frame = frozen_color.copy()
            
            # วาดเส้นและจุดที่ผู้ใช้คลิก
            if len(polygon_points) > 0:
                pts = np.array(polygon_points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(display_frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                for pt in polygon_points:
                    cv2.circle(display_frame, pt, 4, (0, 255, 0), -1)

            if exact_target_pixel_manual is not None:
                cv2.circle(display_frame, exact_target_pixel_manual, 5, (0, 0, 255), -1)
                cv2.putText(display_frame, "Target", (exact_target_pixel_manual[0]+10, exact_target_pixel_manual[1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # เมนูสอนการใช้งาน
            cv2.putText(display_frame, "FROZEN: L-Click=Box | R-Click=Target | 's'=Save | 'r'=Retake | 'c'=Clear", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            
            cv2.imshow("Frame", display_frame)
            key = cv2.waitKey(33) & 0xFF

            if key == ord('c'):
                polygon_points.clear()
                exact_target_pixel_manual = None
            elif key == ord('r'):
                app_state = 0 # ยกเลิกรูปนี้ กลับไปเล็งใหม่
                print("\n[RETAKE] Returning to Live View...")
            elif key == ord('s') and len(polygon_points) > 2:
                # คำนวณขอบเขตจากจุดที่วาด
                x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(np.array(polygon_points))
                
                # ครอบ Template จากภาพที่ Freeze ไว้ (ไม่ใช่ Live)
                template_patch = frozen_gray[y_rect:y_rect+h_rect, x_rect:x_rect+w_rect]
                cv2.imwrite(os.path.join(SAVE_DIR, f"{CURRENT_TARGET_NAME}_template.png"), template_patch)
                
                if exact_target_pixel_manual is not None:
                    target_offset = (exact_target_pixel_manual[0] - x_rect, exact_target_pixel_manual[1] - y_rect)
                else:
                    target_offset = (w_rect // 2, h_rect // 2)
                    
                with open(os.path.join(SAVE_DIR, f"{CURRENT_TARGET_NAME}_offset.txt"), "w") as f:
                    f.write(f"{target_offset[0]},{target_offset[1]}")

                # ดึงฟีเจอร์ SIFT จาก Template เก็บไว้
                kp_template, des_template = sift.detectAndCompute(template_patch, None)
                    
                print(f"Target Offset Saved: X={target_offset[0]}, Y={target_offset[1]}")
                print(f"Extracted {len(kp_template)} SIFT keypoints from template.")
                print(f"\n[SUCCESS] Template '{CURRENT_TARGET_NAME}' Saved!")
                print("--- PHASE 2: LIVE AUTO DETECTION ---")
                app_state = 2 # เข้าสู่โหมด Tracking

        # ---------------------------------------------------------
        # STATE 2: สแกนหาอัตโนมัติด้วย SIFT (Live Tracking)
        # ---------------------------------------------------------
        elif app_state == 2:
            display_frame = color_frame.copy()
            target_pixel = None
            th, tw = template_patch.shape
            
            kp_frame, des_frame = sift.detectAndCompute(gray_frame, None)
            
            if des_template is not None and des_frame is not None and len(des_template) > 2 and len(des_frame) > 2:
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                
                try:
                    matches = flann.knnMatch(des_template, des_frame, k=2)
                    
                    good_matches = []
                    for match_pair in matches:
                        if len(match_pair) == 2:
                            m, n = match_pair
                            if m.distance < 0.7 * n.distance:
                                good_matches.append(m)
                    
                    MIN_MATCH_COUNT = 10
                    if len(good_matches) > MIN_MATCH_COUNT:
                        src_pts = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        
                        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                        
                        if M is not None:
                            pts = np.float32([[0, 0], [0, th - 1], [tw - 1, th - 1], [tw - 1, 0]]).reshape(-1, 1, 2)
                            dst = cv2.perspectiveTransform(pts, M)
                            
                            target_pt_src = np.float32([[[target_offset[0], target_offset[1]]]])
                            target_pt_dst = cv2.perspectiveTransform(target_pt_src, M)
                            target_pixel = (int(target_pt_dst[0][0][0]), int(target_pt_dst[0][0][1]))
                            
                            if 0 <= target_pixel[0] < resolution_width and 0 <= target_pixel[1] < resolution_height:
                                display_frame = cv2.polylines(display_frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
                                cv2.circle(display_frame, target_pixel, 5, (0, 0, 255), -1)
                                cv2.putText(display_frame, f"Homography Locked", 
                                            (int(dst[0][0][0]), int(dst[0][0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            else:
                                target_pixel = None 
                        else:
                            cv2.putText(display_frame, "Searching...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        cv2.putText(display_frame, "Searching...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                except Exception as e:
                    cv2.putText(display_frame, "Searching...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.putText(display_frame, "Searching...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.putText(display_frame, f"LIVE TRACKING: Press 'q' to Extract 6-DOF & Exit.", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # โชว์รูป Template อ้างอิงที่มุมซ้ายล่าง
            display_frame[10:10+th, 630-tw:630] = cv2.cvtColor(template_patch, cv2.COLOR_GRAY2BGR)

            cv2.imshow("Frame", display_frame)
            key = cv2.waitKey(33) & 0xFF
            
            # กด Q เพื่อดึงค่า 3D และแสดงผล
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
                                       title=f"SIFT Homography 6-DOF ({CURRENT_TARGET_NAME})")
                break 

    Realsensed435Cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()