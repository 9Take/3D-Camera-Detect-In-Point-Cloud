import numpy as np
import pyrealsense2 as rs
import cv2
import open3d as o3d
import os
import math
import sys
import json
from realsense_depth import DepthCamera
from utils import createPointCloudO3D

# --- Configuration ---
resolution_width, resolution_height = (640, 480)
SAVE_DIR = "test10_multi_target"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

TEMPLATE_PATHS = [
    "test8_template_match/saved_template2.png", 
    "test8_template_match/saved_template1.png"
]

COLORS = [(0, 255, 0), (255, 0, 0)] 
OFFSET_FILE = os.path.join(SAVE_DIR, "template_offsets.json")

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

def get_manual_offset(template_img, title):
    """
    ฟังก์ชันขยายภาพ Template ขึ้น 8 เท่า เพื่อให้ผู้ใช้คลิกเลือก
    ตำแหน่ง Target Point (จุดสีแดง) ได้อย่างแม่นยำ
    """
    offset = None
    scale = 8 
    h, w = template_img.shape
    
    # ขยายภาพแบบเห็นพิกเซลชัดเจน (INTER_NEAREST)
    display_img = cv2.resize(template_img, (w*scale, h*scale), interpolation=cv2.INTER_NEAREST)
    display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)

    def mouse_cb(event, x, y, flags, param):
        nonlocal offset
        if event == cv2.EVENT_LBUTTONDOWN:
            offset = (x // scale, y // scale)

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, mouse_cb)

    print(f"\n--- [CALIBRATION] {title} ---")
    print("1. LEFT CLICK on the image to mark the exact RED TARGET POINT.")
    print("2. Press 'ENTER' to confirm.")

    while True:
        temp_display = display_img.copy()
        if offset is not None:
            # วาดจุดสีแดงให้ตรงกับพิกเซลที่คลิก
            center_x = offset[0]*scale + scale//2
            center_y = offset[1]*scale + scale//2
            cv2.circle(temp_display, (center_x, center_y), 6, (0, 0, 255), -1)
        
        cv2.putText(temp_display, "Click target, then press ENTER", (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
        cv2.imshow(title, temp_display)
        key = cv2.waitKey(10) & 0xFF
        if key == 13 and offset is not None: # ENTER key
            break
            
    cv2.destroyWindow(title)
    return offset

def main():
    # 1. โหลด Template ทั้งหมด
    templates = []
    for path in TEMPLATE_PATHS:
        if not os.path.exists(path):
            print(f"[ERROR] Missing template: {path}")
            sys.exit(1)
        tpl = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        templates.append(tpl)
    
    print(f"[SUCCESS] Loaded {len(templates)} templates successfully.")

    # 2. จัดการ Target Offsets (โหมดตั้งค่าจุดแม่นยำ)
    offsets = []
    if os.path.exists(OFFSET_FILE):
        with open(OFFSET_FILE, 'r') as f:
            offsets = json.load(f)
        print(f"[INFO] Loaded custom target offsets from {OFFSET_FILE}")
        
        # ถ้ารูป Template ไม่เท่ากับค่าที่เซฟไว้ ให้ตั้งค่าใหม่
        if len(offsets) != len(templates):
            print("[WARNING] Template count changed. Recalibrating offsets...")
            offsets = []

    # หากยังไม่มีการตั้งค่า Offset (รันครั้งแรก) ให้เปิดหน้าต่าง Calibration
    if not offsets:
        for i, tpl in enumerate(templates):
            off = get_manual_offset(tpl, f"Calibration: Template {i+1}")
            offsets.append(off)
        # บันทึกลงไฟล์เพื่อใช้ครั้งต่อไป
        with open(OFFSET_FILE, 'w') as f:
            json.dump(offsets, f)
        print(f"[SUCCESS] Target offsets saved to {OFFSET_FILE}")

    # 3. เปิดกล้อง
    Realsensed435Cam = DepthCamera(resolution_width, resolution_height)
    print("\n--- MULTI-TARGET AUTO DETECTION ---")
    print("Looking for all targets...")
    print("Press 'q' when targets are locked to extract 6-DOF.")

    while True:
        ret , depth_raw_frame, color_raw_frame = Realsensed435Cam.get_raw_frame()
        if not ret: continue
        
        color_frame = np.asanyarray(color_raw_frame.get_data())
        gray_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
        display_frame = color_frame.copy()

        detected_pixels = [] 

        # --- สแกนหาเป้าหมาย ---
        for idx, template_patch in enumerate(templates):
            th, tw = template_patch.shape
            res = cv2.matchTemplate(gray_frame, template_patch, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
            if max_val >= 0.70:
                top_left = max_loc
                bottom_right = (top_left[0] + tw, top_left[1] + th)
                
                # นำ Offset ที่ผู้ใช้กำหนดมาบวกกับพิกัดมุมซ้ายบนของกรอบ
                target_offset = offsets[idx]
                target_center = (top_left[0] + target_offset[0], top_left[1] + target_offset[1])
                
                detected_pixels.append((idx, target_center)) 
                
                color = COLORS[idx % len(COLORS)]
                cv2.rectangle(display_frame, top_left, bottom_right, color, 2)
                # วาดจุดเป้าหมายตามที่ปรับตั้งไว้
                cv2.circle(display_frame, target_center, 5, (0, 0, 255), -1)
                cv2.putText(display_frame, f"T{idx+1} Match: {max_val*100:.1f}%", 
                            (top_left[0], top_left[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                display_frame[10 + (idx * (th + 5)):10 + (idx * (th + 5)) + th, 630-tw:630] = cv2.cvtColor(template_patch, cv2.COLOR_GRAY2BGR)

        cv2.putText(display_frame, f"Found {len(detected_pixels)}/{len(templates)} Targets. Press 'q' to Save.", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow("Frame", display_frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            if len(detected_pixels) == 0:
                print("\n[WARNING] No targets locked. Cannot extract.")
                continue
                
            print(f"\n[PROCESSING] Extracting 3D Data for {len(detected_pixels)} targets...")
            
            pcd = createPointCloudO3D(color_raw_frame, depth_raw_frame)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
            intrinsics = depth_raw_frame.profile.as_video_stream_profile().intrinsics
            obj_points = np.asarray(pcd.points)
            
            all_markers = []
            
            for target_id, target_pixel in detected_pixels:
                u, v = target_pixel
                
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

                print("\n" + "-"*40)
                print(f"TARGET {target_id + 1} 6-DOF DATA:")
                print(f"Position (X, Y, Z): {exact_target_pos}")
                print(f"Orientation (R, P, Y): {roll:.2f}, {pitch:.2f}, {yaw:.2f}")

                txt_path = os.path.join(SAVE_DIR, f"target_{target_id+1}_data.txt")
                with open(txt_path, "w") as f:
                    f.write(f"Position_X: {exact_target_pos[0]:.6f}\n")
                    f.write(f"Position_Y: {exact_target_pos[1]:.6f}\n")
                    f.write(f"Position_Z: {exact_target_pos[2]:.6f}\n")
                    f.write(f"Roll: {roll:.2f}\n")
                    f.write(f"Pitch: {pitch:.2f}\n")
                    f.write(f"Yaw: {yaw:.2f}\n")
                
                target_ball = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
                bgr_color = COLORS[target_id % len(COLORS)]
                target_ball.paint_uniform_color([bgr_color[2]/255.0, bgr_color[1]/255.0, bgr_color[0]/255.0])
                target_ball.translate(exact_target_pos)
                
                axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.04, origin=[0,0,0])
                axis.rotate(rotation_matrix, center=[0,0,0])
                axis.translate(exact_target_pos)
                
                all_markers.extend([target_ball, axis])

            print("-"*40)
            print(f"[SUCCESS] All target files saved to: {os.path.abspath(SAVE_DIR)}")

            o3d.io.write_point_cloud(os.path.join(SAVE_DIR, "scene_object.ply"), pcd)

            mat_unlit = o3d.visualization.rendering.MaterialRecord()
            mat_unlit.shader = "defaultUnlit"
            
            geometries = [{"name": "pcd", "geometry": pcd, "material": mat_unlit}]
            for i, marker in enumerate(all_markers):
                geometries.append({"name": f"marker_{i}", "geometry": marker, "material": mat_unlit})
                
            o3d.visualization.draw(geometries, title="Multi-Target 6-DOF (Custom Offsets)")
            break 

    Realsensed435Cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()