import numpy as np
import pyrealsense2 as rs
import cv2
import open3d as o3d
import os
import math
import sys
import argparse
import time
import struct
import pymcprotocol
from realsense_depth import DepthCamera
from utils import createPointCloudO3D
from collections import defaultdict

# --- Configuration ---
# ปรับลดความละเอียดลงเพื่อให้ FPS สูงขึ้น ประมวลผลได้เร็วและลื่นไหลขึ้น
resolution_width, resolution_height = (848, 480)

# --- PLC Configuration ---
PLC_IP = "192.168.1.165"  # IP ของ PLC
PLC_PORT = 5010           # Port ที่ตั้งไว้ใน PLC (MC Protocol)
START_D_REG = 1001        # Data Register เริ่มต้น (D1001)

def float_to_scaled_16bit(val):
    """
    คูณ 100, ปัดเศษเป็นจำนวนเต็ม และส่งกลับเป็น List 
    (ให้ pymcprotocol จัดการเรื่องแปลง binary ลง PLC เอง)
    """
    scaled_val = int(round(val * 100))
    
    # ป้องกันค่าล้น (Overflow) ของ 16-bit signed integer
    if scaled_val > 32767: scaled_val = 32767
    if scaled_val < -32768: scaled_val = -32768
        
    # ส่งกลับเป็น list ที่มีสมาชิก 1 ตัว
    return [scaled_val]

def send_to_plc(data_A):
    """ส่งเฉพาะข้อมูล X, Y, Z ของ A เข้า PLC"""
    try:
        print(f"\n[PLC] Connecting to {PLC_IP}:{PLC_PORT}...")
        plc = pymcprotocol.Type3E()
        plc.setaccessopt(commtype="binary")
        plc.connect(PLC_IP, PLC_PORT)
        payload = []
        
        # ใช้เฉพาะ 3 ค่าแรก (X, Y, Z) จาก data_A
        for val in data_A[:3]:
            payload.extend(float_to_scaled_16bit(val))
        
        # --- แทรกลอง Print ดูว่า payload หน้าตาเป็นยังไง ---
        print(f"[DEBUG] Payload list = {payload}")
        print(f"[DEBUG] Will write to: D{START_D_REG}, D{START_D_REG+1}, D{START_D_REG+2}")
        
        # เขียนเข้า PLC 
        plc.batchwrite_wordunits(f"D{START_D_REG}", payload)
        
        print(f"[PLC] SUCCESS! Scaled Data (x100) sent to PLC:")
        print(f"      X -> D{START_D_REG}")
        print(f"      Y -> D{START_D_REG+1}")
        print(f"      Z -> D{START_D_REG+2}")
        plc.close()
        
    except Exception as e:
        print(f"[PLC ERROR] Failed to send data: {e}")

def get_config():
    parser = argparse.ArgumentParser(description='Real Sense 3D Target A - 0.5s Timer & PLC')
    parser.add_argument(
        '-td', '--template-dir', 
        type=str, 
        default="/home/progren/Intership Project/HeatExchangerPointCloudDetection/3D-Camera-Detect-In-Point-Cloud/realsensepy/src/ju",
        help='โฟลเดอร์ที่เก็บ Template และ Offset (default: ju)'
    )
    parser.add_argument('-t', '--targets', type=str, nargs='+', default=None)
    parser.add_argument('-sd', '--save-dir', type=str, default=None)
    args = parser.parse_args()
    
    template_dir = args.template_dir
    if not os.path.exists(template_dir):
        print(f"[ERROR] Template directory {template_dir} not found!")
        sys.exit(1)
    
    save_dir = args.save_dir if args.save_dir else f"{template_dir}_output"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created folder: {save_dir}")
    
    return template_dir, save_dir

TEMPLATE_DIR, SAVE_DIR = get_config()
COLORS = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]

def load_all_template_versions(template_dir, sift_detector):
    templates_by_target = defaultdict(list)
    files = os.listdir(template_dir)
    template_files = defaultdict(list)
    
    for filename in files:
        if filename.endswith('_template.png'):
            base_name = filename.replace('_template.png', '')
            template_files[base_name].append(filename)
    
    if not template_files:
        print(f"[ERROR] No template files found in {template_dir}")
        sys.exit(1)
    
    for base_name in sorted(template_files.keys()):
        template_file = os.path.join(template_dir, f"{base_name}_template.png")
        offset_file = os.path.join(template_dir, f"{base_name}_offset.txt")
        if not os.path.exists(template_file) or not os.path.exists(offset_file): continue
        
        try:
            with open(offset_file, 'r') as f:
                data = f.read().strip().split(',')
                offset = (int(data[0]), int(data[1]))
            
            template = cv2.imread(template_file, cv2.IMREAD_GRAYSCALE)
            if template is None or len(template.shape) != 2: continue
            
            kp, des = sift_detector.detectAndCompute(template, None)
            if des is None or len(des) < 5: continue
                
            target_letter = base_name.split('.')[0]
            templates_by_target[target_letter].append((template, offset, base_name, kp, des))
            print(f"    ✓ Loaded {base_name} (Features: {len(kp)})")
        except Exception as e:
            continue
            
    if not templates_by_target: sys.exit(1)
    return templates_by_target

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
    sift = cv2.SIFT_create()
    templates_by_target = load_all_template_versions(TEMPLATE_DIR, sift)
    target_list = sorted(templates_by_target.keys())
    
    FLANN_INDEX_KDTREE = 1
    flann = cv2.FlannBasedMatcher(dict(algorithm=FLANN_INDEX_KDTREE, trees=5), dict(checks=50))
    
    print(f"[INIT] Initializing RealSense Camera...")
    Realsensed435Cam = DepthCamera(resolution_width, resolution_height)
    # cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Frame", 848, 480)
    
    print("\n--- TARGET A AUTO TRACKING (0.5s TIMER) ---")
    
    best_matches = {}
    lock_start_time = None 
    trigger_extraction = False

    while True:
        ret , depth_raw_frame, color_raw_frame = Realsensed435Cam.get_raw_frame()
        if not ret: continue
        
        color_frame = np.asanyarray(color_raw_frame.get_data())
        gray_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
        display_frame = color_frame.copy()

        detected_pixels = []
        best_matches.clear()
        
        kp_frame, des_frame = sift.detectAndCompute(gray_frame, None)

        if des_frame is not None and len(des_frame) > 2:
            for target_idx, target_name in enumerate(target_list):
                versions = templates_by_target[target_name]
                best_inliers = 0
                best_match_data = None
                
                for template_img, offset, version_name, kp_temp, des_temp in versions:
                    try:
                        matches = flann.knnMatch(des_temp, des_frame, k=2)
                        good_matches = [m for match_pair in matches if len(match_pair) == 2 for m, n in [match_pair] if m.distance < 0.7 * n.distance]
                        
                        if len(good_matches) > 10:
                            src_pts = np.float32([kp_temp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                            
                            if M is not None:
                                inliers = int(np.sum(mask))
                                if inliers > best_inliers:
                                    best_inliers = inliers
                                    best_match_data = {'M': M, 'offset': offset, 'version': version_name, 'template': template_img, 'inliers': inliers}
                    except: continue
                
                if best_inliers >= 12: 
                    M = best_match_data['M']
                    th, tw = best_match_data['template'].shape
                    
                    pts = np.float32([[0, 0], [0, th - 1], [tw - 1, th - 1], [tw - 1, 0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, M)
                    target_pt_dst = cv2.perspectiveTransform(np.float32([[[best_match_data['offset'][0], best_match_data['offset'][1]]]]), M)
                    target_pixel = (int(target_pt_dst[0][0][0]), int(target_pt_dst[0][0][1]))
                    
                    if 0 <= target_pixel[0] < resolution_width and 0 <= target_pixel[1] < resolution_height:
                        detected_pixels.append((target_idx, target_pixel))
                        best_matches[target_name] = {'version': best_match_data['version'], 'inliers': best_inliers}
                        
                        color = COLORS[target_idx % len(COLORS)]
                        display_frame = cv2.polylines(display_frame, [np.int32(dst)], True, color, 3, cv2.LINE_AA)
                        cv2.circle(display_frame, target_pixel, 5, (0, 0, 255), -1)
                        cv2.putText(display_frame, f"{target_name}: {best_inliers} Inliers", (int(dst[0][0][0]), int(dst[0][0][1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        detected_names = [target_list[idx] for idx, _ in detected_pixels]
        
        # --- Timer Logic (เฉพาะ Target A และ 0.5 วิ) ---
        if 'A' in detected_names:
            if lock_start_time is None:
                lock_start_time = time.time()
                
            elapsed_time = time.time() - lock_start_time
            remaining_time = max(0.0, 0.5 - elapsed_time)
            
            overlay_text = f"LOCKED A: Extracting in {remaining_time:.1f}s"
            cv2.putText(display_frame, overlay_text, (50, 60), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 3)
            
            if elapsed_time >= 0.5:
                trigger_extraction = True
        else:
            lock_start_time = None
            trigger_extraction = False
            cv2.putText(display_frame, "WAITING FOR TARGET (A)...", (50, 60), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)

        # cv2.imshow("Frame", display_frame)
        # key = cv2.waitKey(33) & 0xFF
        
        # if key == 27: # ปุ่ม ESC เพื่อออก
        #     break
        
        # --- Data Extraction & PLC Transmission ---
        if trigger_extraction:
            print(f"\n[PROCESSING] Timer Reached! Capturing 70 frames for depth averaging...")
            
            depth_sum = np.zeros((resolution_height, resolution_width), dtype=np.float32)
            color_sum = np.zeros((resolution_height, resolution_width, 3), dtype=np.float32)
            valid_depth_count = np.zeros((resolution_height, resolution_width), dtype=np.float32)
            
            frames_captured = 0
            last_depth_frame = None

            while frames_captured < 70:
                ret_cap, depth_cap, color_cap = Realsensed435Cam.get_raw_frame()
                if not ret_cap: continue
                
                d_arr = np.asanyarray(depth_cap.get_data(), dtype=np.float32)
                c_arr = np.asanyarray(color_cap.get_data(), dtype=np.float32)
                
                mask = d_arr > 0
                depth_sum[mask] += d_arr[mask]
                valid_depth_count[mask] += 1
                color_sum += c_arr
                
                last_depth_frame = depth_cap
                frames_captured += 1
                cv2.waitKey(10)

            print("[PROCESSING] Generating Open3D Point Cloud...")
            valid_depth_count[valid_depth_count == 0] = 1
            avg_depth = depth_sum / valid_depth_count
            avg_depth = np.float32(avg_depth)
            avg_depth = cv2.bilateralFilter(avg_depth, d=5, sigmaColor=10.0, sigmaSpace=10.0)
            
            avg_depth_uint16 = np.clip(avg_depth, 0, 65535).astype(np.uint16)
            avg_color_uint8 = np.clip(color_sum / 70.0, 0, 255).astype(np.uint8)
            
            o3d_color = o3d.geometry.Image(cv2.cvtColor(avg_color_uint8, cv2.COLOR_BGR2RGB))
            o3d_depth = o3d.geometry.Image(avg_depth_uint16)
            
            intrinsics = last_depth_frame.profile.as_video_stream_profile().intrinsics
            depth_scale = last_depth_frame.get_units()
            
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d_color, o3d_depth, depth_scale=1.0 / depth_scale, depth_trunc=1.5, convert_rgb_to_intensity=False)
            
            o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
                resolution_width, resolution_height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)
            
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d_intrinsics)
            pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
            
            pcd_tree = o3d.geometry.KDTreeFlann(pcd)
            
            extracted_6dof = {}

            for target_idx, target_pixel in detected_pixels:
                u, v = target_pixel
                target_name = target_list[target_idx]
                
                target_z_raw = avg_depth[v, u]
                if target_z_raw <= 0 or (target_z_raw * depth_scale) > 1.5: continue
                    
                target_z = target_z_raw * depth_scale
                target_x = (u - intrinsics.ppx) * target_z / intrinsics.fx
                target_y = (v - intrinsics.ppy) * target_z / intrinsics.fy
                
                exact_target_pos = np.array([target_x, -target_y, -target_z])
                [k, idx, _] = pcd_tree.search_knn_vector_3d(exact_target_pos, 1)
                
                if k == 0: continue
                normal = np.asarray(pcd.normals)[idx[0]]
                
                z_axis = normal / np.linalg.norm(normal)
                x_axis = np.array([1, 0, 0]) if abs(z_axis[0]) < 0.9 else np.array([0, 1, 0])
                y_axis = np.cross(z_axis, x_axis)
                y_axis /= np.linalg.norm(y_axis)
                x_axis = np.cross(y_axis, z_axis)
                
                rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
                roll, pitch, yaw = rotation_matrix_to_euler_angles(rotation_matrix)

                extracted_6dof[target_name] = [target_x, -target_y, -target_z, roll, pitch, yaw]

                print("\n" + "-"*50)
                print(f"[{target_name.upper()}] DATA (Raw -> Scaled x100):")
                print(f"X: {target_x:.4f} -> {int(round(target_x*100))}")
                print(f"Y: {-target_y:.4f} -> {int(round(-target_y*100))}")
                print(f"Z: {-target_z:.4f} -> {int(round(-target_z*100))}")
                print(f"Roll: {roll:.2f}°")
                print(f"Pitch: {pitch:.2f}°")
                print(f"Yaw: {yaw:.2f}°")

            # สั่ง Save ไฟล์ .ply เผื่อต้องการตรวจสอบ 3D ย้อนหลัง
            o3d.io.write_point_cloud(os.path.join(SAVE_DIR, "scene_object.ply"), pcd)

            # ส่งข้อมูลเข้า PLC เฉพาะเป้าหมาย A
            if 'A' in extracted_6dof:
                send_to_plc(extracted_6dof['A'])
            else:
                print("\n[WARNING] Could not extract valid 3D data for Target A. PLC Transmission Cancelled.")

            print("-"*50)
            
            # วนกลับไปรอเป้าหมายรอบใหม่แบบ Loop ต่อเนื่อง
            lock_start_time = None
            trigger_extraction = False
            print("\n[INFO] Resuming live view. Waiting for Target A...")
            
    Realsensed435Cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()