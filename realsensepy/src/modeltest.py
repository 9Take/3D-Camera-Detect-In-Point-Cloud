import numpy as np
import pyrealsense2 as rs
import cv2
import open3d as o3d
import os
import math
import sys
import argparse
from realsense_depth import DepthCamera
from utils import createPointCloudO3D
from collections import defaultdict

# --- Configuration ---
resolution_width, resolution_height = (1280, 720)  # Balanced: Better quality without excessive slowdown

def get_config():
    """รับค่า configuration จาก command-line arguments หรือ user input"""
    parser = argparse.ArgumentParser(
        description='Real Sense 3D Multi-Target Detection with Multi-Version Templates',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python modeltest.py --template-dir Model1 --targets A B --save-dir my_output
  python modeltest.py -td Model1 -t A B -sd my_output
  python modeltest.py  (จะถามผู้ใช้ input)
        '''
    )
    
    parser.add_argument(
        '-td', '--template-dir',
        type=str,
        default="Model1",
        help='โฟลเดอร์ที่เก็บ Template และ Offset (default: Model1)'
    )
    parser.add_argument(
        '-t', '--targets',
        type=str,
        nargs='+',
        default=None,
        help='ชื่อเป้าหมาย เช่น A B (default: auto-detect from folder)'
    )
    parser.add_argument(
        '-sd', '--save-dir',
        type=str,
        default=None,
        help='โฟลเดอร์สำหรับบันทึกผลลัพธ์ (default: Model1_output)'
    )
    
    args = parser.parse_args()
    
    template_dir = args.template_dir
    if not os.path.exists(template_dir):
        print(f"[ERROR] Template directory {template_dir} not found!")
        sys.exit(1)
    
    save_dir = args.save_dir
    if save_dir is None:
        save_dir = f"{template_dir}_output"
    
    # สร้าง folder ถ้ายังไม่มี
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created folder: {save_dir}")
    
    print(f"\n--- Configuration ---")
    print(f"Template Directory: {template_dir}")
    print(f"Save Directory: {save_dir}\n")
    
    return template_dir, save_dir

TEMPLATE_DIR, SAVE_DIR = get_config()
COLORS = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0), (255, 0, 255)]

def load_all_template_versions(template_dir):
    """
    Scan folder for all template PNG files and corresponding offset files
    Example: A.1_template.png, A.2_template.png, B.1_template.png, B.2_template.png
    Returns: dict[target_name] -> list of (template_image, offset_tuple, version_name)
    """
    templates_by_target = defaultdict(list)
    
    # Scan all files in folder
    files = os.listdir(template_dir)
    
    # Group files by version name: A.1, A.2, B.1, etc.
    template_files = defaultdict(list)
    
    for filename in files:
        if filename.endswith('_template.png'):
            # Extract base name like "A.1" from "A.1_template.png"
            base_name = filename.replace('_template.png', '')
            template_files[base_name].append(filename)
    
    if not template_files:
        print(f"[ERROR] No template PNG files found in {template_dir}")
        print("Looking for files like: A.1_template.png, A.2_template.png, B.1_template.png, etc.")
        sys.exit(1)
    
    # Load each version and its offset
    for base_name in sorted(template_files.keys()):
        template_file = os.path.join(template_dir, f"{base_name}_template.png")
        offset_file = os.path.join(template_dir, f"{base_name}_offset.txt")
        
        if not os.path.exists(template_file):
            print(f"[WARNING] Template file not found: {base_name}_template.png")
            continue
        if not os.path.exists(offset_file):
            print(f"[WARNING] Offset file not found: {base_name}_offset.txt")
            continue
        
        # Load offset
        try:
            with open(offset_file, 'r') as f:
                data = f.read().strip().split(',')
                offset = (int(data[0]), int(data[1]))
        except Exception as e:
            print(f"[ERROR] Failed to read offset from {offset_file}: {e}")
            continue
        
        # Load template PNG image
        try:
            template = cv2.imread(template_file, cv2.IMREAD_GRAYSCALE)
            if template is None:
                print(f"[WARNING] Could not read image: {base_name}_template.png")
                continue
            
            if len(template.shape) != 2:
                print(f"[WARNING] {base_name}_template.png is not grayscale, skipping...")
                continue
            
        except Exception as e:
            print(f"[ERROR] Failed to load {base_name}_template.png: {e}")
            continue
        
        # Extract target letter (A, B, C, etc.)
        target_letter = base_name.split('.')[0]
        templates_by_target[target_letter].append((template, offset, base_name))
        print(f"    ✓ Loaded {base_name} (size: {template.shape})")
    
    if not templates_by_target:
        print("[ERROR] No templates loaded successfully!")
        sys.exit(1)
    
    print(f"\n[SUCCESS] Loaded templates for {len(templates_by_target)} targets:")
    for target_name, versions in templates_by_target.items():
        print(f"  {target_name}: {len(versions)} version(s)")
    print()
    
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
    # โหลด templates ทั้งหมด (หลาย versions)
    templates_by_target = load_all_template_versions(TEMPLATE_DIR)
    
    # สร้าง list ของ target ที่พบ
    target_list = sorted(templates_by_target.keys())
    num_targets = len(target_list)
    
    print(f"[INIT] Initializing RealSense Camera...")
    Realsensed435Cam = DepthCamera(resolution_width, resolution_height)
    
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Frame", 1280, 720)
    
    print("\n--- MULTI-TARGET AUTO DETECTION (Multi-Version Templates) ---")
    print(f"Targets: {', '.join(target_list)}")
    print("Press 'q' when targets are locked to extract 6-DOF.\n")

    best_matches = {}  # เก็บผลลัพธ์การ match ที่ดีที่สุด

    while True:
        ret , depth_raw_frame, color_raw_frame = Realsensed435Cam.get_raw_frame()
        if not ret: continue
        
        color_frame = np.asanyarray(color_raw_frame.get_data())
        gray_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
        display_frame = color_frame.copy()

        detected_pixels = []
        best_matches.clear()

        # --- สแกนหาเป้าหมายทั้งหมด (ทดสอบ versions ทั้งหมด) ---
        for target_idx, target_name in enumerate(target_list):
            versions = templates_by_target[target_name]
            
            best_match_val = -1
            best_match_loc = None
            best_match_offset = None
            best_match_version = None
            best_match_template = None
            
            # ทดสอบ template versions ทั้งหมด
            print(f"\n[SCAN] Testing {target_name}...")
            for template_img, offset, version_name in versions:
                try:
                    th, tw = template_img.shape
                    res = cv2.matchTemplate(gray_frame, template_img, cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                    
                    print(f"  {version_name}: {max_val*100:.1f}%", end="")
                    
                    if max_val > best_match_val:
                        best_match_val = max_val
                        best_match_loc = max_loc
                        best_match_offset = offset
                        best_match_version = version_name
                        best_match_template = template_img
                        print(" ← BEST")
                    else:
                        print()
                        
                except Exception as e:
                    print(f"  {version_name}: ERROR - {e}")
                    continue
            
            # ถ้าหา match ที่ดีพอ ให้ใช้
            if best_match_val >= 0.75:  # Stricter threshold for better accuracy
                top_left = best_match_loc
                th, tw = best_match_template.shape
                bottom_right = (top_left[0] + tw, top_left[1] + th)
                
                target_center = (top_left[0] + best_match_offset[0], top_left[1] + best_match_offset[1])
                detected_pixels.append((target_idx, target_center))
                best_matches[target_name] = {
                    'location': top_left,
                    'center': target_center,
                    'confidence': best_match_val,
                    'version': best_match_version,
                    'bounds': (top_left, bottom_right)
                }
                
                color = COLORS[target_idx % len(COLORS)]
                cv2.rectangle(display_frame, top_left, bottom_right, color, 2)
                cv2.circle(display_frame, target_center, 5, (0, 0, 255), -1)
                
                cv2.putText(display_frame, f"{target_name}: {best_match_val*100:.1f}% ({best_match_version})", 
                            (top_left[0], top_left[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                print(f"  ✓ Matched {target_name} using {best_match_version} (confidence: {best_match_val*100:.1f}%)")
            else:
                print(f"  ✗ No good match for {target_name} (best: {best_match_val*100:.1f}%)")

        cv2.imshow("Frame", display_frame)
        key = cv2.waitKey(33) & 0xFF
        
        # --- 50 FRAME AVERAGING & FILTERING LOGIC ---
        if key == ord('q'):
            if len(detected_pixels) == 0:
                print("\n[WARNING] No targets locked. Cannot extract.")
                continue
                
            print(f"\n[PROCESSING] Capturing 70 frames for depth averaging...")
            
            # เตรียม Array สำหรับเก็บผลรวม
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
                
                # นำมาบวกกันเฉพาะพิกเซลที่มีค่าความลึก (> 0)
                mask = d_arr > 0
                depth_sum[mask] += d_arr[mask]
                valid_depth_count[mask] += 1
                color_sum += c_arr
                
                last_depth_frame = depth_cap
                frames_captured += 1
                
                if frames_captured % 14 == 0:
                    print(f"  -> Captured {frames_captured}/70 frames...")
                cv2.waitKey(10)

            print("[PROCESSING] Calculating average and generating Point Cloud...")
            
            # ป้องกันการหารด้วย 0
            valid_depth_count[valid_depth_count == 0] = 1
            
            # หาค่าเฉลี่ย (เป็นหน่วยมิลลิเมตร)
            avg_depth = depth_sum / valid_depth_count
            
            # Bilateral Filter แก้รอยเว้าบนระนาบเดียวกัน (improved parameters)
            avg_depth = np.float32(avg_depth)
            avg_depth = cv2.bilateralFilter(avg_depth, d=11, sigmaColor=20.0, sigmaSpace=20.0)
            
            avg_color = color_sum / 70.0
            
            # Depth Gradient Filter ตัดขอบพังผืด (stricter filtering)
            grad_x = cv2.Sobel(avg_depth, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(avg_depth, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = cv2.magnitude(grad_x, grad_y)
            
            GRADIENT_THRESHOLD_MM = 12.0  # Stricter edge filtering 
            flat_surface_mask = grad_mag < GRADIENT_THRESHOLD_MM
            
            # ดึงค่า intrinsics ของกล้อง
            intrinsics = last_depth_frame.profile.as_video_stream_profile().intrinsics
            
            # ดึงค่า Depth Scale จริงจากตัว Hardware กล้อง
            depth_scale = last_depth_frame.get_units()
            
            # แปลงค่าเฉลี่ย 2D กลับเป็น 3D Point Cloud
            u, v = np.meshgrid(np.arange(resolution_width), np.arange(resolution_height))
            u = u.flatten()
            v = v.flatten()
            z = avg_depth.flatten() * depth_scale
            
            # ตัดข้อมูลที่ไกลเกิน 0.55m (slightly adjusted for better coverage)
            max_depth_meters = 0.55 
            valid = (z > 0) & (z < max_depth_meters) & flat_surface_mask.flatten()
            
            u_valid = u[valid]
            v_valid = v[valid]
            z_valid = z[valid]
            
            # Pinhole Camera Model
            x = (u_valid - intrinsics.ppx) * z_valid / intrinsics.fx
            y = (v_valid - intrinsics.ppy) * z_valid / intrinsics.fy
            
            # สร้าง Array ข้อมูล 3D
            points = np.vstack((x, -y, -z_valid)).T
            
            # จัดการสี
            colors_valid = avg_color.reshape(-1, 3)[valid]
            colors_rgb = colors_valid[:, ::-1] / 255.0  
            
            # สร้าง Point Cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors_rgb)
            
            print("[PROCESSING] Cleaning Point Cloud Outliers...")
            pcd, ind = pcd.remove_radius_outlier(nb_points=30, radius=0.012)  # Better flying pixel removal
            pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=25, std_ratio=1.8)  # Better noise removal
            
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
            
            obj_points = np.asarray(pcd.points)
            all_markers = []
            
            print(f"\n[PROCESSING] Extracting 3D Data for {len(detected_pixels)} targets...")
            
            for target_idx, target_pixel in detected_pixels:
                u, v = target_pixel
                target_name = target_list[target_idx]
                version_used = best_matches[target_name]['version']
                confidence = best_matches[target_name]['confidence']
                
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

                print("\n" + "-"*50)
                print(f"[{target_name.upper()}] 6-DOF DATA:")
                print(f"Template Version Used: {version_used} (Confidence: {confidence*100:.1f}%)")
                print(f"Position (X, Y, Z): {exact_target_pos}")
                print(f"Orientation (R, P, Y): {roll:.2f}°, {pitch:.2f}°, {yaw:.2f}°")

                txt_path = os.path.join(SAVE_DIR, f"{target_name}_data.txt")
                with open(txt_path, "w") as f:
                    f.write(f"Target: {target_name}\n")
                    f.write(f"Template_Version: {version_used}\n")
                    f.write(f"Confidence: {confidence*100:.1f}%\n")
                    f.write(f"Position_X: {exact_target_pos[0]:.6f}\n")
                    f.write(f"Position_Y: {exact_target_pos[1]:.6f}\n")
                    f.write(f"Position_Z: {exact_target_pos[2]:.6f}\n")
                    f.write(f"Roll: {roll:.2f}\n")
                    f.write(f"Pitch: {pitch:.2f}\n")
                    f.write(f"Yaw: {yaw:.2f}\n")
                
                target_ball = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
                bgr_color = COLORS[target_idx % len(COLORS)]
                target_ball.paint_uniform_color([bgr_color[2]/255.0, bgr_color[1]/255.0, bgr_color[0]/255.0])
                target_ball.translate(exact_target_pos)
                
                axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.04, origin=[0,0,0])
                axis.rotate(rotation_matrix, center=[0,0,0])
                axis.translate(exact_target_pos)
                
                all_markers.extend([target_ball, axis])

            print("-"*50)
            print(f"[SUCCESS] All target files saved to: {os.path.abspath(SAVE_DIR)}")

            o3d.io.write_point_cloud(os.path.join(SAVE_DIR, "scene_object.ply"), pcd)

            mat_unlit = o3d.visualization.rendering.MaterialRecord()
            mat_unlit.shader = "defaultUnlit"
            
            geometries = [{"name": "pcd", "geometry": pcd, "material": mat_unlit}]
            for i, marker in enumerate(all_markers):
                geometries.append({"name": f"marker_{i}", "geometry": marker, "material": mat_unlit})
                
            o3d.visualization.draw(geometries, title="Multi-Target Detection (Multi-Version Templates)")
            break 

    Realsensed435Cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()