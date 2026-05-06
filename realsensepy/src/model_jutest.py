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
#resolution_width, resolution_height = (1280, 720)  # Balanced: Better quality without excessive slowdown
resolution_width, resolution_height = (848, 480)  # Faster processing with slightly lower resolution

def get_config():
    """รับค่า configuration จาก command-line arguments หรือ user input"""
    parser = argparse.ArgumentParser(
        description='Real Sense 3D Multi-Target Detection with Multi-Version Templates (Homography)',
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
        default="ju",
        help='โฟลเดอร์ที่เก็บ Template และ Offset (default: ju)'
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
        help='โฟลเดอร์สำหรับบันทึกผลลัพธ์ (default: ju_output)'
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

def load_all_template_versions(template_dir, sift_detector):
    """
    Scan folder for all template PNG files and extract SIFT features.
    Returns: dict[target_name] -> list of (template_image, offset_tuple, version_name, kp, des)
    """
    templates_by_target = defaultdict(list)
    files = os.listdir(template_dir)
    template_files = defaultdict(list)
    
    for filename in files:
        if filename.endswith('_template.png'):
            base_name = filename.replace('_template.png', '')
            template_files[base_name].append(filename)
    
    if not template_files:
        print(f"[ERROR] No template PNG files found in {template_dir}")
        sys.exit(1)
    
    for base_name in sorted(template_files.keys()):
        template_file = os.path.join(template_dir, f"{base_name}_template.png")
        offset_file = os.path.join(template_dir, f"{base_name}_offset.txt")
        
        if not os.path.exists(template_file) or not os.path.exists(offset_file):
            continue
        
        try:
            with open(offset_file, 'r') as f:
                data = f.read().strip().split(',')
                offset = (int(data[0]), int(data[1]))
        except Exception as e:
            print(f"[ERROR] Failed to read offset from {offset_file}: {e}")
            continue
        
        try:
            template = cv2.imread(template_file, cv2.IMREAD_GRAYSCALE)
            if template is None or len(template.shape) != 2:
                continue
            
            # --- Extract SIFT Features for the Template ---
            kp, des = sift_detector.detectAndCompute(template, None)
            if des is None or len(des) < 5:
                print(f"[WARNING] Not enough SIFT features in {base_name}_template.png. Skipping.")
                continue
                
        except Exception as e:
            print(f"[ERROR] Failed to load {base_name}_template.png: {e}")
            continue
        
        target_letter = base_name.split('.')[0]
        templates_by_target[target_letter].append((template, offset, base_name, kp, des))
        print(f"    ✓ Loaded {base_name} (Features: {len(kp)})")
    
    if not templates_by_target:
        print("[ERROR] No templates loaded successfully!")
        sys.exit(1)
    
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
    # 1. Initialize SIFT Detector
    sift = cv2.SIFT_create()
    
    # 2. Load templates and pre-compute their features
    templates_by_target = load_all_template_versions(TEMPLATE_DIR, sift)
    target_list = sorted(templates_by_target.keys())
    
    # Setup FLANN Matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    print(f"[INIT] Initializing RealSense Camera...")
    Realsensed435Cam = DepthCamera(resolution_width, resolution_height)
    
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Frame", 1280, 720)
    
    print("\n--- MULTI-TARGET SIFT HOMOGRAPHY DETECTION ---")
    print(f"Targets: {', '.join(target_list)}")
    print("Press 'q' when targets are locked to extract 6-DOF.\n")

    best_matches = {}

    while True:
        ret , depth_raw_frame, color_raw_frame = Realsensed435Cam.get_raw_frame()
        if not ret: continue
        
        color_frame = np.asanyarray(color_raw_frame.get_data())
        gray_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
        display_frame = color_frame.copy()

        detected_pixels = []
        best_matches.clear()
        
        # --- Extract features from the current live frame ---
        kp_frame, des_frame = sift.detectAndCompute(gray_frame, None)

        if des_frame is not None and len(des_frame) > 2:
            # --- Scan all targets ---
            for target_idx, target_name in enumerate(target_list):
                versions = templates_by_target[target_name]
                
                best_inliers = 0
                best_match_data = None
                
                print(f"\n[SCAN] Testing {target_name}...")
                
                # Test all versions of the current target
                for template_img, offset, version_name, kp_temp, des_temp in versions:
                    try:
                        matches = flann.knnMatch(des_temp, des_frame, k=2)
                        
                        good_matches = []
                        for match_pair in matches:
                            if len(match_pair) == 2:
                                m, n = match_pair
                                if m.distance < 0.7 * n.distance:
                                    good_matches.append(m)
                        
                        MIN_MATCH_COUNT = 10
                        if len(good_matches) > MIN_MATCH_COUNT:
                            src_pts = np.float32([kp_temp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                            
                            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                            
                            if M is not None:
                                inliers = int(np.sum(mask))
                                print(f"  {version_name}: {inliers} inliers", end="")
                                
                                if inliers > best_inliers:
                                    best_inliers = inliers
                                    best_match_data = {
                                        'M': M,
                                        'offset': offset,
                                        'version': version_name,
                                        'template': template_img,
                                        'inliers': inliers
                                    }
                                    print(" ← BEST")
                                else:
                                    print()
                            else:
                                print(f"  {version_name}: Homography failed")
                        else:
                            print(f"  {version_name}: Not enough matches ({len(good_matches)}/{MIN_MATCH_COUNT})")
                            
                    except Exception as e:
                        print(f"  {version_name}: ERROR - {e}")
                        continue
                
                # If a valid homography was found with enough inliers
                if best_inliers >= 12: # Minimum inliers required to trust the lock
                    M = best_match_data['M']
                    th, tw = best_match_data['template'].shape
                    target_offset = best_match_data['offset']
                    
                    # 1. Transform the bounding box
                    pts = np.float32([[0, 0], [0, th - 1], [tw - 1, th - 1], [tw - 1, 0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, M)
                    
                    # 2. Transform the custom target offset
                    target_pt_src = np.float32([[[target_offset[0], target_offset[1]]]])
                    target_pt_dst = cv2.perspectiveTransform(target_pt_src, M)
                    target_pixel = (int(target_pt_dst[0][0][0]), int(target_pt_dst[0][0][1]))
                    
                    # 3. Ensure target is within screen bounds
                    if 0 <= target_pixel[0] < resolution_width and 0 <= target_pixel[1] < resolution_height:
                        detected_pixels.append((target_idx, target_pixel))
                        best_matches[target_name] = {
                            'version': best_match_data['version'],
                            'inliers': best_match_data['inliers']
                        }
                        
                        color = COLORS[target_idx % len(COLORS)]
                        display_frame = cv2.polylines(display_frame, [np.int32(dst)], True, color, 3, cv2.LINE_AA)
                        cv2.circle(display_frame, target_pixel, 5, (0, 0, 255), -1)
                        
                        cv2.putText(display_frame, f"{target_name}: {best_inliers} Inliers ({best_match_data['version']})", 
                                    (int(dst[0][0][0]), int(dst[0][0][1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        print(f"  ✓ Matched {target_name} using {best_match_data['version']} (Inliers: {best_inliers})")
                    else:
                        print(f"  ✗ Target out of frame bounds")
                else:
                    print(f"  ✗ No reliable match for {target_name} (best inliers: {best_inliers})")

        cv2.imshow("Frame", display_frame)
        key = cv2.waitKey(33) & 0xFF
        
        # --- 50 FRAME AVERAGING & 3D EXTRACTION LOGIC ---
        if key == ord('q'):
            if len(detected_pixels) == 0:
                print("\n[WARNING] No targets locked. Cannot extract.")
                continue
                
            print(f"\n[PROCESSING] Capturing 70 frames for depth averaging...")
            
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
                
                if frames_captured % 14 == 0:
                    print(f"  -> Captured {frames_captured}/70 frames...")
                cv2.waitKey(10)

            print("[PROCESSING] Calculating average and generating Point Cloud...")
            
            valid_depth_count[valid_depth_count == 0] = 1
            avg_depth = depth_sum / valid_depth_count
            
            # ลด Noise นิดหน่อย แต่ไม่ทำลายพื้นผิว
            avg_depth = np.float32(avg_depth)
            avg_depth = cv2.bilateralFilter(avg_depth, d=5, sigmaColor=10.0, sigmaSpace=10.0)
            
            # 1. แปลง Array กลับเป็นฟอร์แมตที่ Open3D อ่านได้ตรงๆ
            avg_depth_uint16 = np.clip(avg_depth, 0, 65535).astype(np.uint16)
            avg_color_uint8 = np.clip(color_sum / 70.0, 0, 255).astype(np.uint8)
            
            o3d_color = o3d.geometry.Image(cv2.cvtColor(avg_color_uint8, cv2.COLOR_BGR2RGB))
            o3d_depth = o3d.geometry.Image(avg_depth_uint16)
            
            intrinsics = last_depth_frame.profile.as_video_stream_profile().intrinsics
            depth_scale = last_depth_frame.get_units()
            
            # 2. ใช้ Native Function ของ Open3D สร้าง Point Cloud (เสถียรกว่ามาก)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d_color,
                o3d_depth,
                depth_scale=1.0 / depth_scale, # ปกติคือ 1000.0 (แปลงหน่วยมิลเป็นเมตร)
                depth_trunc=1.5,               # มองเห็นลึกได้ถึง 1.5 เมตร (แก้ปัญหาเป้าหาย)
                convert_rgb_to_intensity=False
            )
            
            o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
                resolution_width, resolution_height,
                intrinsics.fx, intrinsics.fy,
                intrinsics.ppx, intrinsics.ppy
            )
            
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d_intrinsics)
            
            # กลับแกน Y, Z ให้สอดคล้องกับระบบของกล้อง RealSense
            pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            
            print("[PROCESSING] Cleaning Point Cloud Outliers...")
            pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
            
            all_markers = []
            
            print(f"\n[PROCESSING] Extracting 3D Data for {len(detected_pixels)} targets...")
            
            # สร้าง KDTree เพื่อค้นหาจุดบน Point Cloud ที่เร็วที่สุด
            pcd_tree = o3d.geometry.KDTreeFlann(pcd)
            
            for target_idx, target_pixel in detected_pixels:
                u, v = target_pixel
                target_name = target_list[target_idx]
                version_used = best_matches[target_name]['version']
                inliers = best_matches[target_name]['inliers']
                
                # 3. ใช้สมการคณิตศาสตร์หาพิกัด 3D ตรงๆ จากแกน 2D
                target_z_raw = avg_depth[v, u]
                
                if target_z_raw <= 0 or (target_z_raw * depth_scale) > 1.5:
                    print(f"[ERROR] [{target_name.upper()}] Not found in depth map or too far (>1.5m).")
                    continue
                    
                target_z = target_z_raw * depth_scale
                target_x = (u - intrinsics.ppx) * target_z / intrinsics.fx
                target_y = (v - intrinsics.ppy) * target_z / intrinsics.fy
                
                exact_target_pos = np.array([target_x, -target_y, -target_z])
                
                # ค้นหา Vector Normal จากจุดที่ใกล้ที่สุด
                [k, idx, _] = pcd_tree.search_knn_vector_3d(exact_target_pos, 1)
                
                if k == 0:
                    print(f"[ERROR] [{target_name.upper()}] Could not match target to Point Cloud.")
                    continue
                    
                normal = np.asarray(pcd.normals)[idx[0]]
                
                # คำนวณแกน R, P, Y
                z_axis = normal / np.linalg.norm(normal)
                x_axis = np.array([1, 0, 0]) if abs(z_axis[0]) < 0.9 else np.array([0, 1, 0])
                y_axis = np.cross(z_axis, x_axis)
                y_axis /= np.linalg.norm(y_axis)
                x_axis = np.cross(y_axis, z_axis)
                
                rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
                roll, pitch, yaw = rotation_matrix_to_euler_angles(rotation_matrix)

                print("\n" + "-"*50)
                print(f"[{target_name.upper()}] 6-DOF DATA:")
                print(f"Template Version Used: {version_used} (Inliers: {inliers})")
                print(f"Position (X, Y, Z): {exact_target_pos}")
                print(f"Orientation (R, P, Y): {roll:.2f}°, {pitch:.2f}°, {yaw:.2f}°")

                txt_path = os.path.join(SAVE_DIR, f"{target_name}_data.txt")
                with open(txt_path, "w") as f:
                    f.write(f"Target: {target_name}\n")
                    f.write(f"Template_Version: {version_used}\n")
                    f.write(f"Inliers: {inliers}\n")
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
                
            o3d.visualization.draw(geometries, title="Multi-Target SIFT Homography")
            break 

    Realsensed435Cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()