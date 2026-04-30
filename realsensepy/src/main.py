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

# --- Configuration ---
resolution_width, resolution_height = (848, 480)

def get_config():
    """รับค่า configuration จาก command-line arguments หรือ user input"""
    parser = argparse.ArgumentParser(
        description='Real Sense 3D Multi-Target Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python main.py --template-dir realsensepy/test1 --targets target_A target_B --save-dir my_output
  python main.py -td realsensepy/test1 -t target_A target_B -sd my_output
  python main.py  (จะถามผู้ใช้ input)
        '''
    )
    
    parser.add_argument(
        '-td', '--template-dir',
        type=str,
        default=None,
        help='โฟลเดอร์ที่เก็บ Template และ Offset (default: realsensepy/test1)'
    )
    parser.add_argument(
        '-t', '--targets',
        type=str,
        nargs='+',
        default=None,
        help='ชื่อเป้าหมาย เช่น target_A target_B (default: target_A target_B)'
    )
    parser.add_argument(
        '-sd', '--save-dir',
        type=str,
        default=None,
        help='โฟลเดอร์สำหรับบันทึกผลลัพธ์ (default: test1/test1_template_match)'
    )
    
    args = parser.parse_args()
    
    # ถ้าไม่มี argument ให้ถามผู้ใช้
    template_dir = args.template_dir
    if template_dir is None:
        template_dir = input("กรุณาป้อนโฟลเดอร์ที่เก็บ Template [realsensepy/test1]: ").strip()
        if not template_dir:
            template_dir = "realsensepy/test1"
    
    target_basenames = args.targets
    if target_basenames is None:
        targets_input = input("กรุณาป้อนชื่อเป้าหมาย คั่นด้วยช่องว่าง [target_A target_B]: ").strip()
        if targets_input:
            target_basenames = targets_input.split()
        else:
            target_basenames = ["target_A", "target_B", "target_C"]
    
    # ทำความสะอาด target names
    cleaned_targets = []
    for target in target_basenames:
        target = target.replace('_template.png', '').replace('_offset.txt', '').replace('.png', '').strip()
        if target:
            cleaned_targets.append(target)
    target_basenames = cleaned_targets
    
    if not target_basenames:
        print("[ERROR] No valid target names provided.")
        sys.exit(1)
    
    save_dir = args.save_dir
    if save_dir is None:
        save_dir = input("กรุณาป้อนโฟลเดอร์สำหรับบันทึกผลลัพธ์ [test1/test1_template_match]: ").strip()
        if not save_dir:
            save_dir = "test1/test1_template_match"
    
    # สร้าง folder ถ้ายังไม่มี
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"✓ สร้าง folder '{save_dir}' เรียบร้อย")
    
    print(f"\n--- ตั้งค่า Configuration ---")
    print(f"Template Directory: {template_dir}")
    print(f"Target Names: {', '.join(target_basenames)}")
    print(f"Save Directory: {save_dir}\n")
    
    return template_dir, target_basenames, save_dir

TEMPLATE_DIR, TARGET_BASENAMES, SAVE_DIR = get_config()
COLORS = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)] # เพิ่มสีเผื่อมีหลายเป้าหมาย

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
    templates = []
    offsets = []
    
    # 1. โหลดรูป Template และไฟล์ Offset อัตโนมัติ
    print(f"\n[LOADING] Looking for templates in: {os.path.abspath(TEMPLATE_DIR)}")
    for base in TARGET_BASENAMES:
        img_path = os.path.join(TEMPLATE_DIR, f"{base}_template.png")
        txt_path = os.path.join(TEMPLATE_DIR, f"{base}_offset.txt")
        
        if not os.path.exists(img_path) or not os.path.exists(txt_path):
            print(f"    [ERROR] Template or Offset file NOT found for {base}")
            sys.exit(1)
            
        tpl = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        templates.append(tpl)
        
        with open(txt_path, 'r') as f:
            data = f.read().strip().split(',')
            offsets.append((int(data[0]), int(data[1])))
            
    print(f"\n[SUCCESS] Loaded {len(templates)} templates with custom target points.\n")

    # 2. เปิดกล้อง
    print("[INIT] Initializing RealSense Camera...")
    Realsensed435Cam = DepthCamera(resolution_width, resolution_height)
    
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Frame", 848, 480)
    
    print("\n--- MULTI-TARGET AUTO DETECTION ---")
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
                
                target_offset = offsets[idx]
                target_center = (top_left[0] + target_offset[0], top_left[1] + target_offset[1])
                detected_pixels.append((idx, target_center)) 
                
                color = COLORS[idx % len(COLORS)]
                cv2.rectangle(display_frame, top_left, bottom_right, color, 2)
                cv2.circle(display_frame, target_center, 5, (0, 0, 255), -1)
                
                target_name = TARGET_BASENAMES[idx]
                cv2.putText(display_frame, f"{target_name}: {max_val*100:.1f}%", 
                            (top_left[0], top_left[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Frame", display_frame)
        key = cv2.waitKey(33) & 0xFF
        
        # --- 100 FRAME AVERAGING & FILTERING LOGIC ---
        if key == ord('q'):
            if len(detected_pixels) == 0:
                print("\n[WARNING] No targets locked. Cannot extract.")
                continue
                
            print(f"\n[PROCESSING] Capturing 100 frames for depth averaging...")
            
            # เตรียม Array สำหรับเก็บผลรวม
            depth_sum = np.zeros((resolution_height, resolution_width), dtype=np.float32)
            color_sum = np.zeros((resolution_height, resolution_width, 3), dtype=np.float32)
            valid_depth_count = np.zeros((resolution_height, resolution_width), dtype=np.float32)
            
            frames_captured = 0
            last_depth_frame = None

            while frames_captured < 100:
                ret_cap, depth_cap, color_cap = Realsensed435Cam.get_raw_frame()
                if not ret_cap: continue
                
                d_arr = np.asanyarray(depth_cap.get_data(), dtype=np.float32)
                c_arr = np.asanyarray(color_cap.get_data(), dtype=np.float32)
                
                # นำมาบวกกันเฉพาะพิกเซลที่มีค่าความลึก (> 0) เพื่อไม่ให้ค่า 0 มาดึงค่าเฉลี่ยให้เพี้ยน
                mask = d_arr > 0
                depth_sum[mask] += d_arr[mask]
                valid_depth_count[mask] += 1
                color_sum += c_arr
                
                last_depth_frame = depth_cap
                frames_captured += 1
                
                if frames_captured % 20 == 0:
                    print(f"  -> Captured {frames_captured}/100 frames...")
                cv2.waitKey(10)

            print("[PROCESSING] Calculating average and generating Point Cloud...")
            
            print("[PROCESSING] Calculating average and generating Point Cloud...")
            
            # ป้องกันการหารด้วย 0
            valid_depth_count[valid_depth_count == 0] = 1
            
            # หาค่าเฉลี่ย (เป็นหน่วยมิลลิเมตร)
            avg_depth = depth_sum / valid_depth_count
            
            # --- [เพิ่มใหม่] Bilateral Filter แก้รอยเว้าบนระนาบเดียวกัน ---
            # แปลงให้อยู่ในฟอร์แมต float32 ที่ OpenCV รองรับ
            avg_depth = np.float32(avg_depth)
            
            # รีดพื้นผิวให้เรียบเนียน (d=ขนาดพื้นที่เกลี่ย, sigmaColor=ความต่างระยะ Z ที่ยอมให้เกลี่ยเข้าหากัน)
            # ตั้งค่า sigmaColor=15 แปลว่าถ้าระยะเว้าแหว่งไม่เกิน 15 มิลลิเมตร โค้ดจะถมให้เรียบเสมอกัน
            avg_depth = cv2.bilateralFilter(avg_depth, d=9, sigmaColor=15.0, sigmaSpace=15.0)
            
            avg_color = color_sum / 100.0
            
            # --- Depth Gradient Filter ตัดขอบพังผืด (ของเดิม) ---
            grad_x = cv2.Sobel(avg_depth, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(avg_depth, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = cv2.magnitude(grad_x, grad_y)
            
            GRADIENT_THRESHOLD_MM = 15.0 
            flat_surface_mask = grad_mag < GRADIENT_THRESHOLD_MM
            
            # ดึงค่า intrinsics ของกล้อง
            intrinsics = last_depth_frame.profile.as_video_stream_profile().intrinsics
            
            # ดึงค่า Depth Scale จริงจากตัว Hardware กล้อง
            depth_scale = last_depth_frame.get_units()
            
            # --- แปลงค่าเฉลี่ย 2D กลับเป็น 3D Point Cloud ---
            u, v = np.meshgrid(np.arange(resolution_width), np.arange(resolution_height))
            u = u.flatten()
            v = v.flatten()
            z = avg_depth.flatten() * depth_scale  # แปลงเป็นหน่วยเมตรอย่างแม่นยำ
            
            # ตัดข้อมูลที่ไกลเกิน 0.5m + เอาเฉพาะจุดที่ความชันผ่านเกณฑ์ (flat_surface_mask)
            max_depth_meters = 0.50 
            valid = (z > 0) & (z < max_depth_meters) & flat_surface_mask.flatten()
            
            u_valid = u[valid]
            v_valid = v[valid]
            z_valid = z[valid]
            
            # สมการ Pinhole Camera Model
            x = (u_valid - intrinsics.ppx) * z_valid / intrinsics.fx
            y = (v_valid - intrinsics.ppy) * z_valid / intrinsics.fy
            
            # สร้าง Array ข้อมูล 3D (พลิกแกน Y และ Z ให้ตรงกับมาตรฐานการแสดงผลของ Open3D)
            points = np.vstack((x, -y, -z_valid)).T
            
            # จัดการสี (OpenCV เป็น BGR, ต้องสลับเป็น RGB ให้ Open3D)
            colors_valid = avg_color.reshape(-1, 3)[valid]
            colors_rgb = colors_valid[:, ::-1] / 255.0  
            
            # สร้าง Point Cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors_rgb)
            
            print("[PROCESSING] Cleaning Point Cloud Outliers...")
            # 1. ลบ Flying Pixels (พังผืดที่ขอบ) ด้วย Radius Outlier Removal
            pcd, ind = pcd.remove_radius_outlier(nb_points=25, radius=0.015) 
            
            # 2. ลบ Noise ผิวชิ้นงานด้วย Statistical Outlier Removal
            pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
            
            obj_points = np.asarray(pcd.points)
            all_markers = []
            
            print(f"\n[PROCESSING] Extracting 3D Data for {len(detected_pixels)} targets...")
            
            for target_id, target_pixel in detected_pixels:
                u, v = target_pixel
                target_name = TARGET_BASENAMES[target_id]
                
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
                print(f"[{target_name.upper()}] 6-DOF DATA:")
                print(f"Position (X, Y, Z): {exact_target_pos}")
                print(f"Orientation (R, P, Y): {roll:.2f}, {pitch:.2f}, {yaw:.2f}")

                txt_path = os.path.join(SAVE_DIR, f"{target_name}_data.txt")
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
                
            o3d.visualization.draw(geometries, title="Multi-Target Production Mode (100 Frames Averaged & Filtered)")
            break 

    Realsensed435Cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()