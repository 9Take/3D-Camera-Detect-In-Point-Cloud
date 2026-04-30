import numpy as np
import pyrealsense2 as rs
import cv2
import open3d as o3d
import os
import math
import sys
import argparse
from realsense_depth import DepthCamera
import copy
from scipy.spatial.transform import Rotation as R

# --- Configuration ---
resolution_width, resolution_height = (848, 480)

def get_config():
    # ... (ฟังก์ชัน get_config เหมือนเดิม) ...
    parser = argparse.ArgumentParser(description='Real Sense 3D Multi-Target Detection')
    parser.add_argument('-td', '--template-dir', type=str, default=None)
    parser.add_argument('-t', '--targets', type=str, nargs='+', default=None)
    parser.add_argument('-sd', '--save-dir', type=str, default=None)
    args = parser.parse_args()
    
    template_dir = args.template_dir or "realsensepy/test1"
    target_basenames = args.targets or ["target_B", "target_C"]
    
    cleaned_targets = [t.replace('_template.png', '').replace('_offset.txt', '').replace('.png', '').strip() for t in target_basenames if t]
    target_basenames = cleaned_targets
    
    save_dir = args.save_dir or "test1/test1_template_match"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    return template_dir, target_basenames, save_dir

TEMPLATE_DIR, TARGET_BASENAMES, SAVE_DIR = get_config()
COLORS = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]

def rotation_matrix_to_euler_angles(R_mat):
    sy = math.sqrt(R_mat[0, 0] * R_mat[0, 0] + R_mat[1, 0] * R_mat[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R_mat[2, 1], R_mat[2, 2])
        y = math.atan2(-R_mat[2, 0], sy)
        z = math.atan2(R_mat[1, 0], R_mat[0, 0])
    else:
        x = math.atan2(-R_mat[1, 2], R_mat[1, 1])
        y = math.atan2(-R_mat[2, 0], sy)
        z = 0
    return np.rad2deg([x, y, z])

def create_camera_transform_matrix(tx, ty, tz, rx, ry, rz):
    rot = R.from_euler('xyz', [rx, ry, rz], degrees=True)
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rot.as_matrix()
    transformation_matrix[:3, 3] = [tx, ty, tz]
    return transformation_matrix

def align_with_camera_pose(source_pcd, target_pcd, init_transform_matrix, voxel_size=0.01):
    source_pcd_aligned = copy.deepcopy(source_pcd)
    source_pcd_aligned.transform(init_transform_matrix)
    
    target_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    source_pcd_aligned.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    
    icp_result = o3d.pipelines.registration.registration_icp(
        source_pcd_aligned, target_pcd, voxel_size * 0.5, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )
    source_pcd_aligned.transform(icp_result.transformation)
    return source_pcd_aligned

def main():
    templates = []
    offsets = []
    
    # 1. โหลด Template
    for base in TARGET_BASENAMES:
        img_path = os.path.join(TEMPLATE_DIR, f"{base}_template.png")
        txt_path = os.path.join(TEMPLATE_DIR, f"{base}_offset.txt")
        if not os.path.exists(img_path) or not os.path.exists(txt_path):
            print(f"[ERROR] Missing files for {base}")
            sys.exit(1)
        templates.append(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE))
        with open(txt_path, 'r') as f:
            data = f.read().strip().split(',')
            offsets.append((int(data[0]), int(data[1])))

    # 2. เปิดกล้อง
    print("[INIT] Initializing RealSense Camera...")
    Realsensed435Cam = DepthCamera(resolution_width, resolution_height)
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Frame", 848, 480)
    
    print("\n--- 3-VIEW SCANNING MODE ---")
    print("Press '1' -> Capture Center (Main View - Must capture first)")
    print("Press '2' -> Capture Left View")
    print("Press '3' -> Capture Right View")
    print("Press 'q' -> Merge Point Clouds & Extract Targets")

    captured_views = {}
    center_detected_pixels = [] # เก็บพิกัด 2D จากมุมมองตรงกลางเพื่อใช้อ้างอิง
    last_intrinsics = None
    
    # ตั้งค่าตำแหน่งกล้อง (แก้ตัวเลขให้ตรงกับระยะจริงของหุ่นยนต์คุณ)
    camera_pose_left = create_camera_transform_matrix(tx=-0.1, ty=0.0, tz=0.0, rx=0, ry=15, rz=0)
    camera_pose_right = create_camera_transform_matrix(tx=0.1, ty=0.0, tz=0.0, rx=0, ry=-15, rz=0)

    while True:
        ret , depth_raw_frame, color_raw_frame = Realsensed435Cam.get_raw_frame()
        if not ret: continue
        
        color_frame = np.asanyarray(color_raw_frame.get_data())
        gray_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
        display_frame = color_frame.copy()

        current_detected_pixels = [] 

        # สแกนหาเป้าหมายโชว์บนจอเรียลไทม์
        for idx, template_patch in enumerate(templates):
            th, tw = template_patch.shape
            res = cv2.matchTemplate(gray_frame, template_patch, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
            if max_val >= 0.70:
                top_left = max_loc
                bottom_right = (top_left[0] + tw, top_left[1] + th)
                target_center = (top_left[0] + offsets[idx][0], top_left[1] + offsets[idx][1])
                current_detected_pixels.append((idx, target_center)) 
                
                color = COLORS[idx % len(COLORS)]
                cv2.rectangle(display_frame, top_left, bottom_right, color, 2)
                cv2.circle(display_frame, target_center, 5, (0, 0, 255), -1)
                cv2.putText(display_frame, f"{TARGET_BASENAMES[idx]}: {max_val*100:.1f}%", 
                            (top_left[0], top_left[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Frame", display_frame)
        key = cv2.waitKey(33) & 0xFF
        
        # --- เริ่มระบบถ่ายภาพ 50 เฟรม ---
        if key in [ord('1'), ord('2'), ord('3')]:
            view_name = "CENTER" if key == ord('1') else "LEFT" if key == ord('2') else "RIGHT"
            
            # ถ้าเป็น Center ให้ล็อกค่าพิกัดเป้าหมาย 2D เก็บไว้
            if view_name == "CENTER":
                if not current_detected_pixels:
                    print("\n[WARNING] No targets detected in Center view! Cannot lock targets.")
                    continue
                center_detected_pixels = copy.deepcopy(current_detected_pixels)
                
            print(f"\n[CAPTURING {view_name}] Averaging 50 frames...")
            
            depth_sum = np.zeros((resolution_height, resolution_width), dtype=np.float32)
            color_sum = np.zeros((resolution_height, resolution_width, 3), dtype=np.float32)
            valid_depth_count = np.zeros((resolution_height, resolution_width), dtype=np.float32)
            
            frames_captured = 0
            last_depth_frame = None

            while frames_captured < 50:
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
                if frames_captured % 10 == 0: print(f"  -> {frames_captured}/50 frames...")
                cv2.waitKey(10)

            # คำนวณค่าเฉลี่ยและฟิลเตอร์
            valid_depth_count[valid_depth_count == 0] = 1
            avg_depth = np.float32(depth_sum / valid_depth_count)
            avg_depth = cv2.bilateralFilter(avg_depth, d=9, sigmaColor=15.0, sigmaSpace=15.0)
            avg_color = color_sum / 50.0
            
            grad_x = cv2.Sobel(avg_depth, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(avg_depth, cv2.CV_32F, 0, 1, ksize=3)
            flat_surface_mask = cv2.magnitude(grad_x, grad_y) < 15.0
            
            # แปลงเป็น Point Cloud แบบ Pinhole Model
            intrinsics = last_depth_frame.profile.as_video_stream_profile().intrinsics
            if view_name == "CENTER": last_intrinsics = intrinsics
            depth_scale = last_depth_frame.get_units()
            
            u, v = np.meshgrid(np.arange(resolution_width), np.arange(resolution_height))
            z = avg_depth.flatten() * depth_scale
            
            valid = (z > 0) & (z < 0.50) & flat_surface_mask.flatten()
            u_valid, v_valid, z_valid = u.flatten()[valid], v.flatten()[valid], z[valid]
            
            x = (u_valid - intrinsics.ppx) * z_valid / intrinsics.fx
            y = (v_valid - intrinsics.ppy) * z_valid / intrinsics.fy
            
            points = np.vstack((x, -y, -z_valid)).T
            colors_rgb = avg_color.reshape(-1, 3)[valid][:, ::-1] / 255.0  
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors_rgb)
            
            # ลบ Noise ของแต่ละมุม
            pcd, _ = pcd.remove_radius_outlier(nb_points=25, radius=0.015)
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            
            captured_views[view_name] = pcd
            print(f"[SUCCESS] {view_name} view point cloud generated.")

        # --- รวม Point Cloud และหา 6-DOF ---
        elif key == ord('q'):
            if "CENTER" not in captured_views:
                print("\n[WARNING] You MUST capture the '1' (CENTER) view first!")
                continue
            
            print("\n[PROCESSING] Merging Views...")
            merged_pcd = copy.deepcopy(captured_views["CENTER"])
            
            if "LEFT" in captured_views:
                print("    -> Aligning LEFT to CENTER...")
                aligned_left = align_with_camera_pose(captured_views["LEFT"], captured_views["CENTER"], camera_pose_left)
                merged_pcd += aligned_left
                
            if "RIGHT" in captured_views:
                print("    -> Aligning RIGHT to CENTER...")
                aligned_right = align_with_camera_pose(captured_views["RIGHT"], captured_views["CENTER"], camera_pose_right)
                merged_pcd += aligned_right
                
            # Downsample ครั้งสุดท้ายหลังรวมกัน
            final_pcd = merged_pcd.voxel_down_sample(voxel_size=0.002)
            final_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
            
            print(f"\n[PROCESSING] Extracting 3D Data for targets...")
            obj_points = np.asarray(final_pcd.points)
            all_markers = []
            
            # ใช้พิกัดที่ล็อกไว้ตอนกด '1' (Center) เพื่อหา 6-DOF บนโมเดลที่รวมกันแล้ว
            for target_id, target_pixel in center_detected_pixels:
                u_target, v_target = target_pixel
                target_name = TARGET_BASENAMES[target_id]
                
                raw_x, raw_y, raw_z = obj_points[:, 0], -obj_points[:, 1], -obj_points[:, 2]
                valid_z = raw_z != 0 
                u_proj, v_proj = np.zeros_like(raw_x), np.zeros_like(raw_y)
                
                u_proj[valid_z] = (raw_x[valid_z] * last_intrinsics.fx / raw_z[valid_z]) + last_intrinsics.ppx
                v_proj[valid_z] = (raw_y[valid_z] * last_intrinsics.fy / raw_z[valid_z]) + last_intrinsics.ppy
                
                dist_sq = (u_proj - u_target)**2 + (v_proj - v_target)**2
                best_idx = np.argmin(dist_sq)
                exact_target_pos = obj_points[best_idx]
                
                normal = np.asarray(final_pcd.normals)[best_idx]
                z_axis = normal / np.linalg.norm(normal)
                x_axis = np.array([1, 0, 0]) if abs(z_axis[0]) < 0.9 else np.array([0, 1, 0])
                y_axis = np.cross(z_axis, x_axis)
                y_axis /= np.linalg.norm(y_axis)
                x_axis = np.cross(y_axis, z_axis)
                
                rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
                roll, pitch, yaw = rotation_matrix_to_euler_angles(rotation_matrix)

                print(f"[{target_name.upper()}] Pos: {exact_target_pos} | RPY: {roll:.2f}, {pitch:.2f}, {yaw:.2f}")

                with open(os.path.join(SAVE_DIR, f"{target_name}_data.txt"), "w") as f:
                    f.write(f"Position_X: {exact_target_pos[0]:.6f}\nPosition_Y: {exact_target_pos[1]:.6f}\nPosition_Z: {exact_target_pos[2]:.6f}\nRoll: {roll:.2f}\nPitch: {pitch:.2f}\nYaw: {yaw:.2f}\n")
                
                target_ball = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
                bgr_color = COLORS[target_id % len(COLORS)]
                target_ball.paint_uniform_color([bgr_color[2]/255.0, bgr_color[1]/255.0, bgr_color[0]/255.0])
                target_ball.translate(exact_target_pos)
                
                axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.04, origin=[0,0,0])
                axis.rotate(rotation_matrix, center=[0,0,0])
                axis.translate(exact_target_pos)
                all_markers.extend([target_ball, axis])

            o3d.io.write_point_cloud(os.path.join(SAVE_DIR, "3view_merged_scene.ply"), final_pcd)
            
            geometries = [{"name": "merged_pcd", "geometry": final_pcd}]
            for i, marker in enumerate(all_markers): geometries.append({"name": f"marker_{i}", "geometry": marker})
            o3d.visualization.draw(geometries, title="3-View Mapped Point Cloud (Base+Left+Right)")
            break 

    Realsensed435Cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()