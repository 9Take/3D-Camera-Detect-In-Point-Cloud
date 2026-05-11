import numpy as np
import cv2
import open3d as o3d
import os
import math
import argparse
import yaml
import sys
import json
import pyrealsense2 as rs  # Added for hardware-accelerated filtering

# ให้ Python มองเห็นโฟลเดอร์หลักของโปรเจกต์
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from communication.realsense import DepthCamera # แก้เป็น path ตามโฟลเดอร์ hardware ของคุณ

def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = load_config()
resolution_width = config['camera']['resolution_width']
resolution_height = config['camera']['resolution_height']

def get_params():
    parser = argparse.ArgumentParser(description='Create Template for Heat Exchanger')
    parser.add_argument('-t', '--target', type=str, default=None, help='ชื่อเป้าหมาย (default: A)')
    parser.add_argument('--debug', action='store_true', help='เปิดโหมด Debug เพื่อดู 3D Point Cloud')
    args = parser.parse_args()
    
    target_name = args.target
    if target_name is None:
        target_name = input("กรุณาป้อนชื่อเป้าหมาย [A, B, C, etc.]: ").strip()
        if not target_name: target_name = "A"
        
    save_dir = config['paths']['template_dir']
    log_dir = config['paths']['save_dir']
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    return save_dir, log_dir, target_name, args.debug

SAVE_DIR, LOG_DIR, CURRENT_TARGET_NAME, IS_DEBUG = get_params()

app_state = 0 # 0: Live View, 1: Frozen Annotation, 2: Tracking
polygon_points = []
template_patch = None
exact_target_pixel_manual = None
target_offset = None
frozen_color = None
frozen_gray = None

def draw_shape_callback(event, x, y, flags, param):
    global polygon_points, exact_target_pixel_manual, app_state
    if app_state != 1: return
        
    if event == cv2.EVENT_LBUTTONDOWN:
        polygon_points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        exact_target_pixel_manual = (x, y)

def rotation_matrix_to_euler_angles(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x, y, z = math.atan2(R[2, 1], R[2, 2]), math.atan2(-R[2, 0], sy), math.atan2(R[1, 0], R[0, 0])
    else:
        x, y, z = math.atan2(-R[1, 2], R[1, 1]), math.atan2(-R[2, 0], sy), 0
    return np.rad2deg([x, y, z])

def main():
    global polygon_points, template_patch, exact_target_pixel_manual, target_offset
    global app_state, frozen_color, frozen_gray
    
    print("\n[INIT] Initializing RealSense Camera...")
    cam = DepthCamera(resolution_width, resolution_height)
    
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Frame", 848, 480)
    cv2.setMouseCallback("Frame", draw_shape_callback)

    sift = cv2.SIFT_create()
    kp_template, des_template = None, None

    # --- Initialize RealSense Hardware Post-Processing Filters ---
    decimate = rs.decimation_filter()      # Downsamples for noise reduction
    decimate.set_option(rs.option.filter_magnitude, 2) 
    
    threshold = rs.threshold_filter(min_dist=0.1, max_dist=1.5) # Cuts out far background immediately
    
    spatial = rs.spatial_filter()          # Smooths depth within frames (edge-preserving)
    spatial.set_option(rs.option.filter_magnitude, 2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    
    temporal = rs.temporal_filter()        # Smooths depth over multiple frames (extremely effective for static scenes)
    
    hole_filling = rs.hole_filling_filter(1) # Fills in empty black pixels/gaps on reflective surfaces

    print(f"\n--- PHASE 1: TARGET CAPTURE ---")
    print("1. กด 'SPACEBAR' เพื่อแช่ภาพ")
    print("2. คลิกซ้ายวาดกรอบ และคลิกขวาระบุจุด Target")

    while True:
        if app_state == 0 or app_state == 2:
            ret, depth_raw_frame, color_raw_frame = cam.get_raw_frame()
            if not ret: continue
            color_frame = np.asanyarray(color_raw_frame.get_data())
            depth_frame = np.asanyarray(depth_raw_frame.get_data())
            gray_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)

        if app_state == 0:
            display_frame = color_frame.copy()
            cv2.putText(display_frame, "Aim camera -> Press SPACEBAR to Capture", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Frame", display_frame)
            key = cv2.waitKey(33) & 0xFF
            
            if key == 32: # SPACEBAR
                frozen_color = color_frame.copy()
                frozen_gray = gray_frame.copy()
                polygon_points.clear()
                exact_target_pixel_manual = None
                app_state = 1
                
        elif app_state == 1:
            display_frame = frozen_color.copy()
            if len(polygon_points) > 0:
                pts = np.array(polygon_points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(display_frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                for pt in polygon_points: cv2.circle(display_frame, pt, 4, (0, 255, 0), -1)

            if exact_target_pixel_manual is not None:
                cv2.circle(display_frame, exact_target_pixel_manual, 5, (0, 0, 255), -1)

            cv2.putText(display_frame, "L-Click=Box | R-Click=Target | 's'=Save | 'r'=Retake | 'c'=Clear", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            cv2.imshow("Frame", display_frame)
            key = cv2.waitKey(33) & 0xFF

            if key == ord('c'):
                polygon_points.clear()
                exact_target_pixel_manual = None
            elif key == ord('r'):
                app_state = 0
            elif key == ord('s') and len(polygon_points) > 2:
                x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(np.array(polygon_points))
                template_patch = frozen_gray[y_rect:y_rect+h_rect, x_rect:x_rect+w_rect]
    
                if exact_target_pixel_manual is not None:
                    target_offset = (int(exact_target_pixel_manual[0] - x_rect), int(exact_target_pixel_manual[1] - y_rect))
                else:
                    target_offset = (int(w_rect // 2), int(h_rect // 2))

                cv2.imwrite(os.path.join(SAVE_DIR, f"{CURRENT_TARGET_NAME}_template.png"), template_patch)
                print(f"[SUCCESS] Template Image saved for {CURRENT_TARGET_NAME}")
                
                kp_template, des_template = sift.detectAndCompute(template_patch, None)
                app_state = 2

        elif app_state == 2:
            display_frame = color_frame.copy()
            target_pixel = None
            th, tw = template_patch.shape
            kp_frame, des_frame = sift.detectAndCompute(gray_frame, None)
            
            if des_template is not None and des_frame is not None and len(des_template) > 2 and len(des_frame) > 2:
                flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
                try:
                    matches = flann.knnMatch(des_template, des_frame, k=2)
                    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
                    
                    if len(good_matches) > 10:
                        src_pts = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                        
                        if M is not None:
                            pts = np.float32([[0, 0], [0, th - 1], [tw - 1, th - 1], [tw - 1, 0]]).reshape(-1, 1, 2)
                            dst = cv2.perspectiveTransform(pts, M)
                            target_pt_dst = cv2.perspectiveTransform(np.float32([[[target_offset[0], target_offset[1]]]]), M)
                            target_pixel = (int(target_pt_dst[0][0][0]), int(target_pt_dst[0][0][1]))
                            
                            display_frame = cv2.polylines(display_frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
                            cv2.circle(display_frame, target_pixel, 5, (0, 0, 255), -1)
                except Exception: pass

            cv2.putText(display_frame, f"Press 'q' to Save 3D Data & JSON, or 'ESC' to exit.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow("Frame", display_frame)
            key = cv2.waitKey(33) & 0xFF
            
            if key == 27: # ESC
                break
            
            # --- จังหวะคำนวณและบันทึกข้อมูล 3D ---
            if key == ord('q') and target_pixel is not None:
                print(f"\n[PROCESSING] Extracting & Filtering 3D Data for {CURRENT_TARGET_NAME}...")
                
                # 1. Apply RealSense Hardware Post-Processing Filters
                filtered_depth_frame = threshold.process(depth_raw_frame)
                filtered_depth_frame = decimate.process(filtered_depth_frame)
                filtered_depth_frame = spatial.process(filtered_depth_frame)
                filtered_depth_frame = temporal.process(filtered_depth_frame)
                filtered_depth_frame = hole_filling.process(filtered_depth_frame)
                
                # Extract filtered depth data
                depth_filtered_data = np.asanyarray(filtered_depth_frame.get_data())
                depth_scale = cam.get_depth_scale()
                
                # 2. Re-acquire Intrinsics (Decimation changes resolution, so we pull updated intrinsics)
                filtered_intrinsics = filtered_depth_frame.profile.as_video_stream_profile().intrinsics
                
                # Prepare Open3D Inputs
                color_np_rgb = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
                
                # If decimation filter was used, we must resize the color frame to match the filtered depth map
                if depth_filtered_data.shape[1] != color_np_rgb.shape[1]:
                    color_np_rgb = cv2.resize(color_np_rgb, (depth_filtered_data.shape[1], depth_filtered_data.shape[0]), interpolation=cv2.INTER_AREA)

                o3d_color = o3d.geometry.Image(color_np_rgb)
                o3d_depth = o3d.geometry.Image(depth_filtered_data)
                
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d_color, o3d_depth, depth_scale=1.0/depth_scale, depth_trunc=1.5, convert_rgb_to_intensity=False)
                
                o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
                    depth_filtered_data.shape[1], depth_filtered_data.shape[0], 
                    filtered_intrinsics.fx, filtered_intrinsics.fy, filtered_intrinsics.ppx, filtered_intrinsics.ppy)
                
                pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d_intrinsics)
                pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]) # Flip axes
                
                # --- [NEW] 1. คำนวณหาตำแหน่งเป้าหมาย 3D ก่อนเป็นอันดับแรก ---
                scale_factor_x = depth_filtered_data.shape[1] / resolution_width
                scale_factor_y = depth_filtered_data.shape[0] / resolution_height
                u, v = int(target_pixel[0] * scale_factor_x), int(target_pixel[1] * scale_factor_y)
                
                # ป้องกันพิกัดหลุดขอบจอ
                u = max(0, min(u, depth_filtered_data.shape[1] - 1))
                v = max(0, min(v, depth_filtered_data.shape[0] - 1))
                
                target_z_raw = depth_filtered_data[v, u]
                
                if target_z_raw > 0:
                    target_z = target_z_raw * depth_scale
                    target_x = (u - filtered_intrinsics.ppx) * target_z / filtered_intrinsics.fx
                    target_y = (v - filtered_intrinsics.ppy) * target_z / filtered_intrinsics.fy
                    
                    exact_target_pos = np.array([target_x, -target_y, -target_z])

                    # --- [NEW] 2. หั่น Background ทิ้งด้วย 3D Bounding Box ---
                    # กำหนดระยะ (หน่วยเป็นเมตร) ว่าจะเก็บพื้นที่รอบๆ จุดแดงไว้กว้างแค่ไหน
                    z_margin_behind = 0.05  # สำคัญสุด! เก็บพื้นที่ "ด้านหลัง" จุดแดงแค่ 5 ซม. (หั่นกล่องกระดาษทิ้ง)
                    z_margin_front = 0.15   # เก็บพื้นที่ "ด้านหน้า" 15 ซม.
                    x_margin = 0.30         # เก็บพื้นที่ ซ้าย-ขวา ข้างละ 30 ซม. (รวม 60 ซม.)
                    y_margin = 0.20         # เก็บพื้นที่ บน-ล่าง ข้างละ 20 ซม.
                    
                    min_bound = exact_target_pos - np.array([x_margin, y_margin, z_margin_behind])
                    max_bound = exact_target_pos + np.array([x_margin, y_margin, z_margin_front])
                    
                    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
                    pcd = pcd.crop(bbox) # ตัดส่วนที่อยู่นอกกล่องล่องหนนี้ทิ้งทั้งหมด
                    
                    # --- 3. ทำความสะอาดจุดลอยๆ (Outlier Removal) หลังจากตัดขอบแล้ว ---
                    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
                    pcd = pcd.select_by_index(ind)
                    
                    # --- 4. คำนวณ Normal Vector ---
                    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.015, max_nn=30))
                    pcd.orient_normals_towards_camera_location(camera_location=np.array([0.0, 0.0, 0.0]))
                    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
                    
                    # หาจุด 3D ที่ใกล้เป้าหมายที่สุดใน Point Cloud ที่ถูกทำความสะอาดแล้ว
                    [k, idx, _] = pcd_tree.search_knn_vector_3d(exact_target_pos, 1)
                    
                    if k > 0:
                        normal = np.asarray(pcd.normals)[idx[0]]
                        # ... (โค้ดส่วนคำนวณ Rotation Matrix และบันทึก JSON ยังคงเหมือนเดิม) ...
                        
                        z_axis = normal / np.linalg.norm(normal)
                        x_axis = np.array([1, 0, 0]) if abs(z_axis[0]) < 0.9 else np.array([0, 1, 0])
                        y_axis = np.cross(z_axis, x_axis)
                        y_axis /= np.linalg.norm(y_axis)
                        x_axis = np.cross(y_axis, z_axis)
                        
                        rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
                        roll, pitch, yaw = rotation_matrix_to_euler_angles(rotation_matrix)
                        
                        # [โค้ดส่วนบันทึกไฟล์ JSON และแสดงผล 3D ด้านล่างนี้ใช้ของเดิมของคุณได้เลย]

                        # Write Full Meta JSON
                        full_meta = {
                            "target_name": CURRENT_TARGET_NAME,
                            "Position_X": round(float(target_x), 4),
                            "Position_Y": round(float(-target_y), 4),
                            "Position_Z": round(float(-target_z), 4),
                            "Roll": round(float(roll), 2),
                            "Pitch": round(float(pitch), 2),
                            "Yaw": round(float(yaw), 2),
                            "offset_x": int(target_offset[0]),
                            "offset_y": int(target_offset[1])
                        }
                        with open(os.path.join(SAVE_DIR, f"{CURRENT_TARGET_NAME}_meta.json"), "w") as f:
                            json.dump(full_meta, f, indent=4)
                        print(f"[SUCCESS] Saved clean metadata to {CURRENT_TARGET_NAME}_meta.json")

                        # Debug Output
                        if IS_DEBUG:
                            print("[DEBUG] Generating filtered PLY files and showing 3D Alignment...")
                            
                            o3d.io.write_point_cloud(os.path.join(SAVE_DIR, f"{CURRENT_TARGET_NAME}_object.ply"), pcd)
                            
                            point_marker_pcd = o3d.geometry.PointCloud()
                            point_marker_pcd.points = o3d.utility.Vector3dVector([exact_target_pos])
                            point_marker_pcd.colors = o3d.utility.Vector3dVector([[0, 1, 0]])
                            o3d.io.write_point_cloud(os.path.join(SAVE_DIR, f"{CURRENT_TARGET_NAME}_marker.ply"), point_marker_pcd)
                            
                            print(f"[DEBUG] Saved filtered {CURRENT_TARGET_NAME}_object.ply and marker.ply in {SAVE_DIR}")

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
                                                   title=f"Target {CURRENT_TARGET_NAME} (Close window to continue)")
                    break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()