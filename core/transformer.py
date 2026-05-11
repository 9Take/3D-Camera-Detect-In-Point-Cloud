import math
import numpy as np
import open3d as o3d
import cv2
import os

def rotation_matrix_to_euler_angles(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x, y, z = math.atan2(R[2, 1], R[2, 2]), math.atan2(-R[2, 0], sy), math.atan2(R[1, 0], R[0, 0])
    else:
        x, y, z = math.atan2(-R[1, 2], R[1, 1]), math.atan2(-R[2, 0], sy), 0
    return [x, y, z]

class PointCloudTransformer:
    def __init__(self, camera, res_width, res_height, save_dir):
        self.camera = camera
        self.res_width = res_width
        self.res_height = res_height
        self.save_dir = save_dir

    def extract_3d_data(self, target_pixels, target_names, show_3d=True):
        ret, depth_raw, color_raw = self.camera.get_raw_frame()
        if not ret: return {}

        color_np = np.asanyarray(color_raw.get_data())
        depth_np = np.asanyarray(depth_raw.get_data())
        
        o3d_color = o3d.geometry.Image(cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB))
        o3d_depth = o3d.geometry.Image(depth_np)
        
        intrinsics = depth_raw.profile.as_video_stream_profile().intrinsics
        depth_scale = self.camera.get_depth_scale()
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, o3d_depth, depth_scale=1.0/depth_scale, depth_trunc=1.5, convert_rgb_to_intensity=False)
        
        o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            self.res_width, self.res_height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)
        
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d_intrinsics)
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]) # Flip upside down
        
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)

        extracted_6dof = {}
        
        # สำหรับเก็บวัตถุ 3D (Sphere และ Axis) ของทุกตัวที่ประมวลผลสำเร็จ เพื่อเอาไว้วาดทีเดียวพร้อมกัน
        geometries_to_draw = [{"name": "pcd", "geometry": pcd, "material": o3d.visualization.rendering.MaterialRecord()}]
        geometries_to_draw[0]["material"].shader = "defaultUnlit"

        for i, (u, v) in enumerate(target_pixels):
            target_name = target_names[i]
            target_z_raw = depth_np[v, u]
            
            # --- อัปเกรด Logic แก้ไขจุดบอดของกล้อง (อ่านค่าความลึกเป็น 0) ---
            if target_z_raw <= 0:
                # ลองค้นหาค่าเฉลี่ยในพื้นที่รอบๆ โดยขยายรัศมีออกไปเรื่อยๆ (ขยายสูงสุดเป็น 15x15 pixels)
                found_valid_depth = False
                for r in range(2, 8):  # ลองขยายรัศมีทีละ 2, 3, 4, 5, 6, 7 พิกเซล
                    y_min, y_max = max(0, v - r), min(self.res_height, v + r + 1)
                    x_min, x_max = max(0, u - r), min(self.res_width, u + r + 1)
                    
                    roi = depth_np[y_min:y_max, x_min:x_max]
                    valid_depths = roi[roi > 0]
                    
                    if len(valid_depths) > 0:
                        # นำค่าเฉลี่ยที่ได้มาใช้งาน และหลุดออกจากลูป
                        target_z_raw = np.mean(valid_depths)
                        found_valid_depth = True
                        print(f"[TRANSFORMER] Remedied Depth for '{target_name}' at radius {r}. Depth: {target_z_raw * depth_scale:.4f}m")
                        break
                
                if not found_valid_depth:
                    print(f"[TRANSFORMER WARNING] Skipped '{target_name}' because Depth value is completely 0 in 15x15 neighborhood.")
                    continue
            
            target_z = target_z_raw * depth_scale
            target_x = (u - intrinsics.ppx) * target_z / intrinsics.fx
            target_y = (v - intrinsics.ppy) * target_z / intrinsics.fy
            
            exact_target_pos = np.array([target_x, -target_y, -target_z])
            [k, idx, _] = pcd_tree.search_knn_vector_3d(exact_target_pos, 1)
            
            if k == 0: 
                print(f"[TRANSFORMER WARNING] Skipped '{target_name}' because 3D Point Cloud matching failed.")
                continue
            
            normal = np.asarray(pcd.normals)[idx[0]]
            z_axis = normal / np.linalg.norm(normal)
            x_axis = np.array([1, 0, 0]) if abs(z_axis[0]) < 0.9 else np.array([0, 1, 0])
            y_axis = np.cross(z_axis, x_axis)
            y_axis /= np.linalg.norm(y_axis)
            x_axis = np.cross(y_axis, z_axis)
            
            rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
            roll, pitch, yaw = rotation_matrix_to_euler_angles(rotation_matrix)
            
            # บันทึกข้อมูล 6-DOF
            extracted_6dof[target_name] = [target_x, -target_y, -target_z, roll, pitch, yaw]

            # บันทึกไฟล์ .ply แยกของแต่ละวัตถุตามปกติ
            try:
                ply_path = os.path.join(self.save_dir, f"{target_name}_scene.ply")
                o3d.io.write_point_cloud(ply_path, pcd)
                print(f"[TRANSFORMER] Saved Point Cloud to {ply_path}")
            except Exception as e:
                print(f"[ERROR] Could not save .ply: {e}")

            # สร้ากราฟิก 3D (Sphere & Axis) ของตัวนี้ เตรียมนำไปแสดงรวมกัน
            if show_3d:
                # สร้าง Sphere สีเขียว (จุด Center 3D)
                target_ball = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
                # แยกสีเพื่อให้ดูง่าย (A = เขียว, B = ฟ้า)
                ball_color = [0, 1, 0] if target_name.startswith('A') else [0, 0.8, 1]
                target_ball.paint_uniform_color(ball_color)
                target_ball.translate(exact_target_pos)
                
                # สร้าง แกน 3 มิติ (Local Coordinate Frame)
                axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.04, origin=[0,0,0])
                axis.rotate(rotation_matrix, center=[0,0,0])
                axis.translate(exact_target_pos)

                mat_unlit = o3d.visualization.rendering.MaterialRecord()
                mat_unlit.shader = "defaultUnlit"

                geometries_to_draw.append({"name": f"target_{target_name}", "geometry": target_ball, "material": mat_unlit})
                geometries_to_draw.append({"name": f"axis_{target_name}", "geometry": axis, "material": mat_unlit})

        # --- ส่วนแสดงผลหน้าต่าง Open3D หลังคำนวณวัตถุครบทุกชิ้นแล้ว ---
        if show_3d and len(geometries_to_draw) > 1:
            o3d.visualization.draw(geometries_to_draw, title="All Targets 6-DOF Detection Grid")
            
        return extracted_6dof