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
    return np.rad2deg([x, y, z])

class PointCloudTransformer:
    def __init__(self, camera, res_width, res_height, save_dir):
        self.camera = camera
        self.res_width = res_width
        self.res_height = res_height
        self.save_dir = save_dir

    def extract_3d_data(self, target_pixels, target_names, show_3d=True):
        # เพื่อความรวดเร็ว ดึง 1 เฟรมปัจจุบัน (ถ้าจะดึง 70 เฟรมสามารถใช้ลูปบวกค่าเฉลี่ยแบบเดิมได้)
        ret, depth_raw, color_raw = self.camera.get_raw_frame()
        if not ret: return {}

        # สร้าง Open3D RGBD Image 
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

        for i, (u, v) in enumerate(target_pixels):
            target_name = target_names[i]
            target_z_raw = depth_np[v, u]
            
            if target_z_raw <= 0: continue
            
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
            
            # เก็บข้อมูล 6-DOF (X, Y, Z, Roll, Pitch, Yaw) ลง List
            extracted_6dof[target_name] = [target_x, -target_y, -target_z, roll, pitch, yaw]

            # ----------------------------------------------------
            # เซฟไฟล์ .ply และโชว์หน้าต่าง 3D ตามที่คุณต้องการ
            # ----------------------------------------------------
            try:
                ply_path = os.path.join(self.save_dir, f"{target_name}_scene.ply")
                o3d.io.write_point_cloud(ply_path, pcd)
                print(f"[TRANSFORMER] Saved Point Cloud to {ply_path}")
            except Exception as e:
                print(f"[ERROR] Could not save .ply: {e}")

            if show_3d:
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
                                       title=f"Target: {target_name} 6-DOF")

        return extracted_6dof