import numpy as np
import pyrealsense2 as rs
import cv2
import open3d as o3d
import copy  
import os
import math
from realsense_depth import DepthCamera
from utils import createPointCloudO3D

# --- Configuration ---
resolution_width, resolution_height = (640, 480)
SAVE_DIR = "test5"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)   

selected_pixel = None

def mouse_callback(event, x, y, flags, param):
    global selected_pixel
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_pixel = (x, y)

def rotation_matrix_to_euler_angles(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    if not (sy < 1e-6):
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.rad2deg([x, y, z])

def capture_multiple_frames(camera, num_frames=5, voxel_size=0.002):
    """Captures N frames quickly and merges them into one downsampled cloud."""
    print(f"Capturing {num_frames} frames... keep camera still.")
    pcds = []
    for _ in range(num_frames):
        ret, d_frame, c_frame = camera.get_raw_frame()
        if ret:
            pcds.append(createPointCloudO3D(c_frame, d_frame))
            
    merged_pcd = o3d.geometry.PointCloud()
    for pcd in pcds:
        merged_pcd += pcd
        
    return merged_pcd.voxel_down_sample(voxel_size=voxel_size)

def preprocess_point_cloud(pcd, voxel_size):
    """Prepares the point cloud for Global Registration by calculating normals and FPFH features."""
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    # Calculate normals (required for Point-to-Plane ICP and FPFH)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    # Calculate FPFH features (geometric signatures to help align large movements)
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def align_point_clouds_robust(source, target, voxel_size=0.005):
    """Aligns source to target using Global RANSAC followed by Local ICP."""
    print("Extracting geometric features...")
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    distance_threshold = voxel_size * 1.5

    print("Running Global Alignment (RANSAC)...")
    # 1. Global Registration: Finds a rough alignment even if the camera moved a lot
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))

    print("Running Local Alignment (Point-to-Plane ICP)...")
    # 2. Local Registration: Uses the RANSAC result as the starting point and tightly locks the surfaces
    result_icp = o3d.pipelines.registration.registration_icp(
        source_down, target_down, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))

    # Apply the final mathematically calculated transformation to the original source cloud
    source.transform(result_icp.transformation)
    return source

def main():
    global selected_pixel
    Realsensed435Cam = DepthCamera(resolution_width, resolution_height)
    
    pcd_front = None
    pcd_right = None
    pcd_left = None
    color_frame_front = None
    intrinsics_front = None

    print("\n--- 3D SCANNER CONTROLS ---")
    print("Press 'f': Capture FRONT view (Base Coordinate System)")
    print("Press 'r': Capture RIGHT view")
    print("Press 'l': Capture LEFT view")
    print("Press 'c': COMPUTE & Stitch all captured views")
    print("---------------------------\n")

    while True:
        ret , depth_raw_frame, color_raw_frame = Realsensed435Cam.get_raw_frame()
        if not ret: continue
        
        color_frame = np.asanyarray(color_raw_frame.get_data())
        
        status_text = f"Views Captured -> F:{'Yes' if pcd_front else 'No'} | R:{'Yes' if pcd_right else 'No'} | L:{'Yes' if pcd_left else 'No'}"
        cv2.putText(color_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Frame", color_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('f'):
            pcd_front = capture_multiple_frames(Realsensed435Cam)
            color_frame_front = color_frame.copy()
            intrinsics_front = depth_raw_frame.profile.as_video_stream_profile().intrinsics
            print("[SUCCESS] Front view captured.")
            
        elif key == ord('r'):
            pcd_right = capture_multiple_frames(Realsensed435Cam)
            print("[SUCCESS] Right view captured.")
            
        elif key == ord('l'):
            pcd_left = capture_multiple_frames(Realsensed435Cam)
            print("[SUCCESS] Left view captured.")
            
        elif key == ord('c'):
            if pcd_front is None:
                print("[ERROR] You MUST capture the Front view ('f') first! It is the base.")
                continue
            
            print("\n--- Starting Robust Stitching Process ---")
            final_pcd = o3d.geometry.PointCloud()
            final_pcd += pcd_front
            
            # Using 5mm voxel size for alignment features to process quickly and reliably
            alignment_voxel_size = 0.005 
            
            if pcd_right is not None:
                print("\n-> Aligning RIGHT view to FRONT view")
                aligned_right = align_point_clouds_robust(pcd_right, pcd_front, alignment_voxel_size)
                final_pcd += aligned_right
                
            if pcd_left is not None:
                print("\n-> Aligning LEFT view to FRONT view")
                # We align left to the newly updated final_pcd to give it more surface area to latch onto
                aligned_left = align_point_clouds_robust(pcd_left, final_pcd, alignment_voxel_size)
                final_pcd += aligned_left
                
            # Final cleanup of the stitched cloud for visualization and coordinate picking
            final_pcd = final_pcd.voxel_down_sample(voxel_size=0.002)
            print("\n[SUCCESS] Point clouds merged smoothly!\n")
            break 
            
    # --- PHASE 2: TARGET SELECTION ---
    print("Click on the copper pipe in the image, then press ENTER.")
    cv2.setMouseCallback("Frame", mouse_callback)
    
    while True:
        display_frame = color_frame_front.copy()
        if selected_pixel is not None:
            cv2.circle(display_frame, selected_pixel, 5, (0, 255, 0), -1)
        cv2.imshow("Frame", display_frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 13 and selected_pixel is not None: break

    # 1. Estimate Normals for the Stitched Cloud
    final_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    
    # 2. Snap to Surface (Using the Front Camera Intrinsics)
    u, v = selected_pixel
    obj_points = np.asarray(final_pcd.points)
    
    raw_x, raw_y, raw_z = obj_points[:, 0], -obj_points[:, 1], -obj_points[:, 2]
    u_proj = (raw_x * intrinsics_front.fx / raw_z) + intrinsics_front.ppx
    v_proj = (raw_y * intrinsics_front.fy / raw_z) + intrinsics_front.ppy
    
    dist_sq = (u_proj - u)**2 + (v_proj - v)**2
    best_idx = np.argmin(dist_sq)
    target_pos = obj_points[best_idx]
    
    # 3. Calculate Orientation
    normal = np.asarray(final_pcd.normals)[best_idx]
    z_axis = normal / np.linalg.norm(normal)
    x_axis = np.array([1, 0, 0]) if abs(z_axis[0]) < 0.9 else np.array([0, 1, 0])
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    
    rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
    roll, pitch, yaw = rotation_matrix_to_euler_angles(rotation_matrix)

    # --- 4. PRINT RESULT ---
    print("\n" + "="*50)
    print("GREEN POINT 6-DOF DATA:")
    print(f"Position (X, Y, Z): {target_pos}")
    print(f"Orientation (R, P, Y): {roll:.2f}, {pitch:.2f}, {yaw:.2f}")
    print("="*50)

    # --- 5. SAVE FILES ---
    point_marker_pcd = o3d.geometry.PointCloud()
    point_marker_pcd.points = o3d.utility.Vector3dVector([target_pos])
    point_marker_pcd.colors = o3d.utility.Vector3dVector([[0, 1, 0]])

    pcd_path = os.path.join(SAVE_DIR, "target_object.ply")
    marker_path = os.path.join(SAVE_DIR, "green_point_marker.ply")
    txt_path = os.path.join(SAVE_DIR, "target_data.txt")

    try:
        o3d.io.write_point_cloud(pcd_path, final_pcd)
        o3d.io.write_point_cloud(marker_path, point_marker_pcd)
        with open(txt_path, "w") as f:
            f.write(f"Position_X: {target_pos[0]:.6f}\n")
            f.write(f"Position_Y: {target_pos[1]:.6f}\n")
            f.write(f"Position_Z: {target_pos[2]:.6f}\n")
            f.write(f"Roll: {roll:.2f}\n")
            f.write(f"Pitch: {pitch:.2f}\n")
            f.write(f"Yaw: {yaw:.2f}\n")
        print(f"[SUCCESS] Files saved to: {os.path.abspath(SAVE_DIR)}")
    except Exception as e:
        print(f"[ERROR] Could not save files: {e}")

    # 6. Visualization
    target_ball = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
    target_ball.paint_uniform_color([0, 1, 0])
    target_ball.translate(target_pos)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02, origin=target_pos)
    axis.rotate(rotation_matrix, center=target_pos)

    mat_unlit = o3d.visualization.rendering.MaterialRecord()
    mat_unlit.shader = "defaultUnlit"
    
    o3d.visualization.draw([{"name": "pcd", "geometry": final_pcd, "material": mat_unlit},
                           {"name": "target", "geometry": target_ball, "material": mat_unlit},
                           {"name": "axis", "geometry": axis, "material": mat_unlit}],
                           title="Stitched 6-DOF Target Information")
            
    Realsensed435Cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()