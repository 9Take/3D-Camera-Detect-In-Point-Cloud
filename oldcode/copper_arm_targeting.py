# -*- coding: utf-8 -*-
"""
Copper Pipe Detection with Robotic Arm Targeting
Combines:
- realcopper.py: Camera initialization and 6-DOF pose calculation
- copper_hsv_edge.py: HSV detection pipeline and center point calculation
- ROI Selection: Mouse drag to select region of interest

Author: Development Team
"""

import numpy as np
import pyrealsense2 as rs
import cv2
import open3d as o3d
import copy  
import os
import math
from datetime import datetime
from realsense_depth import DepthCamera
from utils import createPointCloudO3D

# --- Configuration ---
resolution_width, resolution_height = (640, 480)

# HSV range for copper color
COPPER_HSV_LOWER1 = np.array([0, 50, 50])
COPPER_HSV_UPPER1 = np.array([25, 255, 255])
COPPER_HSV_LOWER2 = np.array([0, 30, 30])
COPPER_HSV_UPPER2 = np.array([30, 255, 200])

# Preprocessing parameters
BLUR_KERNEL = (7, 7)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

# Output settings
SAVE_DIR = "arm_target_data"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Global variables
selected_roi = None
mouse_drawing = False
mouse_start = None
mouse_end = None


def mouse_callback(event, x, y, flags, param):
    """Mouse callback for ROI selection (drag rectangle)"""
    global mouse_drawing, mouse_start, mouse_end
    
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_drawing = True
        mouse_start = (x, y)
        mouse_end = None
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if mouse_drawing:
            mouse_end = (x, y)
    
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_drawing = False
        if mouse_start and mouse_end:
            mouse_end = (x, y)


def preprocess_image(image):
    """Preprocess image to handle reflections"""
    bilateral = cv2.bilateralFilter(image, 9, 75, 75)
    lab = cv2.cvtColor(bilateral, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    preprocessed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return cv2.GaussianBlur(preprocessed, BLUR_KERNEL, 0)


def apply_hsv_mask(image):
    """Apply HSV mask for copper detection"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, COPPER_HSV_LOWER1, COPPER_HSV_UPPER1)
    mask2 = cv2.inRange(hsv, COPPER_HSV_LOWER2, COPPER_HSV_UPPER2)
    return cv2.bitwise_or(mask1, mask2), hsv


def apply_morphological_operations(mask):
    """Apply morphological operations to clean mask"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_large, iterations=2)
    return cv2.morphologyEx(closed, cv2.MORPH_DILATE, kernel, iterations=1)


def detect_contours(mask):
    """Detect contours in mask"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > 300:
            valid_contours.append(contour)
    return valid_contours


def calculate_pipe_center(contour):
    """Calculate center point of a pipe using moments"""
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None, None, None
    
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    (circle_cx, circle_cy), radius = cv2.minEnclosingCircle(contour)
    
    return cx, cy, int(radius)


def detect_copper_in_roi(color_frame, x1, y1, x2, y2):
    """Detect copper pipe and get center in ROI"""
    # Extract ROI
    roi = color_frame[y1:y2, x1:x2].copy()
    
    # Apply preprocessing
    preprocessed = preprocess_image(roi)
    
    # Apply HSV segmentation
    copper_mask, _ = apply_hsv_mask(preprocessed)
    
    # Clean mask
    cleaned_mask = apply_morphological_operations(copper_mask)
    
    # Detect contours
    contours = detect_contours(cleaned_mask)
    
    if contours:
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        cx_roi, cy_roi, radius = calculate_pipe_center(largest_contour)
        
        if cx_roi is not None:
            # Convert ROI coordinates to full frame coordinates
            cx = cx_roi + x1
            cy = cy_roi + y1
            
            return cx, cy, radius, roi, copper_mask, cleaned_mask
    
    return None, None, None, roi, None, None


def rotation_matrix_to_euler_angles(R):
    """Convert rotation matrix to Euler angles (Roll, Pitch, Yaw)"""
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


def create_and_snap_to_surface(color_frame, depth_frame, cx, cy):
    """Create point cloud and snap pixel to nearest surface using projection method (from realcopper.py)"""
    # Create point cloud
    pcd = createPointCloudO3D(color_frame, depth_frame)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    
    # Get depth intrinsics
    intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
    
    # Get all point cloud points
    obj_points = np.asarray(pcd.points)
    
    if len(obj_points) == 0:
        print("[WARNING] Empty point cloud")
        return pcd, np.array([0, 0, 0]), None
    
    # Project 3D points back to 2D pixel coordinates (realcopper.py method)
    raw_x = obj_points[:, 0]
    raw_y = -obj_points[:, 1]
    raw_z = -obj_points[:, 2]
    
    # Project using intrinsic matrix
    u_proj = (raw_x * intrinsics.fx / raw_z) + intrinsics.ppx
    v_proj = (raw_y * intrinsics.fy / raw_z) + intrinsics.ppy
    
    # Find closest point to selected pixel
    dist_sq = (u_proj - cx)**2 + (v_proj - cy)**2
    best_idx = np.argmin(dist_sq)
    snapped_point = obj_points[best_idx]
    
    return pcd, snapped_point, best_idx


def estimate_surface_normal_at_point(pcd, point_idx):
    """Get surface normal at a specific point"""
    if point_idx is None or point_idx >= len(pcd.normals):
        return np.array([0, 0, 1])
    
    return np.asarray(pcd.normals)[point_idx]


def calculate_target_pose(target_point, surface_normal):
    """Calculate 6-DOF pose (position + orientation) for robotic arm"""
    # Normalize surface normal
    z_axis = surface_normal / (np.linalg.norm(surface_normal) + 1e-6)
    
    # Create orthogonal axes
    if abs(z_axis[0]) < 0.9:
        x_axis = np.array([1, 0, 0])
    else:
        x_axis = np.array([0, 1, 0])
    
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-6)
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-6)
    
    # Create rotation matrix
    rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
    
    # Convert to Euler angles
    roll, pitch, yaw = rotation_matrix_to_euler_angles(rotation_matrix)
    
    return {
        'position': target_point,
        'x': target_point[0],
        'y': target_point[1],
        'z': target_point[2],
        'orientation': [roll, pitch, yaw],
        'roll': roll,
        'pitch': pitch,
        'yaw': yaw,
        'rotation_matrix': rotation_matrix
    }


def save_target_data(target_pose, pixel_coords):
    """Save target pose data with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    txt_filename = os.path.join(SAVE_DIR, "target_data_{}.txt".format(timestamp))
    ply_filename = os.path.join(SAVE_DIR, "target_data_{}.ply".format(timestamp))
    
    with open(txt_filename, "w") as f:
        f.write("="*70 + "\n")
        f.write("ROBOTIC ARM TARGET DATA\n")
        f.write("="*70 + "\n\n")
        
        f.write("POSITION (meters):\n")
        f.write("  X: {:.6f}\n".format(target_pose['x']))
        f.write("  Y: {:.6f}\n".format(target_pose['y']))
        f.write("  Z: {:.6f}\n\n".format(target_pose['z']))
        
        f.write("ORIENTATION (degrees):\n")
        f.write("  Roll  (X-axis): {:.2f}\n".format(target_pose['roll']))
        f.write("  Pitch (Y-axis): {:.2f}\n".format(target_pose['pitch']))
        f.write("  Yaw   (Z-axis): {:.2f}\n\n".format(target_pose['yaw']))
        
        f.write("PIXEL COORDINATES:\n")
        f.write("  X: {}\n".format(pixel_coords[0]))
        f.write("  Y: {}\n".format(pixel_coords[1]))
        f.write("\n")
        
        f.write("ROTATION MATRIX:\n")
        for row in target_pose['rotation_matrix']:
            f.write("  {}\n".format(row))
    
    print("[SAVED] Target data: {}".format(txt_filename))


def visualize_target(pcd, target_pose):
    """Visualize point cloud with target point and coordinate frame"""
    # Create target sphere (green marker)
    target_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
    target_sphere.paint_uniform_color([0, 1, 0])
    target_sphere.translate(target_pose['position'])
    
    # Create coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.05, origin=target_pose['position']
    )
    coord_frame.rotate(target_pose['rotation_matrix'], center=target_pose['position'])
    
    # Show visualization
    o3d.visualization.draw(
        [
            {"name": "PointCloud", "geometry": pcd},
            {"name": "Target", "geometry": target_sphere},
            {"name": "Axes", "geometry": coord_frame}
        ],
        title="Copper Pipe Detection - Robotic Arm Target"
    )


def main():
    """Main function for robotic arm targeting"""
    global mouse_drawing, mouse_start, mouse_end, selected_roi
    
    print("\n" + "="*70)
    print("COPPER PIPE DETECTION - ROBOTIC ARM TARGETING")
    print("RealSense D435i Camera + HSV Detection + Drag ROI Selection")
    print("="*70 + "\n")
    
    # Initialize camera
    try:
        cam = DepthCamera(resolution_width, resolution_height)
        print("[OK] Camera initialized: {}x{}".format(resolution_width, resolution_height))
    except Exception as e:
        print("[ERROR] Failed to initialize camera: {}".format(str(e)))
        return
    
    print("\nInstructions:")
    print("  1. Click and drag mouse to select ROI containing copper pipe")
    print("  2. Release mouse to confirm selection")
    print("  3. Press 'c' to confirm and process, 'r' to reselect, 'ESC' to quit\n")
    
    try:
        frame_count = 0
        
        while True:
            # Get frames
            ret, depth_raw_frame, color_raw_frame = cam.get_raw_frame()
            
            if not ret:
                print("[WARNING] Failed to get frame")
                continue
            
            frame_count += 1
            color_frame = np.asarray(color_raw_frame.get_data())
            
            # Draw ROI selection
            display = color_frame.copy()
            
            if mouse_start and mouse_end:
                x1, y1 = mouse_start
                x2, y2 = mouse_end
                # Ensure proper ordering
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display, "ROI: ({},{}) to ({},{})".format(x1, y1, x2, y2),
                          (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display, "Press 'c' to confirm, 'r' to redraw, ESC to quit",
                          (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
            else:
                cv2.putText(display, "Click and drag to select ROI",
                          (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 0), 2)
            
            cv2.imshow("RealSense - Select ROI (Drag)", display)
            cv2.setMouseCallback("RealSense - Select ROI (Drag)", mouse_callback)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                print("\n[INFO] Application terminated by user")
                break
            
            elif key == ord('r'):  # Reset
                mouse_start = None
                mouse_end = None
                print("[INFO] ROI selection reset")
            
            elif key == ord('c') and mouse_start and mouse_end:  # Confirm
                x1, y1 = mouse_start
                x2, y2 = mouse_end
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                print("\n" + "="*70)
                print("PROCESSING ROI: ({},{}) to ({},{})".format(x1, y1, x2, y2))
                print("="*70)
                
                # Detect copper in ROI
                print("[1/5] Detecting copper in selected ROI...")
                cx, cy, radius, roi, copper_mask, cleaned_mask = detect_copper_in_roi(
                    color_frame, x1, y1, x2, y2
                )
                
                if cx is None:
                    print("[WARNING] No copper detected in ROI. Try again.")
                    mouse_start = None
                    mouse_end = None
                    continue
                
                print("[1/5] Copper detected at pixel ({}, {})".format(cx, cy))
                
                # Create point cloud and snap to surface
                print("[2/5] Creating point cloud and snapping to surface...")
                pcd, snapped_point, point_idx = create_and_snap_to_surface(
                    color_raw_frame, depth_raw_frame, cx, cy
                )
                
                # Estimate surface normal
                print("[3/5] Estimating surface normal...")
                surface_normal = estimate_surface_normal_at_point(pcd, point_idx)
                
                # Calculate 6-DOF pose
                print("[4/5] Calculating 6-DOF pose...")
                target_pose = calculate_target_pose(snapped_point, surface_normal)
                
                # Display results
                print("\n" + "="*70)
                print("ROBOTIC ARM TARGET COORDINATES")
                print("="*70)
                print("\nPOSITION (meters):")
                print("  X: {:.6f} m".format(target_pose['x']))
                print("  Y: {:.6f} m".format(target_pose['y']))
                print("  Z: {:.6f} m".format(target_pose['z']))
                
                print("\nORIENTATION (degrees):")
                print("  Roll  (X-axis): {:.2f}°".format(target_pose['roll']))
                print("  Pitch (Y-axis): {:.2f}°".format(target_pose['pitch']))
                print("  Yaw   (Z-axis): {:.2f}°".format(target_pose['yaw']))
                
                print("\nPIXEL COORDINATES:")
                print("  X: {} pixels".format(cx))
                print("  Y: {} pixels".format(cy))
                print("="*70)
                
                # Save results
                print("[5/5] Saving results...")
                save_target_data(target_pose, (cx, cy))
                
                # Visualization
                print("\n[INFO] Launching 3D visualization...")
                visualize_target(pcd, target_pose)
                
                # Reset for next iteration
                mouse_start = None
                mouse_end = None
                print("\n[INFO] Ready for next target")
    
    finally:
        cam.release()
        cv2.destroyAllWindows()
        print("\n" + "="*70)
        print("Camera released and application closed")
        print("="*70 + "\n")


if __name__ == "__main__":
    main()
