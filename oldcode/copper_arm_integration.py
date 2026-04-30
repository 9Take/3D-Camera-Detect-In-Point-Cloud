"""
Copper Pipe Detection with Robotic Arm Integration
Real-time detection and arm targeting
"""

import cv2
import numpy as np
from src.copper_hsv_edge import (
    preprocess_image,
    apply_hsv_mask,
    apply_morphological_operations,
    detect_contours,
    get_all_pipe_centers,
    get_target_point_for_arm
)
from src.robot_arm_controller import RoboticArmController, create_arm_command_string


def detect_copper_and_get_arm_target(image, arm_controller):
    """
    Process image and get robotic arm target
    
    Parameters:
    -----------
    image : np.ndarray
        Input image
    arm_controller : RoboticArmController
        Arm controller instance
    
    Returns:
    --------
    tuple : (contours, centers, target)
    """
    # Process image
    preprocessed = preprocess_image(image)
    copper_mask, _ = apply_hsv_mask(preprocessed)
    cleaned_mask = apply_morphological_operations(copper_mask)
    contours = detect_contours(cleaned_mask)
    
    if not contours:
        return contours, [], None
    
    # Get pipe centers
    centers = get_all_pipe_centers(contours)
    
    # Get arm target
    target = arm_controller.get_arm_target_from_image(image)
    
    return contours, centers, target


def visualize_with_arm_target(image, contours, centers, target):
    """
    Visualize image with detected pipes and arm target
    
    Parameters:
    -----------
    image : np.ndarray
        Original image
    contours : list
        Detected contours
    centers : list
        Pipe center points
    target : dict
        Target information
    
    Returns:
    --------
    np.ndarray : Annotated image
    """
    result = image.copy()
    
    # Draw all contours
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    
    # Draw all pipe centers
    for i, center_data in enumerate(centers):
        cx, cy = center_data['x'], center_data['y']
        radius = center_data['radius']
        
        # Draw center point
        cv2.circle(result, (cx, cy), 5, (255, 0, 0), -1)  # Blue
        # Draw pipe outline
        cv2.circle(result, (cx, cy), radius, (255, 255, 0), 2)  # Cyan
        
        # Label
        label = "P{}".format(i+1)
        cv2.putText(result, label, (cx+15, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 0, 0), 1)
    
    # Highlight primary target
    if target:
        px, py = target['pixel_coords']['x'], target['pixel_coords']['y']
        pr = target['pixel_coords']['radius']
        
        # Draw target crosshair
        cv2.line(result, (px-20, py), (px+20, py), (0, 0, 255), 2)  # Red
        cv2.line(result, (px, py-20), (px, py+20), (0, 0, 255), 2)
        cv2.circle(result, (px, py), pr+3, (0, 0, 255), 3)  # Target circle
        
        # Display target info
        info_text = "TARGET: ({},{})".format(px, py)
        cv2.putText(result, info_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 0, 255), 2)
        
        # Display arm command
        arm_cmd = create_arm_command_string(target)
        if arm_cmd:
            cv2.putText(result, "ARM CMD: {}".format(arm_cmd[:30]), (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
    
    # Display pipe count
    cv2.putText(result, "Pipes: {}".format(len(contours)), (20, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return result


def process_image_with_arm(image_path, arm_controller):
    """
    Process single image and display arm target
    
    Parameters:
    -----------
    image_path : str
        Path to image file
    arm_controller : RoboticArmController
        Arm controller instance
    """
    image = cv2.imread(image_path)
    if image is None:
        print("ERROR: Cannot load image: {}".format(image_path))
        return
    
    image = cv2.resize(image, (640, 480))
    
    print("\nProcessing: {}".format(image_path))
    
    # Detect and get target
    contours, centers, target = detect_copper_and_get_arm_target(image, arm_controller)
    
    # Visualize
    result = visualize_with_arm_target(image, contours, centers, target)
    
    # Display
    cv2.imshow("Copper Detection + Arm Target", result)
    
    # Print information
    print("Pipes detected: {}".format(len(contours)))
    
    if target:
        arm_controller.print_target_info(target)
        arm_cmd = create_arm_command_string(target)
        print("\nARM COMMAND: {}".format(arm_cmd))
    else:
        print("No target found!")
    
    print("Press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_live_camera(arm_controller):
    """
    Process live camera feed for continuous arm targeting
    
    Parameters:
    -----------
    arm_controller : RoboticArmController
        Arm controller instance
    """
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return
    
    print("\n" + "="*70)
    print("LIVE CAMERA - ROBOTIC ARM TARGETING")
    print("="*70)
    print("Press 'q' to quit, 's' to save target")
    print("="*70 + "\n")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("ERROR: Cannot read frame")
            break
        
        frame = cv2.resize(frame, (640, 480))
        frame_count += 1
        
        # Detect and get target
        contours, centers, target = detect_copper_and_get_arm_target(frame, arm_controller)
        
        # Visualize
        result = visualize_with_arm_target(frame, contours, centers, target)
        
        # Add FPS counter
        cv2.putText(result, "Frame: {}".format(frame_count), (500, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow("Live Copper Detection + Arm Control", result)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s') and target:
            # Save target information
            arm_cmd = create_arm_command_string(target)
            print("\nTarget saved!")
            print("Coordinates: X={:.2f}, Y={:.2f}, Z={:.2f}".format(
                target['world_coords']['x'],
                target['world_coords']['y'],
                target['world_coords']['z']
            ))
            print("Arm Command: {}".format(arm_cmd))
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nCamera closed")


def main():
    """Main function"""
    print("\n" + "="*70)
    print("COPPER PIPE DETECTION + ROBOTIC ARM INTEGRATION")
    print("="*70)
    
    # Initialize arm controller
    arm = RoboticArmController(
        image_width=640,
        image_height=480,
        camera_fov_x=69,
        camera_fov_y=42,
        camera_distance=50  # 50 cm distance
    )
    
    print("\nMode Selection:")
    print("[1] Process static images from 'pic' folder")
    print("[2] Live camera feed")
    print("[3] Run arm controller test")
    
    mode = input("\nSelect mode (1/2/3): ").strip()
    
    if mode == '1':
        # Process images from pic folder
        import glob
        import os
        
        image_files = glob.glob("pic/*.jpg") + glob.glob("pic/*.png")
        
        if not image_files:
            print("ERROR: No images found in 'pic' folder")
            return
        
        for img_path in sorted(image_files):
            process_image_with_arm(img_path, arm)
    
    elif mode == '2':
        # Live camera feed
        process_live_camera(arm)
    
    elif mode == '3':
        # Test arm controller
        test_arm_controller(arm)
    
    else:
        print("Invalid mode!")


def test_arm_controller(arm):
    """Test arm controller with dummy data"""
    print("\n" + "="*70)
    print("ARM CONTROLLER TEST")
    print("="*70)
    
    # Test coordinate conversion
    test_pixels = [
        (320, 240),  # Center
        (100, 100),  # Top-left
        (500, 400),  # Bottom-right
    ]
    
    print("\nPixel -> World -> Arm Coordinates Conversion Test:\n")
    
    for px, py in test_pixels:
        world = arm.pixel_to_world_coordinates(px, py)
        arm_coords = arm.world_to_arm_coordinates(world)
        
        print("Pixel: ({}, {})".format(px, py))
        print("  -> World: X={:.2f}, Y={:.2f}, Z={:.2f}".format(
            world['x'], world['y'], world['z']))
        print("  -> Arm: Base={:.2f}°, Pitch={:.2f}°".format(
            arm_coords['base_angle'], arm_coords['pitch_angle']))
        print()
    
    print("="*70)


if __name__ == "__main__":
    main()
