"""
Robotic Arm Integration Module
Converts copper pipe center points to robotic arm coordinates and movement commands
"""

import numpy as np
import cv2
from src.copper_hsv_edge import (
    preprocess_image,
    apply_hsv_mask,
    apply_morphological_operations,
    detect_contours,
    get_all_pipe_centers,
    get_target_point_for_arm
)


class RoboticArmController:
    """
    Controller for robotic arm targeting based on copper pipe detection
    """
    
    def __init__(self, image_width=640, image_height=480, 
                 camera_fov_x=69, camera_fov_y=42, 
                 camera_distance=50):
        """
        Initialize robotic arm controller
        
        Parameters:
        -----------
        image_width : int
            Camera image width (pixels)
        image_height : int
            Camera image height (pixels)
        camera_fov_x : float
            Camera horizontal field of view (degrees)
        camera_fov_y : float
            Camera vertical field of view (degrees)
        camera_distance : float
            Distance from camera to target (cm)
        """
        self.image_width = image_width
        self.image_height = image_height
        self.camera_fov_x = camera_fov_x
        self.camera_fov_y = camera_fov_y
        self.camera_distance = camera_distance
        
        # Calculate pixel to angle conversion
        self.pixel_to_angle_x = camera_fov_x / image_width
        self.pixel_to_angle_y = camera_fov_y / image_height
        
        # Calibration offset (set through calibration process)
        self.offset_x = 0
        self.offset_y = 0
        self.offset_z = 0
    
    def pixel_to_world_coordinates(self, pixel_x, pixel_y, depth=None):
        """
        Convert pixel coordinates to world/robotic arm coordinates
        
        Parameters:
        -----------
        pixel_x : int
            Pixel X coordinate (0 = left)
        pixel_y : int
            Pixel Y coordinate (0 = top)
        depth : float
            Depth from RealSense (optional)
        
        Returns:
        --------
        dict : World coordinates {x, y, z}
        """
        # Convert pixel to angle from center
        angle_x = (pixel_x - self.image_width/2) * self.pixel_to_angle_x
        angle_y = (pixel_y - self.image_height/2) * self.pixel_to_angle_y
        
        # Use provided depth or default camera distance
        z_distance = depth if depth is not None else self.camera_distance
        
        # Convert angle and depth to XY coordinates
        # Using pinhole camera model
        tan_x = np.tan(np.radians(angle_x))
        tan_y = np.tan(np.radians(angle_y))
        
        x = z_distance * tan_x
        y = z_distance * tan_y
        z = z_distance
        
        # Apply calibration offset
        x += self.offset_x
        y += self.offset_y
        z += self.offset_z
        
        return {
            'x': x,
            'y': y,
            'z': z,
            'angle_x': angle_x,
            'angle_y': angle_y
        }
    
    def world_to_arm_coordinates(self, world_coords):
        """
        Convert world coordinates to robotic arm coordinates
        (Depends on specific arm kinematics)
        
        Parameters:
        -----------
        world_coords : dict
            World coordinates {x, y, z}
        
        Returns:
        --------
        dict : Arm coordinates (joint angles or position)
        """
        # This is a placeholder - actual implementation depends on arm type
        # For a typical 6-DOF arm with base at origin:
        
        x, y, z = world_coords['x'], world_coords['y'], world_coords['z']
        
        # Calculate distance in XY plane
        r_xy = np.sqrt(x**2 + y**2)
        
        # Calculate angles
        theta_base = np.degrees(np.arctan2(y, x))  # Base rotation
        theta_pitch = np.degrees(np.arctan2(z, r_xy))  # Pitch angle
        
        return {
            'base_angle': theta_base,
            'pitch_angle': theta_pitch,
            'reach': r_xy,
            'height': z
        }
    
    def get_arm_target_from_image(self, image, selection_mode='largest'):
        """
        Process image and get robotic arm target coordinates
        
        Parameters:
        -----------
        image : np.ndarray
            Input image from camera
        selection_mode : str
            'largest' - target largest pipe
            'closest_to_center' - target closest to image center
        
        Returns:
        --------
        dict : Arm target with all coordinate systems
        """
        # Process image
        preprocessed = preprocess_image(image)
        copper_mask, _ = apply_hsv_mask(preprocessed)
        cleaned_mask = apply_morphological_operations(copper_mask)
        contours = detect_contours(cleaned_mask)
        
        if not contours:
            return None
        
        # Get pipe centers
        centers = get_all_pipe_centers(contours)
        
        # Select target
        target_pipe = get_target_point_for_arm(centers, selection_mode)
        
        if target_pipe is None:
            return None
        
        # Convert to world coordinates
        pixel_x = target_pipe['x']
        pixel_y = target_pipe['y']
        
        world_coords = self.pixel_to_world_coordinates(pixel_x, pixel_y)
        arm_coords = self.world_to_arm_coordinates(world_coords)
        
        return {
            'pixel_coords': {
                'x': pixel_x,
                'y': pixel_y,
                'radius': target_pipe['radius']
            },
            'world_coords': world_coords,
            'arm_coords': arm_coords,
            'pipe_info': {
                'area': target_pipe['area'],
                'perimeter': target_pipe['perimeter'],
                'diameter': target_pipe['radius'] * 2
            }
        }
    
    def calibrate(self, known_world_points, known_pixel_points):
        """
        Calibrate camera-to-arm coordinate mapping
        
        Parameters:
        -----------
        known_world_points : list of dict
            Known world coordinates [{x, y, z}, ...]
        known_pixel_points : list of tuple
            Corresponding pixel coordinates [(px, py), ...]
        """
        # Simple linear calibration
        # In practice, use more sophisticated methods like Zhang's method
        
        if len(known_world_points) < 3:
            print("ERROR: Need at least 3 calibration points")
            return False
        
        # Calculate average offset
        total_offset_x = 0
        total_offset_y = 0
        total_offset_z = 0
        
        for world, pixel in zip(known_world_points, known_pixel_points):
            world_est = self.pixel_to_world_coordinates(pixel[0], pixel[1])
            total_offset_x += world['x'] - world_est['x']
            total_offset_y += world['y'] - world_est['y']
            if 'z' in world:
                total_offset_z += world['z'] - world_est['z']
        
        n = len(known_world_points)
        self.offset_x = total_offset_x / n
        self.offset_y = total_offset_y / n
        self.offset_z = total_offset_z / n
        
        print("Calibration complete!")
        print("Offset X: {:.2f}".format(self.offset_x))
        print("Offset Y: {:.2f}".format(self.offset_y))
        print("Offset Z: {:.2f}".format(self.offset_z))
        
        return True
    
    def print_target_info(self, target):
        """Print formatted target information for arm movement"""
        if target is None:
            print("No target found!")
            return
        
        print("\n" + "="*70)
        print("ROBOTIC ARM TARGET INFORMATION")
        print("="*70)
        
        print("\nPIXEL COORDINATES (Image):")
        print("-"*70)
        print("  X: {} pixels".format(target['pixel_coords']['x']))
        print("  Y: {} pixels".format(target['pixel_coords']['y']))
        print("  Pipe Diameter: {} pixels".format(target['pixel_coords']['radius']*2))
        
        print("\nWORLD COORDINATES:")
        print("-"*70)
        wc = target['world_coords']
        print("  X: {:.2f} cm".format(wc['x']))
        print("  Y: {:.2f} cm".format(wc['y']))
        print("  Z: {:.2f} cm".format(wc['z']))
        print("  Angle X: {:.2f}°".format(wc['angle_x']))
        print("  Angle Y: {:.2f}°".format(wc['angle_y']))
        
        print("\nARM COORDINATES:")
        print("-"*70)
        ac = target['arm_coords']
        print("  Base Angle: {:.2f}°".format(ac['base_angle']))
        print("  Pitch Angle: {:.2f}°".format(ac['pitch_angle']))
        print("  Reach Distance: {:.2f} cm".format(ac['reach']))
        print("  Height: {:.2f} cm".format(ac['height']))
        
        print("\nPIPE INFORMATION:")
        print("-"*70)
        pi = target['pipe_info']
        print("  Area: {:.1f} pixels²".format(pi['area']))
        print("  Perimeter: {:.1f} pixels".format(pi['perimeter']))
        print("  Diameter: {:.1f} pixels".format(pi['diameter']))
        
        print("="*70)


def create_arm_command_string(target):
    """
    Create a command string for robotic arm controller
    Format depends on your specific arm system
    
    Example for KUKA KRC format:
    MOV BASE_ANGLE PITCH_ANGLE REACH HEIGHT GRIPPER_SPEED
    """
    if target is None:
        return None
    
    ac = target['arm_coords']
    pc = target['pixel_coords']
    
    # Format: BASE_ANGLE | PITCH_ANGLE | REACH | HEIGHT | DIAMETER
    command = "MOVE {:.1f} {:.1f} {:.1f} {:.1f} {:.1f}".format(
        ac['base_angle'],
        ac['pitch_angle'],
        ac['reach'],
        ac['height'],
        pc['radius'] * 2  # Pipe diameter for gripper adjustment
    )
    
    return command


# Example usage
if __name__ == "__main__":
    print("Robotic Arm Controller Initialized")
    
    # Initialize controller
    arm = RoboticArmController(
        image_width=640,
        image_height=480,
        camera_fov_x=69,
        camera_fov_y=42,
        camera_distance=50  # 50 cm default distance
    )
    
    # Example calibration points (you need to measure these physically)
    # known_world_points = [
    #     {'x': 10, 'y': 0, 'z': 50},
    #     {'x': -10, 'y': 0, 'z': 50},
    #     {'x': 0, 'y': 10, 'z': 50}
    # ]
    # known_pixel_points = [(320, 200), (320, 280), (370, 240)]
    # arm.calibrate(known_world_points, known_pixel_points)
    
    print("\nRobot Arm Controller ready for use")
