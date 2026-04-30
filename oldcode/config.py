"""
Configuration file for Copper Pipe Detection System
Adjust these parameters based on your specific hardware and requirements
"""

# ============================================================================
# COLOR DETECTION THRESHOLDS (HSV COLOR SPACE)
# ============================================================================

# Primary copper color range (Bright copper/red tones)
COPPER_HSV_LOWER1 = {
    'H': 0,      # Hue minimum (0-180 in OpenCV)
    'S': 50,     # Saturation minimum (0-255)
    'V': 50      # Value minimum (0-255)
}

COPPER_HSV_UPPER1 = {
    'H': 25,     # Hue maximum
    'S': 255,    # Saturation maximum
    'V': 255     # Value maximum
}

# Secondary copper color range (Darker/aged copper)
COPPER_HSV_LOWER2 = {
    'H': 0,
    'S': 30,
    'V': 30
}

COPPER_HSV_UPPER2 = {
    'H': 30,
    'S': 255,
    'V': 200
}

# ============================================================================
# PREPROCESSING PARAMETERS
# ============================================================================

# Bilateral Filter (reduces noise while preserving edges)
BILATERAL_DIAMETER = 9           # Diameter of each pixel neighborhood
BILATERAL_COLOR_SIGMA = 75       # Filter sigma in the color space
BILATERAL_SPACE_SIGMA = 75       # Filter sigma in the coordinate space

# CLAHE (Contrast Limited Adaptive Histogram Equalization)
CLAHE_CLIP_LIMIT = 2.0           # Contrast limit
CLAHE_TILE_GRID = (8, 8)         # Tile size for histogram equalization

# Gaussian Blur
BLUR_KERNEL = (7, 7)             # Kernel size (must be odd)

# ============================================================================
# MORPHOLOGICAL OPERATIONS
# ============================================================================

# Opening kernel (removes small noise)
OPENING_KERNEL_SIZE = (5, 5)
OPENING_ITERATIONS = 2

# Closing kernel (fills small holes)
CLOSING_KERNEL_SIZE = (7, 7)
CLOSING_ITERATIONS = 2

# Dilation kernel (enhances connectivity)
DILATION_KERNEL_SIZE = (5, 5)
DILATION_ITERATIONS = 1

# ============================================================================
# EDGE DETECTION (Canny)
# ============================================================================

CANNY_THRESHOLD1 = 50            # Lower threshold
CANNY_THRESHOLD2 = 150           # Upper threshold

# ============================================================================
# CONTOUR DETECTION & FILTERING
# ============================================================================

MIN_CONTOUR_AREA = 300           # Minimum area in pixels to consider as pipe
MIN_CONTOUR_PERIMETER = 20       # Minimum perimeter

# ============================================================================
# DISTANCE TRANSFORM & WATERSHED
# ============================================================================

WATERSHED_SURE_FG_THRESHOLD = 0.5  # Threshold for sure foreground detection
WATERSHED_DILATION_ITERATIONS = 2  # For sure background detection

# ============================================================================
# CAMERA CONFIGURATION
# ============================================================================

# RealSense Camera Settings
REALSENSE_COLOR_RESOLUTION = (1280, 720)  # (width, height)
REALSENSE_DEPTH_RESOLUTION = (1280, 720)
REALSENSE_FPS = 30                         # Frames per second

# Image resize parameters
STANDARD_RESIZE = (640, 480)               # Standard processing resolution

# ============================================================================
# DISPLAY & VISUALIZATION
# ============================================================================

DISPLAY_WINDOW_FLAGS = cv2.WINDOW_NORMAL  # Normal or autosize window
VISUALIZATION_FONT = cv2.FONT_HERSHEY_SIMPLEX
VISUALIZATION_FONT_SCALE = 0.7
VISUALIZATION_THICKNESS = 2

# Colors for visualization (BGR format)
COLOR_GREEN = (0, 255, 0)        # Detected contours
COLOR_BLUE = (255, 0, 0)         # Labels
COLOR_RED = (0, 0, 255)          # Errors/warnings
COLOR_YELLOW = (0, 255, 255)     # Selections
COLOR_CYAN = (255, 255, 0)       # Active selections

# ============================================================================
# PROCESSING OPTIONS
# ============================================================================

# Enable/disable processing steps
ENABLE_PREPROCESSING = True
ENABLE_HSV_SEGMENTATION = True
ENABLE_MORPHOLOGICAL_OPS = True
ENABLE_DISTANCE_TRANSFORM = True
ENABLE_WATERSHED = True
ENABLE_EDGE_DETECTION = True
ENABLE_CONTOUR_DETECTION = True

# ============================================================================
# OUTPUT & STATISTICS
# ============================================================================

PRINT_DETAILED_STATS = True      # Print detailed pipe statistics
PRINT_PROCESSING_TIME = True     # Print processing time for each step
SAVE_RESULTS = False             # Save processed images
RESULTS_OUTPUT_DIR = "results"   # Directory for saving results

# ============================================================================
# ADVANCED PARAMETERS
# ============================================================================

# Histogram matching (for consistency across different lighting conditions)
USE_HISTOGRAM_MATCHING = False
REFERENCE_IMAGE_PATH = ""  # Path to reference image for histogram matching

# Multi-scale processing
USE_MULTI_SCALE = False
SCALE_FACTORS = [0.5, 1.0, 1.5]  # Scale factors for processing

# GPU acceleration (if available)
USE_CUDA = False                 # Enable CUDA acceleration
CUDA_DEVICE_ID = 0              # GPU device ID to use

# ============================================================================
# FUNCTION TO APPLY CONFIGURATION
# ============================================================================

import numpy as np
import cv2

def get_copper_hsv_lower():
    """Get lower HSV threshold for copper"""
    return np.array([
        COPPER_HSV_LOWER1['H'],
        COPPER_HSV_LOWER1['S'],
        COPPER_HSV_LOWER1['V']
    ])

def get_copper_hsv_upper():
    """Get upper HSV threshold for copper"""
    return np.array([
        COPPER_HSV_UPPER1['H'],
        COPPER_HSV_UPPER1['S'],
        COPPER_HSV_UPPER1['V']
    ])

def get_copper_hsv_lower2():
    """Get lower HSV threshold for darker copper"""
    return np.array([
        COPPER_HSV_LOWER2['H'],
        COPPER_HSV_LOWER2['S'],
        COPPER_HSV_LOWER2['V']
    ])

def get_copper_hsv_upper2():
    """Get upper HSV threshold for darker copper"""
    return np.array([
        COPPER_HSV_UPPER2['H'],
        COPPER_HSV_UPPER2['S'],
        COPPER_HSV_UPPER2['V']
    ])

def get_morphological_kernel(kernel_type='open'):
    """Get morphological kernel based on type"""
    if kernel_type == 'open':
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, OPENING_KERNEL_SIZE)
    elif kernel_type == 'close':
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, CLOSING_KERNEL_SIZE)
    elif kernel_type == 'dilate':
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, DILATION_KERNEL_SIZE)
    else:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

# Preset for bright, well-lit environments
PRESET_BRIGHT = {
    'CLAHE_CLIP_LIMIT': 1.5,
    'MIN_CONTOUR_AREA': 500,
    'BILATERAL_COLOR_SIGMA': 50
}

# Preset for dark/indoor environments
PRESET_DARK = {
    'CLAHE_CLIP_LIMIT': 3.0,
    'MIN_CONTOUR_AREA': 200,
    'BILATERAL_COLOR_SIGMA': 100
}

# Preset for high precision (slower processing)
PRESET_PRECISION = {
    'OPENING_ITERATIONS': 3,
    'CLOSING_ITERATIONS': 3,
    'MIN_CONTOUR_AREA': 300,
    'USE_MULTI_SCALE': True
}

# Preset for speed (faster processing)
PRESET_SPEED = {
    'STANDARD_RESIZE': (320, 240),
    'OPENING_ITERATIONS': 1,
    'CLOSING_ITERATIONS': 1,
    'MIN_CONTOUR_AREA': 100
}
