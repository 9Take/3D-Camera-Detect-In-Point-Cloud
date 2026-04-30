"""
Copper Pipe Detection using RealSense 435i Camera
Captures live video and detects copper pipes in real-time
"""

import numpy as np
import cv2
import pyrealsense2 as rs

# HSV range for copper color
COPPER_HSV_LOWER1 = np.array([0, 50, 50])
COPPER_HSV_UPPER1 = np.array([25, 255, 255])
COPPER_HSV_LOWER2 = np.array([0, 30, 30])
COPPER_HSV_UPPER2 = np.array([30, 255, 200])

# Preprocessing parameters
BLUR_KERNEL = (7, 7)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

# Global variables for ROI selection
roi_selection_mode = True
roi_start = None
roi_end = None
current_frame = None


def mouse_callback_roi(event, x, y, flags, param):
    """Mouse callback for ROI selection in camera feed"""
    global roi_start, roi_end, roi_selection_mode
    
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_start = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and roi_start:
        roi_end = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        if roi_start and roi_end:
            roi_selection_mode = False


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
    """Apply morphological operations"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_large, iterations=2)
    return cv2.morphologyEx(closed, cv2.MORPH_DILATE, kernel, iterations=1)


def apply_distance_transform(mask):
    """Apply distance transform"""
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    dist_transform_normalized = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX)
    return dist_transform, np.uint8(dist_transform_normalized)


def detect_contours(mask):
    """Detect contours in mask"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > 300:
            valid_contours.append(contour)
    return valid_contours


def process_frame(frame, roi=None):
    """Process a frame for copper pipe detection"""
    if roi:
        x1, y1, x2, y2 = roi
        frame = frame[y1:y2, x1:x2]
    
    preprocessed = preprocess_image(frame)
    copper_mask, _ = apply_hsv_mask(preprocessed)
    cleaned_mask = apply_morphological_operations(copper_mask)
    contours = detect_contours(cleaned_mask)
    
    return preprocessed, copper_mask, cleaned_mask, contours


def initialize_camera():
    """Initialize RealSense camera"""
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Configure streams
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    
    profile = pipeline.start(config)
    print("\n===== RealSense Camera Initialized =====")
    print("Resolution: 1280x720")
    print("FPS: 30")
    print("Streams: RGB + Depth")
    print("=" * 40)
    
    return pipeline


def main():
    """Main function for real-time detection"""
    global roi_selection_mode, roi_start, roi_end, current_frame
    
    print("\n===== COPPER PIPE DETECTION - RealSense 435i =====")
    print("\nMode Selection:")
    print("[1] ROI Selection Mode - Click & Drag to select region")
    print("[2] Full Frame Mode - Process entire frame")
    print("[3] Batch Processing - Process with existing ROI")
    
    mode = input("\nSelect mode (1/2/3): ").strip()
    
    # Initialize camera
    try:
        pipeline = initialize_camera()
    except Exception as e:
        print("ERROR: Failed to initialize camera: {}".format(str(e)))
        print("Ensure RealSense camera is connected and pyrealsense2 is installed")
        return
    
    selected_roi = None
    processing = False
    
    try:
        while True:
            # Get frames
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                continue
            
            frame = np.asarray(color_frame.get_data())
            current_frame = frame.copy()
            
            # Mode 1: ROI Selection
            if mode == '1' and roi_selection_mode:
                display = frame.copy()
                
                if roi_start and roi_end:
                    x1, y1 = roi_start
                    x2, y2 = roi_end
                    cv2.rectangle(display, (min(x1, x2), min(y1, y2)), 
                                (max(x1, x2), max(y1, y2)), (0, 255, 255), 2)
                    cv2.putText(display, "Selected - Press SPACE to process", 
                              (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(display, "Click and drag to select ROI", 
                              (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 0), 2)
                
                cv2.imshow("RealSense - ROI Selection", display)
                cv2.setMouseCallback("RealSense - ROI Selection", mouse_callback_roi)
                
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == 32 and roi_start and roi_end:  # SPACE
                    x1, y1 = min(roi_start[0], roi_end[0]), min(roi_start[1], roi_end[1])
                    x2, y2 = max(roi_start[0], roi_end[0]), max(roi_start[1], roi_end[1])
                    selected_roi = (x1, y1, x2, y2)
                    roi_selection_mode = False
                    processing = True
            
            # Mode 2 & 3: Processing
            elif mode in ['2', '3'] or processing:
                preprocessed, copper_mask, cleaned_mask, contours = process_frame(frame, selected_roi)
                
                # Create visualization
                result = frame.copy()
                cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
                
                # Add info
                cv2.putText(result, "Pipes detected: {}".format(len(contours)), 
                          (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Create side panel with mask
                mask_colored = cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2BGR)
                display = np.hstack([result, mask_colored])
                
                cv2.imshow("RealSense - Copper Detection", display)
                
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord('r') and mode == '1':  # Reset ROI
                    roi_start = None
                    roi_end = None
                    selected_roi = None
                    roi_selection_mode = True
                    processing = False
    
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("\nCamera closed")


if __name__ == "__main__":
    main()
