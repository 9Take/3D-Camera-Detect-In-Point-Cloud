import numpy as np
import cv2
import os
import glob
from scipy import ndimage

# Path to pic folder
PIC_DIR = "pic"

# HSV range for copper color (reddish-brown)
# Copper has hue around 0-15 (red) and 15-30 (orange-brown)
COPPER_HSV_LOWER1 = np.array([0, 50, 50])      # Lower red hue
COPPER_HSV_UPPER1 = np.array([25, 255, 255])   # Upper red-orange hue

# For darker copper tones
COPPER_HSV_LOWER2 = np.array([0, 30, 30])
COPPER_HSV_UPPER2 = np.array([30, 255, 200])

# Preprocessing parameters
BLUR_KERNEL = (7, 7)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

# Global variables for mouse interaction
mouse_drawing = False
mouse_start = None
mouse_end = None
current_image = None
current_hsv = None

def load_images_from_folder(pic_dir):
    """Load all images from pic folder"""
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(glob.glob(os.path.join(pic_dir, ext)))
    return sorted(image_files)

def preprocess_image(image):
    """
    Preprocess image to handle reflections and improve copper detection
    - Apply bilateral filtering to reduce noise while preserving edges
    - Use CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance contrast
    - Apply Gaussian blur for smoothing
    """
    # Apply bilateral filter to reduce noise while preserving edges (handles reflections)
    bilateral = cv2.bilateralFilter(image, 9, 75, 75)
    
    # Convert to LAB color space for better contrast enhancement
    lab = cv2.cvtColor(bilateral, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel for better contrast
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    l = clahe.apply(l)
    
    # Merge back
    lab = cv2.merge([l, a, b])
    preprocessed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Apply Gaussian blur for smoothing
    preprocessed = cv2.GaussianBlur(preprocessed, BLUR_KERNEL, 0)
    
    return preprocessed

def mouse_callback(event, x, y, flags, param):
    """Mouse callback for drawing rectangle"""
    global mouse_drawing, mouse_start, mouse_end, current_image
    
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

def apply_hsv_mask(image):
    """Apply HSV mask to isolate copper color"""
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create masks for copper color ranges
    mask1 = cv2.inRange(hsv, COPPER_HSV_LOWER1, COPPER_HSV_UPPER1)
    mask2 = cv2.inRange(hsv, COPPER_HSV_LOWER2, COPPER_HSV_UPPER2)
    
    # Combine masks
    copper_mask = cv2.bitwise_or(mask1, mask2)
    
    return copper_mask, hsv

def apply_edge_detection(mask):
    """Apply Canny edge detection"""
    # Blur to reduce noise
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    return edges

def apply_morphological_operations(mask):
    """
    Clean up mask with morphological operations
    - Remove small noise
    - Fill small holes
    - Enhance connectivity
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    
    # Opening to remove small noise
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Closing to fill small holes
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_large, iterations=2)
    
    # Dilation to enhance connectivity
    dilated = cv2.morphologyEx(closed, cv2.MORPH_DILATE, kernel, iterations=1)
    
    return dilated

def apply_distance_transform(mask):
    """
    Apply Distance Transform to separate overlapping pipes
    Returns both distance map and normalized version
    """
    # Compute the distance transform
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    
    # Normalize for visualization
    dist_transform_normalized = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX)
    dist_transform_normalized = np.uint8(dist_transform_normalized)
    
    return dist_transform, dist_transform_normalized

def apply_watershed(image, mask, dist_transform):
    """
    Apply Watershed algorithm to separate overlapping pipes
    """
    # Find peaks in distance transform (these are centers of pipes)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # Find background
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_bg = cv2.dilate(mask, kernel, iterations=2)
    
    # Find unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Label sure foreground
    _, markers = cv2.connectedComponents(sure_fg)
    
    # Add 1 to all labels so sure background is not 0, but 1
    markers = markers + 1
    
    # Mark unknown region as 0
    markers[unknown == 255] = 0
    
    # Apply watershed
    try:
        markers = cv2.watershed(image, markers)
    except Exception as e:
        print("Watershed error: {}".format(e))
        return markers, sure_fg, sure_bg
    
    return markers, sure_fg, sure_bg

def detect_contours(mask):
    """Detect contours in mask"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter by area
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 300:  # Minimum area threshold
            valid_contours.append(contour)
    
    return valid_contours

def calculate_pipe_center(contour):
    """
    Calculate center point of a pipe (centroid)
    Returns: (center_x, center_y, radius)
    """
    # Calculate moment to get centroid
    M = cv2.moments(contour)
    
    if M["m00"] == 0:
        return None, None, None
    
    # Centroid coordinates
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    # Fit circle to get radius (pipe diameter)
    (circle_cx, circle_cy), radius = cv2.minEnclosingCircle(contour)
    
    return cx, cy, int(radius)

def find_primary_pipe_center(contours):
    """
    Find the center of the largest/primary pipe
    Returns: (cx, cy, radius) of the main pipe
    """
    if not contours:
        return None, None, None
    
    # Find largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)
    
    return calculate_pipe_center(largest_contour)

def get_all_pipe_centers(contours):
    """
    Get center coordinates of all detected pipes
    Returns: List of [(cx, cy, radius), ...]
    """
    centers = []
    
    for contour in contours:
        cx, cy, radius = calculate_pipe_center(contour)
        if cx is not None:
            centers.append({
                'x': cx,
                'y': cy,
                'radius': radius,
                'area': cv2.contourArea(contour),
                'perimeter': cv2.arcLength(contour, True)
            })
    
    # Sort by area (largest first)
    centers.sort(key=lambda p: p['area'], reverse=True)
    
    return centers

def get_target_point_for_arm(centers, selection_mode='largest'):
    """
    Get the target point for robotic arm
    
    Parameters:
    -----------
    centers : list
        List of pipe center dictionaries
    selection_mode : str
        'largest' - target the largest pipe
        'closest_to_center' - target pipe closest to image center
        'all' - return all pipe positions
    
    Returns:
    --------
    dict or list : Target point(s) with coordinates and size info
    """
    if not centers:
        return None
    
    if selection_mode == 'largest':
        return centers[0]  # Already sorted by area
    
    elif selection_mode == 'closest_to_center':
        # Find pipe closest to image center (assuming image size known)
        # This requires image dimensions
        min_distance = float('inf')
        closest_pipe = None
        
        for pipe in centers:
            # Calculate distance from center
            distance = np.sqrt(pipe['x']**2 + pipe['y']**2)
            if distance < min_distance:
                min_distance = distance
                closest_pipe = pipe
        
        return closest_pipe
    
    elif selection_mode == 'all':
        return centers
    
    else:
        return centers[0]

def visualize_results(image_name, original, copper_mask, edges, cleaned_mask, contours, 
                      dist_transform_norm, watershed_markers):
    """
    Visualize all processing steps including Watershed and Distance Transform
    Shows pipe centers for robotic arm targeting
    """
    global mouse_drawing, mouse_start, mouse_end, current_image, current_hsv
    
    h, w = original.shape[:2]
    
    # Create colored versions for display
    mask_colored = cv2.cvtColor(copper_mask, cv2.COLOR_GRAY2BGR)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cleaned_colored = cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2BGR)
    dist_colored = cv2.cvtColor(dist_transform_norm, cv2.COLOR_GRAY2BGR)
    
    # Create watershed visualization
    watershed_vis = original.copy()
    if watershed_markers is not None and len(np.unique(watershed_markers)) > 1:
        # Color the watershed regions
        markers_colored = np.uint8(np.clip(watershed_markers, 0, 255))
        markers_colored = cv2.applyColorMap(markers_colored, cv2.COLORMAP_JET)
        watershed_vis = cv2.addWeighted(original, 0.5, markers_colored, 0.5, 0)
    
    # Draw contours on original image
    contour_img = original.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    
    # Calculate and draw pipe centers for robotic arm
    centers = get_all_pipe_centers(contours)
    
    # Draw centers and targeting info
    for i, center_data in enumerate(centers):
        cx, cy = center_data['x'], center_data['y']
        radius = center_data['radius']
        
        # Draw circle at center
        cv2.circle(contour_img, (cx, cy), 5, (255, 0, 0), -1)  # Center point (blue)
        cv2.circle(contour_img, (cx, cy), radius, (255, 255, 0), 2)  # Pipe outline (cyan)
        
        # Label the pipe center
        label = "P{}:({},{})".format(i+1, cx, cy)
        cv2.putText(contour_img, label, (cx+10, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 0, 0), 1)
    
    # Add text labels
    cv2.putText(original, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(mask_colored, "HSV Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(edges_colored, "Edge Detection", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(cleaned_colored, "Cleaned Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(dist_colored, "Distance Transform", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    num_contours = len(contours)
    cv2.putText(contour_img, "Detected: {} pipes".format(num_contours), (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(watershed_vis, "Watershed Separation", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Combine into grid (3 rows x 2 columns)
    row1 = np.hstack([original, mask_colored])
    row2 = np.hstack([edges_colored, cleaned_colored])
    row3 = np.hstack([dist_colored, contour_img])
    
    combined = np.vstack([row1, row2, row3])
    
    # Store current image for processing
    current_image = original.copy()
    current_hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    
    # Display and set mouse callback
    title = "Copper Pipe Detection - {}".format(image_name)
    cv2.imshow(title, combined)
    cv2.setMouseCallback(title, mouse_callback)
    
    # Instructions
    cv2.imshow("Instructions", create_instruction_image())
    
    # Show watershed separately if markers exist
    if watershed_markers is not None:
        cv2.imshow("Watershed Segmentation", watershed_vis)
    
    return contour_img, centers

def create_instruction_image():
    """Create instruction image with usage guide"""
    img = np.ones((250, 500, 3), dtype=np.uint8) * 255
    y_offset = 30
    cv2.putText(img, "COPPER PIPE DETECTION - Instructions", (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    y_offset += 40
    cv2.putText(img, "Click & Drag: Select Region of Interest (ROI)", (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    y_offset += 30
    cv2.putText(img, "Press 'a': Analyze selected region with full pipeline", (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    y_offset += 30
    cv2.putText(img, "Press 'q': Skip to next image", (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    y_offset += 30
    cv2.putText(img, "Press 'ESC': Quit application", (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    y_offset += 40
    cv2.putText(img, "Pipeline: Preprocessing -> HSV Segmentation -> Morphology", (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 128, 0), 1)
    y_offset += 25
    cv2.putText(img, "-> Distance Transform -> Watershed -> Edge Detection", (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 128, 0), 1)
    return img

def process_selected_region(original, roi):
    """
    Process the selected region with complete pipeline:
    1. Preprocessing (reflection handling)
    2. HSV segmentation
    3. Morphological operations
    4. Distance Transform
    5. Watershed algorithm
    6. Edge detection
    7. Contour detection
    """
    global mouse_start, mouse_end
    
    if not mouse_start or not mouse_end:
        print("No region selected")
        return None, None, None, None, None, None, None
    
    # Get bounding box
    x1, y1 = min(mouse_start[0], mouse_end[0]), min(mouse_start[1], mouse_end[1])
    x2, y2 = max(mouse_start[0], mouse_end[0]), max(mouse_start[1], mouse_end[1])
    
    print("\nAnalyzing selected region: ({},{}) to ({},{})".format(x1, y1, x2, y2))
    print("Region size: {}x{} pixels".format(x2-x1, y2-y1))
    
    # Step 1: Preprocess to handle reflections
    print("\n[1/7] Preprocessing image (reflection handling)...")
    preprocessed = preprocess_image(roi)
    
    # Step 2: Apply HSV mask
    print("[2/7] Applying HSV color segmentation...")
    copper_mask, hsv = apply_hsv_mask(preprocessed)
    
    # Step 3: Apply morphological operations
    print("[3/7] Cleaning mask with morphological operations...")
    cleaned_mask = apply_morphological_operations(copper_mask)
    
    # Step 4: Distance Transform
    print("[4/7] Computing Distance Transform...")
    dist_transform, dist_transform_norm = apply_distance_transform(cleaned_mask)
    
    # Step 5: Watershed algorithm for separating overlapping pipes
    print("[5/7] Applying Watershed algorithm for overlapping pipe separation...")
    watershed_markers, sure_fg, sure_bg = apply_watershed(preprocessed, cleaned_mask, dist_transform)
    
    # Step 6: Edge detection
    print("[6/7] Applying edge detection...")
    edges = apply_edge_detection(cleaned_mask)
    
    # Step 7: Detect contours
    print("[7/7] Detecting contours...")
    roi_contours = detect_contours(cleaned_mask)
    
    print("Processing complete! Found {} pipes".format(len(roi_contours)))
    
    return (preprocessed, copper_mask, cleaned_mask, dist_transform_norm, 
            watershed_markers, edges, roi_contours)

def print_statistics(contours, copper_mask, roi_size=None):
    """
    Print detailed statistics about detected copper pipes
    Including center coordinates for robotic arm targeting
    """
    total_copper_pixels = np.sum(copper_mask > 0)
    centers = get_all_pipe_centers(contours)
    
    print("\n" + "="*70)
    print("DETECTION STATISTICS & ROBOTIC ARM TARGETING")
    print("="*70)
    print("Total copper pixels detected: {}".format(total_copper_pixels))
    coverage = 100*total_copper_pixels/(copper_mask.shape[0]*copper_mask.shape[1])
    print("Copper coverage: {:.2f}%".format(coverage))
    print("Number of pipes detected: {}".format(len(contours)))
    print("-"*70)
    
    if len(contours) == 0:
        print("No pipes detected in this region.")
        print("="*70)
        return
    
    # Get primary target for arm
    primary_target = get_target_point_for_arm(centers, selection_mode='largest')
    
    print("\nPRIMARY TARGET FOR ROBOTIC ARM:")
    print("-"*70)
    print("Pipe Position: X={}, Y={}".format(primary_target['x'], primary_target['y']))
    print("Pipe Radius: {} pixels (Diameter: {} pixels)".format(
        primary_target['radius'], primary_target['radius']*2))
    print("Area: {:.1f} pixels".format(primary_target['area']))
    print("-"*70)
    
    # Print all pipe positions
    print("\nALL DETECTED PIPES (Sorted by size):")
    print("-"*70)
    for i, pipe_data in enumerate(centers):
        print("\nPipe {}:".format(i+1))
        print("  Target Point: X={}, Y={}".format(pipe_data['x'], pipe_data['y']))
        print("  Pipe Diameter: {} pixels (Radius: {})".format(
            pipe_data['radius']*2, pipe_data['radius']))
        print("  Area: {:.1f} pixels".format(pipe_data['area']))
        print("  Perimeter: {:.1f} pixels".format(pipe_data['perimeter']))
    
    print("="*70)
    
    # Detailed contour analysis
    print("\nDETAILED PIPE ANALYSIS:")
    print("-"*70)
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)
        
        print("\nPipe {}:".format(i+1))
        print("  Centroid: ({}, {})".format(centers[i]['x'], centers[i]['y']))
        print("  Area: {:.1f} pixels".format(area))
        print("  Perimeter: {:.1f} pixels".format(perimeter))
        print("  Bounding box: {}x{} pixels".format(w, h))
        print("  Approximate diameter: {:.1f} pixels".format(centers[i]['radius']*2))
        
        # Calculate circularity (how circular is the object)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
            print("  Circularity: {:.3f} (1.0 = perfect circle)".format(circularity))
        
        # Hull-based analysis
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area
            print("  Solidity: {:.3f}".format(solidity))
    
    print("="*70)

def main():
    """
    Main function for copper pipe detection
    Workflow:
    1. Load images from 'pic' folder
    2. Display image and allow user to select ROI
    3. Process ROI through complete pipeline
    4. Display results and statistics
    """
    global mouse_drawing, mouse_start, mouse_end, current_image
    
    print("\n" + "="*70)
    print("COPPER PIPE DETECTION SYSTEM - RealSense 435i")
    print("="*70)
    print("Loading images from pic folder...")
    
    image_files = load_images_from_folder(PIC_DIR)
    
    if not image_files:
        print("ERROR: No images found in '{}' folder".format(PIC_DIR))
        print("Please place images in the 'pic' folder and try again.")
        return
    
    print("Found {} image(s)\n".format(len(image_files)))
    
    for idx, img_path in enumerate(image_files):
        image_name = os.path.basename(img_path)
        print("\n" + "="*70)
        print("[Image {}/{}] Processing: {}".format(idx+1, len(image_files), image_name))
        print("="*70)
        
        # Load image
        original = cv2.imread(img_path)
        if original is None:
            print("ERROR: Failed to load {}".format(img_path))
            continue
        
        # Resize to standard size for processing
        original = cv2.resize(original, (640, 480))
        current_image = original.copy()
        mouse_drawing = False
        mouse_start = None
        mouse_end = None
        
        # Show only the original image and let user select ROI
        win_name = "Select Region of Interest (ROI) - {}".format(image_name)
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win_name, mouse_callback)
        
        print("\nInstructions:")
        print("  - Click and drag to select a region of interest")
        print("  - Release mouse when done selecting")
        print("  - Press 'q' to skip to next image")
        print("  - Press 'ESC' to quit application")
        
        roi_selected = False
        while True:
            display_img = current_image.copy()
            
            # Draw rectangle while dragging
            if mouse_drawing and mouse_start and mouse_end:
                cv2.rectangle(display_img, mouse_start, mouse_end, (0, 255, 255), 2)
                cv2.putText(display_img, "Selecting...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 255, 255), 2)
            # Draw fixed rectangle after release
            elif mouse_start and mouse_end:
                cv2.rectangle(display_img, mouse_start, mouse_end, (0, 255, 0), 2)
                cv2.putText(display_img, "Selected - Press ENTER to analyze or drag to reselect", 
                           (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display_img, "Click and drag to select ROI", (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 0), 2)
            
            cv2.imshow(win_name, display_img)
            key = cv2.waitKey(20) & 0xFF
            
            if key == 27:  # ESC
                print("\nQuitting application...")
                cv2.destroyAllWindows()
                return
            
            if key == ord('q'):  # Skip to next image
                print("Skipping to next image...")
                break
            
            if key == 13 and mouse_start and mouse_end:  # ENTER - process selected region
                roi_selected = True
                break
        
        if roi_selected and mouse_start and mouse_end:
            cv2.destroyWindow(win_name)
            
            # Extract ROI
            x1, y1 = min(mouse_start[0], mouse_end[0]), min(mouse_start[1], mouse_end[1])
            x2, y2 = max(mouse_start[0], mouse_end[0]), max(mouse_start[1], mouse_end[1])
            roi = original[y1:y2, x1:x2].copy()
            
            print("\nROI selected: ({},{}) to ({},{}), Size: {}x{}".format(x1, y1, x2, y2, x2-x1, y2-y1))
            
            # Process ROI through complete pipeline
            result = process_selected_region(original, roi)
            preprocessed, copper_mask, cleaned_mask, dist_transform_norm, \
                watershed_markers, edges, roi_contours = result
            
            if preprocessed is not None:
                # Visualize results
                print("\nDisplaying results...")
                contour_img, centers = visualize_results(image_name, roi, copper_mask, edges, cleaned_mask, 
                                roi_contours, dist_transform_norm, watershed_markers)
                
                # Print detailed statistics including arm target points
                print_statistics(roi_contours, copper_mask, roi.shape[:2])
                
                # Display arm target information
                if centers:
                    primary_target = get_target_point_for_arm(centers, selection_mode='largest')
                    print("\nARM TARGET POINT:")
                    print("Position: ({}, {})".format(primary_target['x'], primary_target['y']))
                    print("Diameter: {} pixels".format(primary_target['radius']*2))
                
                print("\nPress any key to continue to next image, or ESC to quit.")
                k2 = cv2.waitKey(0) & 0xFF
                
                if k2 == 27:
                    print("Quitting...")
                    cv2.destroyAllWindows()
                    return
        else:
            cv2.destroyWindow(win_name)
    
    cv2.destroyAllWindows()
    print("\n" + "="*70)
    print("Processing completed successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
