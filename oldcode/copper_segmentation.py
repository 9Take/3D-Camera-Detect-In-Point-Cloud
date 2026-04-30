import numpy as np
import cv2
import os
import glob

# --- Configuration ---
resolution_width, resolution_height = (640, 480)
SAVE_DIR = "test5"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Copper color range in HSV (reddish-brown color)
COPPER_LOWER1 = np.array([0, 50, 100])
COPPER_UPPER1 = np.array([20, 255, 255])
COPPER_LOWER2 = np.array([160, 50, 100])
COPPER_UPPER2 = np.array([180, 255, 255])

# Depth range in millimeters
DEPTH_MIN = 100
DEPTH_MAX = 1000

class CopperSegmentation:
    """Class for segmenting copper pipes from RealSense color and depth images"""
    
    def __init__(self, resolution_width=640, resolution_height=480, use_camera=False):
        self.width = resolution_width
        self.height = resolution_height
        self.depth_scale = 0.001
        if use_camera:
            from realsense_depth import DepthCamera
            self.camera = DepthCamera(resolution_width, resolution_height)
            self.depth_scale = self.camera.get_depth_scale()
        else:
            self.camera = None
        
    def segment_by_color(self, color_image):
        """Segment copper pipes based on color (HSV color space)"""
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv_image, COPPER_LOWER1, COPPER_UPPER1)
        mask2 = cv2.inRange(hsv_image, COPPER_LOWER2, COPPER_UPPER2)
        color_mask = cv2.bitwise_or(mask1, mask2)
        return color_mask
    
    def segment_by_edge(self, color_image):
        """Segment copper pipes based on edge detection"""
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated_edges = cv2.dilate(edges, kernel, iterations=2)
        return dilated_edges
    
    def segment_by_depth(self, depth_image):
        """Segment based on depth range"""
        depth_mm = depth_image * self.depth_scale * 1000
        depth_mask = cv2.inRange(depth_mm.astype(np.float32), DEPTH_MIN, DEPTH_MAX)
        return depth_mask
    
    def apply_morphological_operations(self, mask, kernel_size=5, iterations=2):
        """Apply morphological operations to clean up the mask"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
        cleaned_mask = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        return cleaned_mask
    
    def detect_contours(self, mask, min_area=500):
        """Detect contours in the mask and filter by area"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = []
        filtered_mask = np.zeros_like(mask)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                valid_contours.append(contour)
                cv2.drawContours(filtered_mask, [contour], 0, 255, -1)
        
        return valid_contours, filtered_mask
    
    def extract_pipe_properties(self, contour, color_image, depth_image):
        """Extract properties of detected pipe"""
        properties = {}
        
        x, y, w, h = cv2.boundingRect(contour)
        properties['bbox'] = (x, y, w, h)
        properties['area'] = cv2.contourArea(contour)
        
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        properties['circle'] = ((cx, cy), radius)
        
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            properties['ellipse'] = ellipse
        
        mask = np.zeros_like(depth_image)
        cv2.drawContours(mask, [contour], 0, 255, -1)
        masked_depth = depth_image[mask == 255]
        if len(masked_depth) > 0:
            properties['avg_depth'] = np.mean(masked_depth) * self.depth_scale
            properties['min_depth'] = np.min(masked_depth) * self.depth_scale
            properties['max_depth'] = np.max(masked_depth) * self.depth_scale
        
        masked_color = color_image[mask == 255]
        if len(masked_color) > 0:
            properties['avg_color_bgr'] = np.mean(masked_color.reshape(-1, 3), axis=0)
        
        return properties
    
    def segment_frame(self, color_image, depth_image, use_depth=True, use_color=True, use_edge=True):
        """Complete segmentation pipeline"""
        masks = []
        
        if use_color:
            color_mask = self.segment_by_color(color_image)
            masks.append(color_mask)
        
        if use_edge:
            edge_mask = self.segment_by_edge(color_image)
            masks.append(edge_mask)
        
        if use_depth:
            depth_mask = self.segment_by_depth(depth_image)
            masks.append(depth_mask)
        
        if len(masks) == 0:
            combined_mask = np.ones((self.height, self.width), dtype=np.uint8) * 255
        elif len(masks) == 1:
            combined_mask = masks[0]
        else:
            combined_mask = masks[0]
            for mask in masks[1:]:
                combined_mask = cv2.bitwise_and(combined_mask, mask)
        
        final_mask = self.apply_morphological_operations(combined_mask, kernel_size=5, iterations=2)
        contours, final_mask = self.detect_contours(final_mask, min_area=500)
        
        properties_list = []
        for contour in contours:
            props = self.extract_pipe_properties(contour, color_image, depth_image)
            properties_list.append(props)
        
        return final_mask, contours, properties_list
    
    def visualize_segmentation(self, color_image, depth_image, segmentation_mask, 
                              contours, properties_list, save=False):
        """Visualize segmentation results"""
        vis_image = color_image.copy()
        cv2.drawContours(vis_image, contours, -1, (0, 255, 0), 2)
        
        for i, props in enumerate(properties_list):
            x, y, w, h = props['bbox']
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            (cx, cy), radius = props['circle']
            cv2.circle(vis_image, (int(cx), int(cy)), int(radius), (0, 0, 255), 2)
            
            pipe_text = "Pipe " + str(i+1)
            cv2.putText(vis_image, pipe_text, (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            if 'avg_depth' in props:
                depth_text = "Depth: {:.1f}mm".format(props['avg_depth']*1000)
                cv2.putText(vis_image, depth_text, (x, y+h+20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        mask_display = cv2.cvtColor(segmentation_mask, cv2.COLOR_GRAY2BGR)
        depth_display = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_display = cv2.cvtColor(depth_display, cv2.COLOR_GRAY2BGR)
        depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
        
        top_row = np.hstack([color_image, vis_image])
        bottom_row = np.hstack([mask_display, depth_display])
        display_image = np.vstack([top_row, bottom_row])
        
        cv2.imshow("Copper Pipe Segmentation", display_image)
        
        if save:
            filename = os.path.join(SAVE_DIR, "segmentation_{}.png".format(np.random.randint(10000)))
            cv2.imwrite(filename, display_image)
            print("Saved segmentation image to {}".format(filename))
        
        return vis_image, mask_display, depth_display
    
    def release(self):
        """Release camera resources"""
        self.camera.release()
        cv2.destroyAllWindows()


def main_from_images(pic_dir="pic"):
    """Load images from pic folder and perform segmentation"""
    image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for pattern in image_patterns:
        image_files.extend(glob.glob(os.path.join(pic_dir, pattern)))
    
    if not image_files:
        print("No images found in {}".format(pic_dir))
        return
    
    segmenter = CopperSegmentation(resolution_width, resolution_height, use_camera=False)
    
    print("Found {} image(s)".format(len(image_files)))
    print("Press any key to continue to next image, 'q' to quit")
    
    for img_path in sorted(image_files):
        print("\nProcessing: {}".format(os.path.basename(img_path)))
        
        color_image = cv2.imread(img_path)
        if color_image is None:
            print("Failed to load {}".format(img_path))
            continue
        
        color_image = cv2.resize(color_image, (segmenter.width, segmenter.height))
        depth_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        depth_image = depth_image.astype(np.uint16)
        
        segmentation_mask, contours, properties_list = segmenter.segment_frame(
            color_image, depth_image, use_depth=False, use_color=True, use_edge=True
        )
        
        vis_image, mask_display, _ = segmenter.visualize_segmentation(
            color_image, depth_image, segmentation_mask, contours, properties_list, save=False
        )
        
        if properties_list:
            print("Detected {} copper pipe(s)".format(len(contours)))
            for i, props in enumerate(properties_list):
                print("  Pipe {}:".format(i+1))
                print("    Area: {:.1f} pixels".format(props['area']))
                print("    Circle radius: {:.1f} pixels".format(props['circle'][1]))
                if 'avg_color_bgr' in props:
                    b, g, r = props['avg_color_bgr']
                    print("    Color (BGR): ({:.1f}, {:.1f}, {:.1f})".format(b, g, r))
        else:
            print("No copper pipes detected")
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    print("Image processing completed")


def main():
    """Main function for real-time copper pipe segmentation"""
    segmenter = CopperSegmentation(resolution_width, resolution_height)
    
    print("Starting copper pipe segmentation...")
    print("Press 'q' to quit, 's' to save segmentation result")
    
    while True:
        ret, depth_image, color_image = segmenter.camera.get_frame()
        if not ret:
            continue
        
        segmentation_mask, contours, properties_list = segmenter.segment_frame(
            color_image, depth_image, use_depth=True, use_color=True, use_edge=True
        )
        
        vis_image, mask_display, depth_display = segmenter.visualize_segmentation(
            color_image, depth_image, segmentation_mask, contours, properties_list, save=False
        )
        
        if properties_list:
            print("\nDetected {} copper pipe(s)".format(len(contours)))
            for i, props in enumerate(properties_list):
                print("  Pipe {}:".format(i+1))
                print("    Area: {:.1f} pixels".format(props['area']))
                print("    Circle radius: {:.1f} pixels".format(props['circle'][1]))
                if 'avg_depth' in props:
                    print("    Avg depth: {:.1f} mm".format(props['avg_depth']*1000))
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            segmenter.visualize_segmentation(
                color_image, depth_image, segmentation_mask, contours, properties_list, save=True
            )
    
    segmenter.release()
    print("Segmentation completed")


if __name__ == "__main__":
    main_from_images()
