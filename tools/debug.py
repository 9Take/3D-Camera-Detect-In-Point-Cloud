import cv2
import numpy as np
import yaml
import sys
import os

# เพิ่ม Path ให้มองเห็นโฟลเดอร์ hardware/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from communication.realsense import DepthCamera

def load_config():
    # สมมติว่าไฟล์รันอยู่ที่ root project
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    res_w = config['camera']['resolution_width']
    res_h = config['camera']['resolution_height']

    print(f"[DEBUG] Initializing RealSense Camera ({res_w}x{res_h})...")
    cam = DepthCamera(res_w, res_h)
    
    print("[DEBUG] Camera is LIVE. Press 'ESC' or 'q' to close window.")

    try:
        while True:
            ret, depth_raw, color_raw = cam.get_raw_frame()
            if not ret: 
                continue

            # ดึงภาพสีและแปลงเป็น NumPy Array
            color_frame = np.asanyarray(color_raw.get_data())
            
            # (Optional) ดึงภาพ Depth มาดูด้วยก็ได้
            depth_frame = np.asanyarray(depth_raw.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=0.03), cv2.COLORMAP_JET)

            # แสดงผล
            cv2.imshow("RealSense Debug - RGB", color_frame)
            cv2.imshow("RealSense Debug - Depth", depth_colormap)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'): # กด ESC หรือ q เพื่อออก
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()
        print("[DEBUG] Camera released.")

if __name__ == '__main__':
    main()