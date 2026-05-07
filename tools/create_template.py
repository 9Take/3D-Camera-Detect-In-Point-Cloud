import numpy as np
import cv2
import open3d as o3d
import os
import math
import argparse
import yaml
import sys

# ให้ Python มองเห็นโฟลเดอร์หลักของโปรเจกต์
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from communication.realsense import DepthCamera

def load_config():
    # โหลดคอนฟิกจากโฟลเดอร์หลัก
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = load_config()
resolution_width = config['camera']['resolution_width']
resolution_height = config['camera']['resolution_height']

def get_params():
    parser = argparse.ArgumentParser(description='Create Template for Heat Exchanger')
    parser.add_argument('-t', '--target', type=str, default=None, help='ชื่อเป้าหมาย (default: A)')
    args = parser.parse_args()
    
    target_name = args.target
    if target_name is None:
        target_name = input("กรุณาป้อนชื่อเป้าหมาย [A, B, C, etc.]: ").strip()
        if not target_name: target_name = "A"
        
    save_dir = config['paths']['template_dir']
    log_dir = config['paths']['save_dir']
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    return save_dir, log_dir, target_name

SAVE_DIR, LOG_DIR, CURRENT_TARGET_NAME = get_params()

app_state = 0 # 0: Live View, 1: Frozen Annotation, 2: Tracking
polygon_points = []
template_patch = None
exact_target_pixel_manual = None
target_offset = None
frozen_color = None
frozen_gray = None

def draw_shape_callback(event, x, y, flags, param):
    global polygon_points, exact_target_pixel_manual, app_state
    if app_state != 1: return
        
    if event == cv2.EVENT_LBUTTONDOWN:
        polygon_points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        exact_target_pixel_manual = (x, y)

def rotation_matrix_to_euler_angles(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x, y, z = math.atan2(R[2, 1], R[2, 2]), math.atan2(-R[2, 0], sy), math.atan2(R[1, 0], R[0, 0])
    else:
        x, y, z = math.atan2(-R[1, 2], R[1, 1]), math.atan2(-R[2, 0], sy), 0
    return np.rad2deg([x, y, z])

def main():
    global polygon_points, template_patch, exact_target_pixel_manual, target_offset
    global app_state, frozen_color, frozen_gray
    
    print("\n[INIT] Initializing RealSense Camera...")
    cam = DepthCamera(resolution_width, resolution_height)
    
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Frame", 848, 480)
    cv2.setMouseCallback("Frame", draw_shape_callback)

    sift = cv2.SIFT_create()
    kp_template, des_template = None, None

    print(f"\n--- PHASE 1: TARGET CAPTURE ---")
    print("1. กด 'SPACEBAR' เพื่อแช่ภาพ")
    print("2. คลิกซ้ายวาดกรอบ และคลิกขวาระบุจุด Target")

    while True:
        if app_state == 0 or app_state == 2:
            ret , depth_raw_frame, color_raw_frame = cam.get_raw_frame()
            if not ret: continue
            color_frame = np.asanyarray(color_raw_frame.get_data())
            depth_frame = np.asanyarray(depth_raw_frame.get_data())
            gray_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)

        if app_state == 0:
            display_frame = color_frame.copy()
            cv2.putText(display_frame, "Aim camera -> Press SPACEBAR to Capture", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Frame", display_frame)
            key = cv2.waitKey(33) & 0xFF
            
            if key == 32: # SPACEBAR
                frozen_color = color_frame.copy()
                frozen_gray = gray_frame.copy()
                polygon_points.clear()
                exact_target_pixel_manual = None
                app_state = 1
                
        elif app_state == 1:
            display_frame = frozen_color.copy()
            if len(polygon_points) > 0:
                pts = np.array(polygon_points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(display_frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                for pt in polygon_points: cv2.circle(display_frame, pt, 4, (0, 255, 0), -1)

            if exact_target_pixel_manual is not None:
                cv2.circle(display_frame, exact_target_pixel_manual, 5, (0, 0, 255), -1)

            cv2.putText(display_frame, "L-Click=Box | R-Click=Target | 's'=Save | 'r'=Retake | 'c'=Clear", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            cv2.imshow("Frame", display_frame)
            key = cv2.waitKey(33) & 0xFF

            if key == ord('c'):
                polygon_points.clear()
                exact_target_pixel_manual = None
            elif key == ord('r'):
                app_state = 0
            elif key == ord('s') and len(polygon_points) > 2:
                x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(np.array(polygon_points))
                template_patch = frozen_gray[y_rect:y_rect+h_rect, x_rect:x_rect+w_rect]
                
                # SAVE ไฟล์เข้า data/templates
                cv2.imwrite(os.path.join(SAVE_DIR, f"{CURRENT_TARGET_NAME}_template.png"), template_patch)
                
                if exact_target_pixel_manual is not None:
                    target_offset = (exact_target_pixel_manual[0] - x_rect, exact_target_pixel_manual[1] - y_rect)
                else:
                    target_offset = (w_rect // 2, h_rect // 2)
                    
                with open(os.path.join(SAVE_DIR, f"{CURRENT_TARGET_NAME}_offset.txt"), "w") as f:
                    f.write(f"{target_offset[0]},{target_offset[1]}")

                kp_template, des_template = sift.detectAndCompute(template_patch, None)
                print(f"\n[SUCCESS] Template '{CURRENT_TARGET_NAME}' Saved to {SAVE_DIR}!")
                app_state = 2

        elif app_state == 2:
            display_frame = color_frame.copy()
            target_pixel = None
            th, tw = template_patch.shape
            kp_frame, des_frame = sift.detectAndCompute(gray_frame, None)
            
            if des_template is not None and des_frame is not None and len(des_template) > 2 and len(des_frame) > 2:
                flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
                try:
                    matches = flann.knnMatch(des_template, des_frame, k=2)
                    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
                    
                    if len(good_matches) > 10:
                        src_pts = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                        
                        if M is not None:
                            pts = np.float32([[0, 0], [0, th - 1], [tw - 1, th - 1], [tw - 1, 0]]).reshape(-1, 1, 2)
                            dst = cv2.perspectiveTransform(pts, M)
                            target_pt_dst = cv2.perspectiveTransform(np.float32([[[target_offset[0], target_offset[1]]]]), M)
                            target_pixel = (int(target_pt_dst[0][0][0]), int(target_pt_dst[0][0][1]))
                            
                            display_frame = cv2.polylines(display_frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
                            cv2.circle(display_frame, target_pixel, 5, (0, 0, 255), -1)
                except Exception: pass

            cv2.putText(display_frame, f"Press 'q' to Extract 3D or 'ESC' to exit.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow("Frame", display_frame)
            key = cv2.waitKey(33) & 0xFF
            
            if key == 27: # ESC
                break
            
            if key == ord('q') and target_pixel is not None:
                # ทดสอบหาพิกัด 3D แบบคร่าวๆ (ไม่ต้องรวม 70 เฟรม)
                u, v = target_pixel
                z = depth_frame[v, u] * cam.get_depth_scale()
                intrinsics = depth_raw_frame.profile.as_video_stream_profile().intrinsics
                
                if z > 0:
                    x = (u - intrinsics.ppx) * z / intrinsics.fx
                    y = (v - intrinsics.ppy) * z / intrinsics.fy
                    print(f"\n[TEST DATA] Target {CURRENT_TARGET_NAME}: X={x:.4f}, Y={-y:.4f}, Z={-z:.4f}")
                    break
                else:
                    print("\n[WARNING] Invalid Depth at target pixel.")

    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()