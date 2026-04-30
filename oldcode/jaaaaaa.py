import numpy as np
import pyrealsense2 as rs
import cv2
import os
from realsense_depth import DepthCamera

# --- Configuration ---
resolution_width, resolution_height = (640, 480)
SAVE_DIR = "test6_custom_shape"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# ตัวแปรสำหรับเก็บจุดที่เมาส์คลิก
polygon_points = []
shape_saved = False
custom_mask = None

def draw_shape_callback(event, x, y, flags, param):
    """ฟังก์ชันรับค่าจากเมาส์เพื่อสร้างจุด Polygon"""
    global polygon_points
    if event == cv2.EVENT_LBUTTONDOWN:
        # คลิกซ้ายเพื่อเพิ่มจุด
        polygon_points.append((x, y))
        print(f"Point added: ({x}, {y})")

def main():
    global polygon_points, shape_saved, custom_mask
    Realsensed435Cam = DepthCamera(resolution_width, resolution_height)
    
    print("\n--- PHASE 1: DRAW YOUR TARGET SHAPE ---")
    print("1. LEFT CLICK on the video to create points around the pipe.")
    print("2. Press 'c' to CLEAR points if you make a mistake.")
    print("3. Press 's' to SAVE the shape and move to detection mode.")
    
    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", draw_shape_callback)

    while True:
        ret , depth_raw_frame, color_raw_frame = Realsensed435Cam.get_raw_frame()
        if not ret: continue
        
        color_frame = np.asanyarray(color_raw_frame.get_data())
        display_frame = color_frame.copy()

        # ---------------------------------------------------------
        # PHASE 1: โหมดวาดเส้น (Drawing Mode)
        # ---------------------------------------------------------
        if not shape_saved:
            cv2.putText(display_frame, "DRAW MODE: Click to add points. Press 's' to Save.", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            
            # วาดจุดและเส้นเชื่อม
            if len(polygon_points) > 0:
                # แปลง list เป็น numpy array สำหรับวาดเส้น
                pts = np.array(polygon_points, np.int32)
                pts = pts.reshape((-1, 1, 2))
                
                # วาดเส้นเชื่อมจุด (ยังไม่ปิดวง ถ้าจุดยังน้อยกว่า 3)
                is_closed = True if len(polygon_points) > 2 else False
                cv2.polylines(display_frame, [pts], isClosed=is_closed, color=(0, 255, 0), thickness=2)
                
                # วาดวงกลมทับจุดที่คลิก
                for pt in polygon_points:
                    cv2.circle(display_frame, pt, 4, (0, 0, 255), -1)

            cv2.imshow("Frame", display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'):
                polygon_points.append([]) # Reset
                polygon_points.clear()
                print("Points cleared!")
                
            elif key == ord('s') and len(polygon_points) > 2:
                # เมื่อกด 's' ให้สร้าง Mask สีดำ และเทสีขาวลงในรูปทรงที่เราวาด
                custom_mask = np.zeros(color_frame.shape[:2], dtype=np.uint8)
                pts = np.array(polygon_points, np.int32)
                cv2.fillPoly(custom_mask, [pts], 255)
                
                # บันทึกรูปทรงเก็บไว้
                mask_path = os.path.join(SAVE_DIR, "saved_shape_mask.png")
                cv2.imwrite(mask_path, custom_mask)
                print(f"\n[SUCCESS] Shape saved to {mask_path}")
                print("--- PHASE 2: DETECTION MODE ---")
                
                shape_saved = True
                
        # ---------------------------------------------------------
        # PHASE 2: โหมดตรวจจับ (Detection Mode) 
        # ใช้รูปทรงที่วาดมาตัดฉากหลังทิ้งก่อนทำ Edge Detection
        # ---------------------------------------------------------
        else:
            # ตัดเอาเฉพาะภาพที่อยู่ในขอบเขตที่เราวาดไว้
            masked_frame = cv2.bitwise_and(color_frame, color_frame, mask=custom_mask)
            
            # ทำ Edge Detection เฉพาะในขอบเขตนั้น
            gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            cv2.putText(color_frame, "DETECTION MODE: Press 'q' to Quit.", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # แสดงเส้นขอบที่วาดเอาไว้เป็นไกด์ไลน์จางๆ
            pts = np.array(polygon_points, np.int32)
            cv2.polylines(color_frame, [pts], isClosed=True, color=(0, 100, 0), thickness=1)

            cv2.imshow("Frame", color_frame)
            cv2.imshow("Masked View", masked_frame)
            cv2.imshow("Edges inside Shape", edges)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    Realsensed435Cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()