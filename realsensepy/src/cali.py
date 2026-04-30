import pyrealsense2 as rs
import numpy as np
import cv2
import math
from collections import deque

# ==========================================
# ⚙️ ตั้งค่าความแม่นยำสูง
# ==========================================
TRUE_DIAMETER_MM = 20.0  
BUFFER_SIZE = 10 # จำนวนเฟรมที่ใช้เฉลี่ยค่าเพื่อลด Jitter (Noise)

# ตัวแปรเก็บตำแหน่งและการคลิก
history_p1 = deque(maxlen=BUFFER_SIZE)
history_p2 = deque(maxlen=BUFFER_SIZE)
selected_circles = [] 
click_events = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        click_events.append((x, y))

pipeline = rs.pipeline()
config = rs.config()

# 1. เปิดสตรีมกล้อง (แนะนำ 720p เพื่อความละเอียดในการหาจุดกึ่งกลาง)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

print("กำลังเปิดกล้อง RealSense พร้อมเปิดระบบ Ultra-Accuracy...")
profile = pipeline.start(config)

# ⭐️ 2. ตั้งค่า Hardware Filters เพื่อลด Error ของแกน Z
spatial = rs.spatial_filter()    # ลด Noise ในแนวกว้าง
temporal = rs.temporal_filter()  # ลดการสั่นไหวระหว่างเฟรม
hole_filling = rs.hole_filling_filter() # ถมจุดบอด

align = rs.align(rs.stream.color)

cv2.namedWindow('High Accuracy 3D Measurement')
cv2.setMouseCallback('High Accuracy 3D Measurement', mouse_callback)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            continue

        # ⭐️ 3. ประมวลผล Depth ผ่าน Filter และแปลงกลับเป็น Depth Object
        filtered_depth = spatial.process(depth_frame)
        filtered_depth = temporal.process(filtered_depth)
        filtered_depth = hole_filling.process(filtered_depth)
        filtered_depth = filtered_depth.as_depth_frame() # สำคัญ: เพื่อให้ใช้ get_distance ได้

        depth_intrin = filtered_depth.profile.as_video_stream_profile().intrinsics

        img = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # เพิ่มความคมชัดของขอบเพื่อความแม่นยำ Sub-pixel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray)
        display_img = img.copy()

        _, thresh = cv2.threshold(gray_enhanced, 170, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_circles = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # กรองขนาดช่องให้พอดีกับแผ่น Mech-Mind
            if 150 < area < 2000:
                # ⭐️ 4. หาจุดกึ่งกลางระดับ Sub-pixel ด้วย Moments
                M = cv2.moments(cnt)
                if M["m00"] == 0: continue
                cx_sub = M["m10"] / M["m00"]
                cy_sub = M["m01"] / M["m00"]
                
                # ตรวจสอบความกลม
                perimeter = cv2.arcLength(cnt, True)
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                
                if 0.8 < circularity <= 1.2:
                    depth = filtered_depth.get_distance(int(cx_sub), int(cy_sub))
                    point_3d = None
                    if depth > 0:
                        point_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, [cx_sub, cy_sub], depth)

                    valid_circles.append({
                        'cx': int(cx_sub),
                        'cy': int(cy_sub),
                        'radius': int(math.sqrt(area/np.pi)),
                        'diameter_px': math.sqrt(area/np.pi) * 2,
                        'point_3d': point_3d 
                    })

        # ⭐️ 5. ระบบคลิกเมาส์
        for click_x, click_y in click_events:
            for circle in valid_circles:
                dist_to_click = math.hypot(circle['cx'] - click_x, circle['cy'] - click_y)
                if dist_to_click <= circle['radius'] * 1.5:
                    selected_circles.append(circle)
                    if len(selected_circles) > 2:
                        selected_circles.pop(0)
                    history_p1.clear() # ล้างค่าเฉลี่ยใหม่เมื่อเลือกจุดใหม่
                    history_p2.clear()
                    break
        click_events.clear()

        # ⭐️ 6. วาดผลลัพธ์และคำนวณเฉลี่ย
        if len(valid_circles) > 0:
            avg_ppm = np.median([c['diameter_px'] / TRUE_DIAMETER_MM for c in valid_circles])
            
            for circle in valid_circles:
                cx, cy, r = circle['cx'], circle['cy'], circle['radius']
                diameter_mm = circle['diameter_px'] / avg_ppm
                
                is_selected = any(c['cx'] == cx and c['cy'] == cy for c in selected_circles)
                color = (0, 255, 255) if is_selected else (0, 255, 0)
                
                cv2.circle(display_img, (cx, cy), r, color, 2)
                cv2.circle(display_img, (cx, cy), 2, (0, 0, 255), -1)

        # ⭐️ 7. คำนวณระยะห่าง 3D แบบ Moving Average (หัวใจของความนิ่ง)
        if len(selected_circles) == 2:
            p1_curr = None
            p2_curr = None
            
            # ติดตามจุดที่เลือกในเฟรมปัจจุบัน
            for c in valid_circles:
                if math.hypot(c['cx']-selected_circles[0]['cx'], c['cy']-selected_circles[0]['cy']) < 20:
                    p1_curr = c['point_3d']
                if math.hypot(c['cx']-selected_circles[1]['cx'], c['cy']-selected_circles[1]['cy']) < 20:
                    p2_curr = c['point_3d']
            
            if p1_curr is not None and p2_curr is not None:
                history_p1.append(p1_curr)
                history_p2.append(p2_curr)
                
                # หาค่าเฉลี่ย 10 เฟรมล่าสุด
                avg_p1 = np.mean(history_p1, axis=0)
                avg_p2 = np.mean(history_p2, axis=0)
                
                dist_3d_mm = np.linalg.norm(avg_p1 - avg_p2) * 1000.0
                
                cv2.line(display_img, (selected_circles[0]['cx'], selected_circles[0]['cy']), 
                         (selected_circles[1]['cx'], selected_circles[1]['cy']), (0, 165, 255), 3)
                
                cv2.putText(display_img, f"True Dist: {dist_3d_mm:.2f} mm", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        cv2.putText(display_img, "Press 'C' to clear points. 'Q' to quit.", (30, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('High Accuracy 3D Measurement', display_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            selected_circles.clear()
        elif key == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()