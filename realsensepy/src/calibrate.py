import pyrealsense2 as rs
import numpy as np
import cv2

# ==========================================
# ⚙️ ตั้งค่าความแม่นยำ
# ==========================================
TRUE_DIAMETER_MM = 20.0  

pipeline = rs.pipeline()
config = rs.config()

# ⭐️ 1. เปิดทั้งกล้องสี (Color) และกล้องความลึก (Depth)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

print("กำลังเปิดกล้อง RealSense...")
profile = pipeline.start(config)

# ⭐️ 2. สร้างตัวช่วย "ซ้อนภาพ" (Align) ให้จุด Depth ตรงกับจุด Color
align_to = rs.stream.color
align = rs.align(align_to)

print(f"\n--- 3D Measurement Mode (Diameter & Depth) ---")

try:
    while True:
        # รอรับภาพทั้งคู่จากกล้อง
        frames = pipeline.wait_for_frames()
        
        # ⭐️ 3. สั่งให้กล้องซ้อนพิกัด Depth ให้ตรงกับ Color แบบเป๊ะๆ
        aligned_frames = align.process(frames)
        
        # ดึงภาพที่ซ้อนกันแล้วออกมา
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            continue

        img = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray)
        display_img = img.copy()

        _, thresh = cv2.threshold(gray_enhanced, 160, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        valid_circles = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 150 < area < 1500:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0: continue
                
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                
                if 0.8 < circularity <= 1.2:
                    (x, y), radius = cv2.minEnclosingCircle(cnt)
                    
                    valid_circles.append({
                        'contour': cnt,
                        'cx': int(x),
                        'cy': int(y),
                        'radius': int(radius),
                        'diameter_px': radius * 2
                    })

        circle_count = len(valid_circles)

        if circle_count > 0:
            # คำนวณ Scale เฉลี่ย
            all_ppms = [c['diameter_px'] / TRUE_DIAMETER_MM for c in valid_circles]
            avg_pixels_per_metric = np.median(all_ppms)
            
            for circle in valid_circles:
                cx = circle['cx']
                cy = circle['cy']
                r = circle['radius']
                
                diameter_mm = circle['diameter_px'] / avg_pixels_per_metric
                
                # ⭐️ 4. ดึงค่าความลึก (Z) ณ จุดกึ่งกลางของวงกลม
                # คำสั่ง get_distance จะคืนค่ามาเป็น "เมตร" เราเลยต้องคูณ 1000 ให้เป็น "มิลลิเมตร"
                depth_meters = depth_frame.get_distance(cx, cy)
                depth_mm = depth_meters * 1000.0
                
                # วาดกราฟิก
                cv2.circle(display_img, (cx, cy), r, (0, 255, 0), 2)
                cv2.circle(display_img, (cx, cy), 2, (0, 0, 255), -1)
                
                # ปรับข้อความให้แสดงทั้งขนาดความกว้าง (D) และความลึก (Z)
                # เช็คด้วยว่ากล้องอ่านค่าความลึกได้ไหม (ถ้าอ่านไม่ได้จะเป็น 0.0)
                if depth_mm > 0:
                    info_text = f"D:{diameter_mm:.1f} Z:{depth_mm:.1f}"
                else:
                    info_text = f"D:{diameter_mm:.1f} Z: N/A"
                    
                cv2.putText(display_img, info_text, (cx - 30, cy - r - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.putText(display_img, f"Circles: {circle_count}", (30, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
        if circle_count > 0:
            cv2.putText(display_img, f"Avg Scale: {avg_pixels_per_metric:.2f} px/mm", (30, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow('3D Measurement (X,Y,Z)', display_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()