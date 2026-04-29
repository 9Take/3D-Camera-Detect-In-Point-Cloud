import pyrealsense2 as rs
import numpy as np
import cv2

# ==========================================
# ⚙️ ตั้งค่าความแม่นยำ
# ==========================================
TRUE_DIAMETER_MM = 20.0  # ขนาดเส้นผ่านศูนย์กลางจริงของวงกลม (มิลลิเมตร)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

print("กำลังเปิดกล้อง RealSense...")
pipeline.start(config)
print(f"\n--- True Scale Measurement Mode (Reference: {TRUE_DIAMETER_MM}mm) ---")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        img = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # ปรับแสงเงาให้คงที่ด้วย CLAHE เพื่อความแม่นยำของขอบวงกลม
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray)
        
        display_img = img.copy()

        # 1. Threshold หาพื้นที่สีขาว
        _, thresh = cv2.threshold(gray_enhanced, 160, 255, cv2.THRESH_BINARY)

        # 2. ค้นหารูปทรง
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        valid_circles = []
        pixels_per_metric = None # ตัวแปรเก็บสัดส่วน พิกเซล ต่อ มิลลิเมตร

        # 3. รวบรวมข้อมูลวงกลมทั้งหมดก่อน
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # ⭐️ 1. แก้ไขตรงนี้: เพิ่มค่าขั้นต่ำจาก 10 เป็น 150 (หรือ 200) เพื่อกรองรูน็อตเล็กๆ ทิ้งไป!
            if 150 < area < 1500:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0: continue
                
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                
                if 0.8 < circularity <= 1.2:
                    (x, y), radius = cv2.minEnclosingCircle(cnt)
                    diameter_px = radius * 2
                    
                    valid_circles.append({
                        'contour': cnt,
                        'cx': int(x),
                        'cy': int(y),
                        'radius': int(radius),
                        'diameter_px': diameter_px,
                        'circularity': circularity
                    })

        circle_count = len(valid_circles)

        if circle_count > 0:
            # ⭐️ 1. ดึงขนาดพิกเซลของ "ทุกวงกลม" มาคำนวณหาสัดส่วน PPM ของแต่ละวง
            all_ppms = []
            for c in valid_circles:
                ppm = c['diameter_px'] / TRUE_DIAMETER_MM
                all_ppms.append(ppm)
            
            # ⭐️ 2. หาค่าเฉลี่ย (Average Scale) จากทุกวงรวมกัน
            # แนะนำให้ใช้ np.median (มัธยฐาน) จะเสถียรกว่า np.mean (ค่าเฉลี่ย) ในกรณีที่มีจุดแหว่งเพี้ยนหนักๆ หลุดมา
            avg_pixels_per_metric = np.median(all_ppms)
            
            # 3. นำค่าเฉลี่ยที่ได้ ไปคำนวณและวาดผลลัพธ์ให้กับทุกๆ จุด
            for circle in valid_circles:
                cx = circle['cx']
                cy = circle['cy']
                r = circle['radius']
                
                # ใช้สเกลเฉลี่ยที่นิ่งและแม่นยำแล้ว มาคำนวณกลับเป็นมิลลิเมตร
                diameter_mm = circle['diameter_px'] / avg_pixels_per_metric
                
                cv2.circle(display_img, (cx, cy), r, (0, 255, 0), 2)
                cv2.circle(display_img, (cx, cy), 2, (0, 0, 255), -1)
                
                info_text = f"{diameter_mm:.1f}mm"
                
                # วาดข้อความสีขาวปกติ (เพราะตอนนี้ทุกวงคือ REF ร่วมกันหมดแล้ว)
                cv2.putText(display_img, info_text, (cx - 20, cy - r - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # แสดงยอดรวมและสัดส่วน PPM เฉลี่ย
        cv2.putText(display_img, f"Detected: {circle_count}", (30, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # โชว์ค่า Scale ที่เป็นค่าเฉลี่ยของทั้งกระดาน
        if circle_count > 0:
            cv2.putText(display_img, f"Avg Scale: {avg_pixels_per_metric:.2f} px/mm", (30, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow('True Scale Measurement', display_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()