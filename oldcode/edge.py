import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os

# ตัวแปรสำหรับเก็บค่า mouse position และ depth
mouse_pos = (0, 0)
edge_mouse_pos = (0, 0)
current_depth_image = None
current_depth_frame = None

def mouse_callback_depth(event, x, y, flags, param):
    """Callback function สำหรับติดตามตำแหน่ง mouse ที่ depth map"""
    global mouse_pos
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_pos = (x, y)

def mouse_callback_edge(event, x, y, flags, param):
    """Callback function สำหรับติดตามตำแหน่ง mouse ที่ edge window"""
    global edge_mouse_pos
    if event == cv2.EVENT_MOUSEMOVE:
        edge_mouse_pos = (x, y)

def create_depth_mask(depth_image, max_distance=500):
    """สร้าง mask สำหรับการกรองค่า depth ที่เกิน max_distance"""
    mask = depth_image <= max_distance
    return mask

def apply_depth_filter_to_image(image, mask):
    """ใช้ mask เพื่อทำให้พิกเซลที่เกิน max_distance เป็นสีดำ"""
    filtered_image = image.copy()
    filtered_image[~mask] = 0
    return filtered_image

def create_depth_scale_bar(height=480, width=60):
    """สร้างแถบสเกลความลึกด้วยสีแบบ Colormap"""
    scale_bar = np.zeros((height, width, 3), dtype=np.uint8)
    
    # สร้าง gradient โดยใช้ค่าความเข้ม
    for i in range(height):
        intensity = int((1 - i / height) * 255)  # จากล่างขึ้นบน (0 ไป 255)
        scale_bar[i, :] = intensity
    
    # ใช้ colormap JET เพื่อได้สีที่สวยงาม
    scale_bar = cv2.applyColorMap(scale_bar, cv2.COLORMAP_JET)
    
    return scale_bar

def add_depth_scale_bar_to_image(image, scale_bar, max_depth=5000):
    """เพิ่มแถบสเกลความลึกด้านข้างของภาพ"""
    h, w = image.shape[:2]
    scale_h, scale_w = scale_bar.shape[:2]
    
    # ปรับขนาดแถบสเกล
    scale_bar_resized = cv2.resize(scale_bar, (scale_w, h))
    
    # รวมภาพและแถบสเกล
    combined = np.hstack([image, scale_bar_resized])
    
    # เพิ่มข้อความอธิบายความลึก
    cv2.putText(combined, f"{max_depth}mm", (w + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(combined, "0mm", (w + 5, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return combined

def display_mouse_depth(image, depth_image, mouse_pos, depth_frame):
    """แสดงค่า depth ที่ตำแหน่ง mouse"""
    x, y = mouse_pos
    h, w = image.shape[:2]
    
    # ตรวจสอบว่า mouse อยู่ภายในขอบเขตของ depth map
    if 0 <= x < w and 0 <= y < h and depth_frame is not None:
        depth_value = depth_image[y, x]
        
        # วาดวงกลมที่ตำแหน่ง mouse
        cv2.circle(image, (x, y), 5, (0, 255, 255), 2)
        
        # แสดงค่า depth เป็นข้อความ
        text = f"Depth: {depth_value}mm"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = max(5, min(x - text_size[0]//2, w - text_size[0] - 5))
        text_y = max(20, y - 15)
        
        # วาด background สำหรับข้อความ
        cv2.rectangle(image, (text_x - 2, text_y - text_size[1] - 2), 
                      (text_x + text_size[0] + 2, text_y + 2), (0, 0, 0), -1)
        cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return image

def display_mouse_pointer_on_edge(image, depth_image, mouse_pos):
    """แสดงตัวชี้ mouse และค่า depth บนภาพ edge detection"""
    x, y = mouse_pos
    h, w = image.shape[:2]
    
    # ตรวจสอบว่า mouse อยู่ภายในขอบเขต
    if 0 <= x < w and 0 <= y < h:
        # แปลงเป็น BGR ถ้าเป็นภาพขาวดำ
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # วาดวงกลมที่ตำแหน่ง mouse
        cv2.circle(image, (x, y), 6, (0, 255, 255), 2)
        cv2.circle(image, (x, y), 2, (0, 255, 255), -1)
        
        # ดึงค่า depth จากตำแหน่ง mouse
        depth_value = depth_image[y, x]
        
        # แสดง depth value
        text = f"Depth: {depth_value}mm"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        text_x = max(5, min(x - text_size[0]//2, w - text_size[0] - 5))
        text_y = max(20, y - 15)
        
        # วาด background สำหรับข้อความ
        cv2.rectangle(image, (text_x - 2, text_y - text_size[1] - 2), 
                      (text_x + text_size[0] + 2, text_y + 2), (0, 0, 0), -1)
        cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    
    return image

# ตั้งค่าสตรีมของกล้อง RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# เริ่มการสตรีม
pipeline.start(config)

# ตัวแปรสำหรับจับภาพทุกๆ 5 วินาที
last_capture_time = time.time()
capture_interval = 5  # วินาที
output_dir = "captures"

# สร้างโฟลเดอร์สำหรับเก็บภาพถ้ายังไม่มี
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# สร้างแถบสเกล
scale_bar = create_depth_scale_bar()

# ตั้ง mouse callback สำหรับหน้าต่าง depth map และ edge
cv2.namedWindow('1. Depth Map (Heatmap)')
cv2.setMouseCallback('1. Depth Map (Heatmap)', mouse_callback_depth)
cv2.namedWindow('3. Edge Detection (Canny)')
cv2.setMouseCallback('3. Edge Detection (Canny)', mouse_callback_edge)

try:
    while True:
        # รอรับเฟรมภาพจากกล้อง
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue
        
        # ตรวจสอบว่าถึงเวลาจับภาพหรือไม่ (ทุกๆ 5 วินาที)
        current_time = time.time()
        if current_time - last_capture_time < capture_interval:
            continue
        
        last_capture_time = current_time
        timestamp = int(current_time)

        # แปลงข้อมูลให้อยู่ในรูป Numpy Array
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # เก็บค่า depth สำหรับใช้ใน mouse callback
        current_depth_image = depth_image.copy()
        current_depth_frame = depth_frame
        
        # ---------------------------------------------------------
        # 0. สร้าง mask เพื่อกรองค่า depth ที่เกิน 500mm
        # ---------------------------------------------------------
        depth_mask = create_depth_mask(depth_image, max_distance=500)
        
        # สร้างภาพ depth ที่ถูกกรอง
        filtered_depth_image = depth_image.copy()
        filtered_depth_image[~depth_mask] = 0

        # ---------------------------------------------------------
        # 1. จัดการภาพความลึก (Depth Map)
        # ---------------------------------------------------------
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        depth_colormap = apply_depth_filter_to_image(depth_colormap, depth_mask)
        
        # เพิ่มแถบสเกลและข้อมูล mouse
        depth_with_scale = add_depth_scale_bar_to_image(depth_colormap.copy(), scale_bar)
        depth_with_scale[:depth_colormap.shape[0], :depth_colormap.shape[1]] = display_mouse_depth(
            depth_with_scale[:depth_colormap.shape[0], :depth_colormap.shape[1]].copy(), 
            depth_image, mouse_pos, depth_frame
        )

        # ---------------------------------------------------------
        # 2. ทำ Edge Detection จากภาพ Depth ที่ถูกกรอง (500mm)
        # ---------------------------------------------------------
        # แปลง filtered depth เป็น 8-bit image สำหรับ edge detection
        depth_normalized = cv2.normalize(filtered_depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_8bit = np.uint8(depth_normalized)
        
        blurred_depth = cv2.GaussianBlur(depth_8bit, (5, 5), 0)     # ลด Noise
        edges = cv2.Canny(blurred_depth, 50, 150)                   # หาเส้นขอบ
        
        # เพิ่มตัวชี้ mouse และค่า depth บนภาพ edge
        edges_with_pointer = display_mouse_pointer_on_edge(edges.copy(), depth_image, edge_mouse_pos)

        # ---------------------------------------------------------
        # 4. บันทึกภาพทุกๆ 5 วินาที
        # ---------------------------------------------------------
        cv2.imwrite(f"{output_dir}/1_depth_map_{timestamp}.png", depth_with_scale)
        cv2.imwrite(f"{output_dir}/2_color_image_{timestamp}.png", color_image)
        cv2.imwrite(f"{output_dir}/3_edge_detection_{timestamp}.png", edges_with_pointer)
        print(f"[{timestamp}] Captured and saved frames")

        # ---------------------------------------------------------
        # 5. แสดงผลหน้าต่าง
        # ---------------------------------------------------------
        cv2.imshow('1. Depth Map (Heatmap)', depth_with_scale)
        cv2.imshow('2. Normal Color (RGB)', color_image)
        cv2.imshow('3. Edge Detection (Canny)', edges_with_pointer)

        # กด 'q' เพื่อออก
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()