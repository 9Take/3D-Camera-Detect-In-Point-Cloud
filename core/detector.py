import cv2
import os
import numpy as np
import json
from collections import defaultdict

class ObjectDetector:
    def __init__(self, template_dir):
        self.sift = cv2.SIFT_create()
        # ใช้ Flann เหมือนใน temp2.py เป๊ะ
        FLANN_INDEX_KDTREE = 1
        self.flann = cv2.FlannBasedMatcher(dict(algorithm=FLANN_INDEX_KDTREE, trees=5), dict(checks=50))
        self.templates_by_target = self._load_templates(template_dir)
        self.target_list = sorted(self.templates_by_target.keys())

    def _load_templates(self, template_dir):
        templates = {}
        if not os.path.exists(template_dir):
            print(f"[WARNING] Template directory not found: {template_dir}")
            return templates

        for filename in os.listdir(template_dir):
            if filename.endswith('_template.png'):
                target_name = filename.replace('_template.png', '')
                
                # โหลดภาพ Template
                img_path = os.path.join(template_dir, filename)
                template_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                # พยายามโหลด offset จาก meta.json (หรือ offset.txt)
                offset = (template_img.shape[1]//2, template_img.shape[0]//2) # default center
                json_path = os.path.join(template_dir, f"{target_name}_meta.json")
                txt_path = os.path.join(template_dir, f"{target_name}_offset.txt")
                
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        meta = json.load(f)
                        offset = (meta.get("offset_x", offset[0]), meta.get("offset_y", offset[1]))
                elif os.path.exists(txt_path):
                    with open(txt_path, 'r') as f:
                        data = f.read().strip().split(',')
                        offset = (int(data[0]), int(data[1]))

                kp, des = self.sift.detectAndCompute(template_img, None)
                if des is not None and len(des) > 5:
                    templates[target_name] = {'img': template_img, 'offset': offset, 'kp': kp, 'des': des}
                    print(f"[CORE] Loaded Template '{target_name}' (Features: {len(kp)})")
        return templates
    def detect(self, color_frame, res_width, res_height):
        gray_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
        display_frame = color_frame.copy()
        detected_pixels = []
        detected_names = []
        detected_confidences = [] # เพิ่ม List เก็บค่าความมั่นใจ
        
        kp_frame, des_frame = self.sift.detectAndCompute(gray_frame, None)
        
        if des_frame is not None and len(des_frame) > 2:
            for target_name, t_data in self.templates_by_target.items():
                des_temp = t_data['des']
                kp_temp = t_data['kp']
                
                try:
                    matches = self.flann.knnMatch(des_temp, des_frame, k=2)
                    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
                    
                    if len(good_matches) > 12: # Min match count
                        src_pts = np.float32([kp_temp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        
                        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                        
                        if M is not None:
                            # --- คำนวณ Confidence Score ---
                            inliers = np.sum(mask) 
                            # สมมติฐาน: ถ้าจุดตรงกัน (Inliers) มากกว่า 30 จุด ถือว่ามั่นใจ 100%
                            confidence = min(100.0, (inliers / 30.0) * 100.0)
                            
                            th, tw = t_data['img'].shape
                            pts = np.float32([[0, 0], [0, th - 1], [tw - 1, th - 1], [tw - 1, 0]]).reshape(-1, 1, 2)
                            dst = cv2.perspectiveTransform(pts, M)
                            
                            target_pt_dst = cv2.perspectiveTransform(np.float32([[[t_data['offset'][0], t_data['offset'][1]]]]), M)
                            target_pixel = (int(target_pt_dst[0][0][0]), int(target_pt_dst[0][0][1]))
                            
                            if 0 <= target_pixel[0] < res_width and 0 <= target_pixel[1] < res_height:
                                detected_pixels.append(target_pixel)
                                detected_names.append(target_name)
                                detected_confidences.append(confidence) # เก็บค่า
                                
                                cv2.polylines(display_frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
                                cv2.circle(display_frame, target_pixel, 5, (0, 0, 255), -1)
                                # แสดงเปอเซ็นต์บนหน้าจอ
                                cv2.putText(display_frame, f"Locked: {target_name} ({confidence:.1f}%)", 
                                            (int(dst[0][0][0]), int(dst[0][0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                except Exception as e:
                    continue

        return detected_pixels, detected_names, detected_confidences, display_frame