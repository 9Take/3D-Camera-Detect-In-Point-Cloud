import cv2
import os
import numpy as np
import json

class ObjectDetector:
    def __init__(self, template_dir):
        self.sift = cv2.SIFT_create()
        FLANN_INDEX_KDTREE = 1
        self.flann = cv2.FlannBasedMatcher(dict(algorithm=FLANN_INDEX_KDTREE, trees=5), dict(checks=50))
        self.templates_by_target = self._load_templates(template_dir)
        self.target_list = sorted(self.templates_by_target.keys())

    def _load_templates(self, template_dir):
        templates = {}
        if not os.path.exists(template_dir):
            print(f"[WARNING] Template directory not found: {template_dir}")
            return templates

        for root, dirs, files in os.walk(template_dir):
            dirs[:] = [d for d in dirs if d.startswith('Point')]  # only Point*/ folders inside a program
            point_name = os.path.basename(root)
            if not point_name.startswith('Point'):
                continue
            for filename in files:
                if not filename.endswith('_template.png'):
                    continue
                target_name = filename.replace('_template.png', '')

                img_path = os.path.join(root, filename)
                template_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                offset = (template_img.shape[1]//2, template_img.shape[0]//2)
                json_path = os.path.join(root, f"{target_name}_meta.json")
                txt_path = os.path.join(root, f"{target_name}_offset.txt")

                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        meta = json.load(f)
                        offset = (meta.get("offset_x", offset[0]), meta.get("offset_y", offset[1]))
                elif os.path.exists(txt_path):
                    with open(txt_path, 'r') as f:
                        data = f.read().strip().split(',')
                        offset = (int(data[0]), int(data[1]))
                else:
                    print(f"[WARNING] No meta.json for '{target_name}' — using image center as offset")

                kp, des = self.sift.detectAndCompute(template_img, None)
                if des is not None and len(des) > 5:
                    templates[target_name] = {'img': template_img, 'offset': offset, 'kp': kp, 'des': des, 'point': point_name}
                    print(f"[CORE] Loaded Template '{target_name}' under '{point_name}' (Features: {len(kp)})")
        return templates

    def detect(self, color_frame, res_width, res_height):
        gray_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
        display_frame = color_frame.copy()
        detected_pixels = []
        detected_names = []
        detected_confidences = [] 
        detected_homographies = [] # เพิ่ม List เก็บพิกัดมุมกล่อง Homography
        
        kp_frame, des_frame = self.sift.detectAndCompute(gray_frame, None)
        
        if des_frame is not None and len(des_frame) > 2:
            for target_name, t_data in self.templates_by_target.items():
                des_temp = t_data['des']
                kp_temp = t_data['kp']
                
                try:
                    matches = self.flann.knnMatch(des_temp, des_frame, k=2)
                    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
                    
                    if len(good_matches) > 12: 
                        src_pts = np.float32([kp_temp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        
                        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                        
                        if M is not None:
                            inliers = np.sum(mask) 
                            confidence = min(100.0, (inliers / 30.0) * 100.0)
                            
                            th, tw = t_data['img'].shape
                            pts = np.float32([[0, 0], [0, th - 1], [tw - 1, th - 1], [tw - 1, 0]]).reshape(-1, 1, 2)
                            dst = cv2.perspectiveTransform(pts, M)
                            
                            target_pt_dst = cv2.perspectiveTransform(np.float32([[[t_data['offset'][0], t_data['offset'][1]]]]), M)
                            target_pixel = (int(target_pt_dst[0][0][0]), int(target_pt_dst[0][0][1]))
                            
                            if 0 <= target_pixel[0] < res_width and 0 <= target_pixel[1] < res_height:
                                detected_pixels.append(target_pixel)
                                detected_names.append(target_name)
                                detected_confidences.append(confidence) 
                                detected_homographies.append(dst) # เก็บค่ามุมกล่อง
                                
                                cv2.polylines(display_frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
                                cv2.circle(display_frame, target_pixel, 5, (0, 0, 255), -1)
                                cv2.putText(display_frame, f"Locked: {target_name} ({confidence:.1f}%)", 
                                            (int(dst[0][0][0]), int(dst[0][0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                except Exception as e:
                    continue

        return detected_pixels, detected_names, detected_confidences, detected_homographies, display_frame

    def build_sub_window_grid(self, color_frame, detected_pixels, detected_names, confidences, detected_homographies, sub_w=280, sub_h=220):
        """
        สร้างภาพ Grid โดยการ Crop จาก color_frame ดิบ แล้ววาดเฉพาะเส้น Contour (Homography) 
        และป้ายกำกับของตนเองแยกอิสระเพื่อไม่ให้ภาพซ้อนทับกัน
        """
        if not detected_pixels:
            empty_grid = np.zeros((sub_h, sub_w, 3), dtype=np.uint8)
            cv2.putText(empty_grid, "No Objects Detected", (15, sub_h // 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return empty_grid

        cropped_images = []
        h_frame, w_frame = color_frame.shape[:2]
        
        crop_size_w = 120
        crop_size_h = 100

        for idx, (pixel, name, conf, dst_poly) in enumerate(zip(detected_pixels, detected_names, confidences, detected_homographies)):
            px, py = pixel
            
            x_start = max(0, px - crop_size_w)
            y_start = max(0, py - crop_size_h)
            x_end = min(w_frame, px + crop_size_w)
            y_end = min(h_frame, py + crop_size_h)
            
            crop = color_frame[y_start:y_end, x_start:x_end].copy()
            
            if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                continue
                
            local_poly = dst_poly.copy()
            local_poly[:, 0, 0] = local_poly[:, 0, 0] - x_start  
            local_poly[:, 0, 1] = local_poly[:, 0, 1] - y_start  
            
            scale_x = sub_w / crop.shape[1]
            scale_y = sub_h / crop.shape[0]
            
            crop_resized = cv2.resize(crop, (sub_w, sub_h))
            
            local_poly_resized = local_poly.copy()
            local_poly_resized[:, 0, 0] = local_poly_resized[:, 0, 0] * scale_x
            local_poly_resized[:, 0, 1] = local_poly_resized[:, 0, 1] * scale_y
            
            cv2.polylines(crop_resized, [np.int32(local_poly_resized)], True, (0, 255, 0), 2, cv2.LINE_AA)
            
            local_center = (int((px - x_start) * scale_x), int((py - y_start) * scale_y))
            cv2.circle(crop_resized, local_center, 6, (0, 0, 255), -1)
            
            color = (0, 255, 0) if name.startswith('A') else (255, 255, 0) if name.startswith('B') else (0, 165, 255)
            cv2.putText(crop_resized, f"ID: {name}", (10, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(crop_resized, f"Conf: {conf:.1f}%", (10, sub_h - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.rectangle(crop_resized, (0, 0), (sub_w - 1, sub_h - 1), (100, 100, 100), 2)
            
            cropped_images.append(crop_resized)

        if not cropped_images:
            return np.zeros((sub_h, sub_w, 3), dtype=np.uint8)

        num_items = len(cropped_images)
        cols = min(4, num_items)
        rows = (num_items + cols - 1) // cols
        grid_frame = np.zeros((rows * sub_h, cols * sub_w, 3), dtype=np.uint8)
        
        for idx, crop_img in enumerate(cropped_images):
            r = idx // cols
            c = idx % cols
            grid_frame[r * sub_h : (r + 1) * sub_h, c * sub_w : (c + 1) * sub_w] = crop_img
            
        return grid_frame