import cv2
import time
import yaml
import json
import os
import argparse
import numpy as np

# อิมพอร์ตโมดูลต่างๆ ตามโครงสร้าง Project Root
from communication.realsense import DepthCamera
from communication.plc_comm import PLCCommunicator
from core.detector import ObjectDetector
from core.transformer import PointCloudTransformer

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Heat Exchanger Vision System")
    parser.add_argument('--debug', action='store_true', help='เปิดหน้าต่างเพื่อดูการตรวจจับ 2D และ 3D Point Cloud')
    parser.add_argument('--show2d', action='store_true', help='แสดงผลการตรวจจับ 2D')
    
    args = parser.parse_args()

    config = load_config()
    os.makedirs(config['paths']['save_dir'], exist_ok=True)

    print("\n[INIT] Initializing Systems...")
    cam = DepthCamera(config['camera']['resolution_width'], config['camera']['resolution_height'])
    detector = ObjectDetector(config['paths']['template_dir'])
    transformer = PointCloudTransformer(cam, config['camera']['resolution_width'], config['camera']['resolution_height'], config['paths']['save_dir'])
    
    # เชื่อมต่อ PLC
    plc = PLCCommunicator(config['plc']['ip'], config['plc']['port'])
    plc.connect()

    print("\n[SYSTEM READY] Starting Vision Loop...")
    
    try:
        while True:
            ret, depth_raw, color_raw = cam.get_raw_frame()
            if not ret: continue

            color_frame = np.asanyarray(color_raw.get_data())
            
            # 1. ค้นหาวัตถุทั้งหมด (2D Detection) จากตัวรับ Homographies เพิ่มเติม
            detected_pixels, detected_names, confidences, detected_homographies, display_frame = detector.detect(
                color_frame, config['camera']['resolution_width'], config['camera']['resolution_height']
            )

            # อัปเดตสถานะแบบ Real-time ลง JSON
            realtime_status = {
                "timestamp": time.time(),
                "targets_in_view": detected_names,
                "confidences": confidences
            }
            with open(os.path.join(config['paths']['save_dir'], "current_detect.json"), "w") as f:
                json.dump(realtime_status, f)

            # --- FILTERING LOGIC: หาตัวย่อยที่มี Confidence สูงสุดของตระกูล A และ B ---
            best_a_idx = -1
            best_a_conf = -1.0
            best_a_subname = ""
            
            best_b_idx = -1
            best_b_conf = -1.0
            best_b_subname = ""
            
            for idx, (name, conf) in enumerate(zip(detected_names, confidences)):
                if name.startswith('A'):
                    if conf > best_a_conf:
                        best_a_conf = conf
                        best_a_idx = idx
                        best_a_subname = name
                elif name.startswith('B'):
                    if conf > best_b_conf:
                        best_b_conf = conf
                        best_b_idx = idx
                        best_b_subname = name

            # --- สร้างหน้าจอหลัก (Main Window) ที่แสดงเฉพาะตัวที่มั่นใจที่สุด ---
            main_display = color_frame.copy()
            if best_a_idx != -1:
                pixel_a = detected_pixels[best_a_idx]
                cv2.circle(main_display, pixel_a, 8, (0, 255, 0), -1)
                cv2.putText(main_display, f"BEST A: {best_a_subname} ({best_a_conf:.1f}%)", 
                            (pixel_a[0] - 40, pixel_a[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
            if best_b_idx != -1:
                pixel_b = detected_pixels[best_b_idx]
                cv2.circle(main_display, pixel_b, 8, (255, 255, 0), -1)
                cv2.putText(main_display, f"BEST B: {best_b_subname} ({best_b_conf:.1f}%)", 
                            (pixel_b[0] - 40, pixel_b[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # --- จัดการเรื่องการเปิดหน้าต่างแสดงผลตาม Argument ---
            if args.debug or args.show2d:
                # 1. แสดงหน้าต่างหลัก (แสดงเฉพาะตัวพิกัดเดี่ยว Best A, Best B คลีนๆ)
                window_title = "Vision System - Main Camera" if args.debug else "2D Detection"
                cv2.imshow(window_title, main_display)
                
                # 2. เรียกใช้เมธอดสร้าง Grid ผ่าน Class ObjectDetector ตรงๆ
                grid_img = detector.build_sub_window_grid(
                    color_frame, 
                    detected_pixels, 
                    detected_names, 
                    confidences, 
                    detected_homographies, 
                    sub_w=280, 
                    sub_h=220
                )
                cv2.imshow("Detected Sub-Windows Grid", grid_img)
                cv2.setWindowProperty("Detected Sub-Windows Grid", cv2.WND_PROP_TOPMOST, 1)
                
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'): break
            else:
                # ถ้ารันโหมดปกติ ไม่แสดงหน้าต่างเพื่อความเร็ว
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'): break

            # 2. ตรวจสอบสัญญาณ Trigger จาก PLC
            trigger_status = plc.read_bit(config['plc']['trigger_device'])
            
            if trigger_status[0] == 1 and len(detected_pixels) > 0:
                print("\n[TRIGGER] Received signal from PLC. Finding best A and B sub-templates...")
                
                filtered_pixels = []
                filtered_names = []
                
                if best_a_idx != -1:
                    filtered_pixels.append(detected_pixels[best_a_idx])
                    filtered_names.append(best_a_subname)
                    print(f"[FILTER] Selected Best A: {best_a_subname} (Conf: {best_a_conf:.2f}%)")
                    
                if best_b_idx != -1:
                    filtered_pixels.append(detected_pixels[best_b_idx])
                    filtered_names.append(best_b_subname)
                    print(f"[FILTER] Selected Best B: {best_b_subname} (Conf: {best_b_conf:.2f}%)")
                
                if len(filtered_pixels) == 0:
                    print("[WARNING] Trigger received but no valid 'A' or 'B' targets found.")
                    plc.write_bit(config['plc']['trigger_device'], 0)
                    continue

                # 3. คำนวณพิกัด 3D โดยใช้ตัวย่อยที่คัดสรรแล้ว
                extracted_6dof = transformer.extract_3d_data(
                    filtered_pixels, 
                    filtered_names, 
                    show_3d=args.debug
                )
                
                if extracted_6dof:
                    print(f"\n[INFO] Found {len(extracted_6dof)} targets.")
                    
                    # บันทึกค่าลง JSON ก่อนส่ง PLC
                    memory_state = {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "targets": {}
                    }
                    for t_name, t_data in extracted_6dof.items():
                        memory_state["targets"][t_name] = {
                            "Position_X": round(t_data[0], 4),
                            "Position_Y": round(t_data[1], 4),
                            "Position_Z": round(t_data[2], 4),
                            "Roll": round(t_data[3], 2),
                            "Pitch": round(t_data[4], 2),
                            "Yaw": round(t_data[5], 2)
                        }
                    
                    with open(config['paths']['position_mem'], "w") as f:
                        json.dump(memory_state, f, indent=4)
                    print(f"[LOG] Memory saved to {config['paths']['position_mem']}")
                    
                    num_targets = min(len(extracted_6dof), config['plc']['max_points'])
                    plc.write_scaled_word(config['plc']['point_count_device'], num_targets, multiplier=1)

                    target_configs = config['plc'].get('targets', {})
                    
                    # 4. วนลูปส่งข้อมูลไป PLC แบบเจาะจง Register โดยแมปชื่อย่อยกลับเป็นพอร์ตส่งหลักของ A และ B
                    for i, (target_name, target_data) in enumerate(list(extracted_6dof.items())[:num_targets]):
                        
                        plc_target_key = "A" if target_name.startswith("A") else "B" if target_name.startswith("B") else target_name

                        if plc_target_key not in target_configs:
                            print(f"[WARNING] PLC Config key '{plc_target_key}' not found in config.yaml")
                            continue
                            
                        t_conf = target_configs[plc_target_key]
                        x, y, z = target_data[0:3]
                        r_rad, p_rad, yw_rad = target_data[3:6]
                        r_deg, p_deg, yw_deg = np.rad2deg([r_rad, p_rad, yw_rad])

                        print(f"[{i+1}/{num_targets}] Target '{target_name}' mapped to PLC Register '{plc_target_key}'")
                        print(f" ➔ Pos(m): X:{x:.4f}, Y:{y:.4f}, Z:{z:.4f}")
                        print(f" ➔ Ori(rad): R:{r_rad:.4f}, P:{p_rad:.4f}, Y:{yw_rad:.4f}")
                        print(f" ➔ Ori(deg): R:{r_deg:.2f}°, P:{p_deg:.2f}°, Y:{yw_deg:.2f}°")
                        
                        plc.write_scaled_word(t_conf['Input_X'], x, multiplier=10000) 
                        plc.write_scaled_word(t_conf['Input_Y'], y, multiplier=10000)
                        plc.write_scaled_word(t_conf['Input_Z'], z, multiplier=10000)
                        plc.write_scaled_word(t_conf['Input_r'], r_rad, multiplier=10000)
                        plc.write_scaled_word(t_conf['Input_p'], p_rad, multiplier=10000)
                        plc.write_scaled_word(t_conf['Input_y'], yw_rad, multiplier=10000)
                        
                    print(f"[PLC] Sent 6-DOF Data for {num_targets} points successfully.")

                    # 5. Handshake
                    plc.write_bit(config['plc']['status_device'], 1)
                    time.sleep(0.5)
                    plc.write_bit(config['plc']['status_device'], 0)
                    
                    print("[PLC] Handshake complete. Waiting for next trigger...\n")
                    plc.write_bit(config['plc']['trigger_device'], 0) 

    except KeyboardInterrupt:
        print("\n[INFO] Exiting program...")
    finally:
        plc.disconnect()
        cam.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()