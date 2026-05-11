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
            
            # 1. ค้นหาวัตถุทั้งหมด (2D Detection)
            detected_pixels, detected_names, confidences, display_frame = detector.detect(
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

            if args.debug:
                cv2.imshow("Vision System - Main Camera", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'): break
            if args.show2d:
                cv2.imshow("2D Detection", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'): break


            # 2. ตรวจสอบสัญญาณ Trigger จาก PLC
            trigger_status = plc.read_bit(config['plc']['trigger_device'])
            
            if trigger_status[0] == 1 and len(detected_pixels) > 0:
                print("\n[TRIGGER] Received signal from PLC. Finding best A and B templates...")
                
                # --- FILTERING LOGIC: หาตัวที่มี Confidence สูงสุดของ A และ B ---
                best_a_idx = -1
                best_a_conf = -1.0
                best_b_idx = -1
                best_b_conf = -1.0
                
                for idx, (name, conf) in enumerate(zip(detected_names, confidences)):
                    if name.startswith('A'):
                        if conf > best_a_conf:
                            best_a_conf = conf
                            best_a_idx = idx
                    elif name.startswith('B'):
                        if conf > best_b_conf:
                            best_b_conf = conf
                            best_b_idx = idx
                
                filtered_pixels = []
                filtered_names = []
                
                if best_a_idx != -1:
                    filtered_pixels.append(detected_pixels[best_a_idx])
                    filtered_names.append("A")
                    
                if best_b_idx != -1:
                    filtered_pixels.append(detected_pixels[best_b_idx])
                    filtered_names.append("B")
                
                if len(filtered_pixels) == 0:
                    print("[WARNING] Trigger received but no valid 'A' or 'B' targets found.")
                    plc.write_bit(config['plc']['trigger_device'], 0)
                    continue

                # 3. คำนวณพิกัด 3D
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
                    
                    # 4. วนลูปส่งข้อมูลไป PLC แบบเจาะจง Register
                    for i, (target_name, target_data) in enumerate(list(extracted_6dof.items())[:num_targets]):
                        
                        if target_name not in target_configs:
                            continue
                            
                        t_conf = target_configs[target_name]
                        x, y, z = target_data[0:3]
                        r_rad, p_rad, yw_rad = target_data[3:6]
                        r_deg, p_deg, yw_deg = np.rad2deg([r_rad, p_rad, yw_rad])

                        print(f"[{i+1}/{num_targets}] Target '{target_name}'")
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
