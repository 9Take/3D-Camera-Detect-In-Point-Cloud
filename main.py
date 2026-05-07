import cv2
import time
import yaml
import json
import os
import argparse
import numpy as np

# อิมพอร์ตโมดูลต่างๆ (ปรับชื่อโฟลเดอร์ให้ตรงกับโครงสร้างของคุณ)
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
            
            # 1. ค้นหาวัตถุ (2D Detection) - เพิ่ม confidences
            detected_pixels, detected_names, confidences, display_frame = detector.detect(
                color_frame, config['camera']['resolution_width'], config['camera']['resolution_height']
            )

            # อัปเดตสถานะแบบ Real-time ลง JSON (สำหรับทำ Dashboard ดูผ่านเว็บได้)
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

            # 2. ตรวจสอบสัญญาณ Trigger จาก PLC
            # (ถ้า PLC ส่งค่า M100 มาเป็น 1 แปลว่าหุ่นยนต์หยุดนิ่งและพร้อมรับค่าแล้ว)
            trigger_status = plc.read_bit(config['plc']['trigger_device'])
            
            if trigger_status[0] == 1 and len(detected_pixels) > 0:
                print("\n[TRIGGER] Received signal from PLC. Extracting 6-DOF...")
                
                # 3. คำนวณพิกัด 3D แบบแม่นยำ ณ วินาทีนั้น
                extracted_6dof = transformer.extract_3d_data(
                    detected_pixels, 
                    detected_names, 
                    show_3d=args.debug
                )
                
                # วางโค้ดใหม่ส่วนนี้ต่อจาก extracted_6dof = transformer.extract_3d_data(...)
                
                if extracted_6dof:
                    print(f"\n[INFO] Found {len(extracted_6dof)} targets.")
                    
                    # --- 1. [แก้ปัญหา JSON] บันทึกค่าลง JSON ก่อนส่ง PLC ---
                    # ปรับให้บันทึกได้หลายเป้าหมายพร้อมกัน
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
                    # ----------------------------------------------------

                    # 2. ส่งจำนวนจุดที่พบไปที่ D1101 (จำกัดสูงสุดตาม max_points ใน config)
                    num_targets = min(len(extracted_6dof), config['plc']['max_points'])
                    plc.write_scaled_word(config['plc']['point_count_device'], num_targets, multiplier=1)

                    # เตรียมตัวแปรหาเลขเริ่มต้น (ดึงเลข 1001 ออกมาจาก "D1001")
                    start_reg_num = int(config['plc']['data_start_device'][1:]) 
                    
                    # 3. วนลูปส่งข้อมูลทีละชุด
                    for i, (target_name, target_data) in enumerate(list(extracted_6dof.items())[:num_targets]):
                        
                        # target_data ตอนนี้คือ [X, Y, Z, Roll_rad, Pitch_rad, Yaw_rad]
                        x, y, z = target_data[0:3]
                        r_rad, p_rad, yw_rad = target_data[3:6]
                        
                        # แปลงเป็น Degree สำหรับการ Print
                        r_deg, p_deg, yw_deg = np.rad2deg([r_rad, p_rad, yw_rad])

                        # [TODO 1] ปริ้นทั้ง Rad และ Degree ในบรรทัดเดียว
                        print(f"[{i+1}/{num_targets}] Target '{target_name}'")
                        print(f" ➔ Pos(m): X:{x:.4f}, Y:{y:.4f}, Z:{z:.4f}")
                        print(f" ➔ Ori(rad): R:{r_rad:.4f}, P:{p_rad:.4f}, Y:{yw_rad:.4f}")
                        print(f" ➔ Ori(deg): R:{r_deg:.2f}°, P:{p_deg:.2f}°, Y:{yw_deg:.2f}°")
                        
                        current_offset = start_reg_num + (i * config['plc']['registers_per_point'])
                        
                        # ส่ง Position (เมตร * 1000 = มิลลิเมตร เพื่อรักษาทศนิยม 3 ตำแหน่งในรูปแบบ Int)
                        plc.write_scaled_word(f"D{current_offset}",     x, multiplier=1000) 
                        plc.write_scaled_word(f"D{current_offset + 1}", y, multiplier=1000)
                        plc.write_scaled_word(f"D{current_offset + 2}", z, multiplier=1000)
                        
                        # ส่ง Orientation เป็น Radian (คูณ 1000 เพื่อรักษาทศนิยม 3 ตำแหน่ง)
                        # เช่น 0.1234 rad -> PLC จะได้รับ 123
                        plc.write_scaled_word(f"D{current_offset + 3}", r_rad, multiplier=1000)
                        plc.write_scaled_word(f"D{current_offset + 4}", p_rad, multiplier=1000)
                        plc.write_scaled_word(f"D{current_offset + 5}", yw_rad, multiplier=1000)
                    print(f"[PLC] Sent 6-DOF Data for {num_targets} points successfully.")

                    # 4. ส่งสัญญาณกลับ (Handshake Done) ไปที่ M1001 ว่าคำนวณและส่งค่าเสร็จแล้ว
                    plc.write_bit(config['plc']['status_device'], 1)
                    
                    # หน่วงเวลาเล็กน้อยให้ PLC รับรู้ แล้วเคลียร์สัญญาณ Done
                    time.sleep(0.5)
                    plc.write_bit(config['plc']['status_device'], 0)
                    
                    print("[PLC] Handshake complete. Waiting for next trigger...")
                    
                    # เคลียร์ Trigger M1000 กลับเป็น 0
                    plc.write_bit(config['plc']['trigger_device'], 0) 

    except KeyboardInterrupt:
        print("\n[INFO] Exiting program...")
    finally:
        plc.disconnect()
        cam.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()