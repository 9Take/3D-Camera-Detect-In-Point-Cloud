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
                
                # สมมติว่าต้องการส่งค่าของเป้าหมาย 'A' (ถ้าในอนาคตมีหลายเป้าหมาย สามารถวนลูปได้)
                if 'A' in extracted_6dof:
                    target_data = extracted_6dof['A'] # [X, Y, Z, Roll, Pitch, Yaw]
                    
                    # 4. บันทึก Memory State (Snapshot) ลง JSON
                    memory_state = {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "target": "A",
                        "Position_X": round(target_data[0], 4),
                        "Position_Y": round(target_data[1], 4),
                        "Position_Z": round(target_data[2], 4),
                        "Roll": round(target_data[3], 2),
                        "Pitch": round(target_data[4], 2),
                        "Yaw": round(target_data[5], 2)
                    }
                    with open(config['paths']['position_mem'], "w") as f:
                        json.dump(memory_state, f, indent=4)
                    print(f"[LOG] Memory saved to {config['paths']['position_mem']}")

                    # 5. ส่งข้อมูลไปยัง PLC ตาม Data Register ที่ตั้งใน config
                    dev = config['plc']['devices']
                    # ส่ง Position (คูณ 1000 เพื่อแปลง เมตร เป็น มิลลิเมตร)
                    plc.write_scaled_word(dev['x'], target_data[0], multiplier=1000)
                    plc.write_scaled_word(dev['y'], target_data[1], multiplier=1000)
                    plc.write_scaled_word(dev['z'], target_data[2], multiplier=1000)
                    # ส่ง Orientation (คูณ 100 เพื่อเก็บทศนิยม 2 ตำแหน่งสำหรับองศา)
                    plc.write_scaled_word(dev['roll'], target_data[3], multiplier=100)
                    plc.write_scaled_word(dev['pitch'], target_data[4], multiplier=100)
                    plc.write_scaled_word(dev['yaw'], target_data[5], multiplier=100)
                    
                    print(f"[PLC] 6-DOF Data sent successfully to D-Registers.")

                    # 6. ส่งสัญญาณกลับ (Handshake Done) ไปที่ M101 ว่าคำนวณเสร็จแล้ว
                    plc.write_bit(config['plc']['status_device'], 1)
                    
                    # หน่วงเวลาเล็กน้อยให้ PLC รับรู้ แล้วเคลียร์สัญญาณ Done
                    time.sleep(0.5)
                    plc.write_bit(config['plc']['status_device'], 0)
                    
                    print("[PLC] Handshake complete. Waiting for next trigger...")
                    
                    # เคลียร์ Trigger ตัวเองฝั่ง Python (หรือในความจริง PLC ต้องเป็นคนเคลียร์ M100 เอง)
                    plc.write_bit(config['plc']['trigger_device'], 0) 

    except KeyboardInterrupt:
        print("\n[INFO] Exiting program...")
    finally:
        plc.disconnect()
        cam.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()