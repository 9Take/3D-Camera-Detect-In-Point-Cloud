import cv2
import time
import yaml
import json
import os
import numpy as np
import argparse

from communication.realsense import DepthCamera
from communication.plc_comm import PLCCommunicator
from core.detector import ObjectDetector
from core.transformer import PointCloudTransformer

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def float_to_scaled(value, scale=1000):
    """แปลง float เป็น int สำหรับ PLC (scaled by 1000)"""
    return int(value * scale)

def main():
    parser = argparse.ArgumentParser(description="Heat Exchanger Vision System")
    parser.add_argument('--debug', action='store_true', help='เปิดหน้าต่าง Debug เพื่อดูภาพจากกล้องและการตรวจจับ')
    args = parser.parse_args()

    config = load_config()
    
    os.makedirs(os.path.dirname(config['paths']['position_mem']), exist_ok=True)
    os.makedirs(config['paths']['save_dir'], exist_ok=True)

    print("[INIT] Initializing Camera and Modules...")
    cam = DepthCamera(config['camera']['resolution_width'], config['camera']['resolution_height'])
    detector = ObjectDetector(config['paths']['template_dir'])
    transformer = PointCloudTransformer(cam, config['camera']['resolution_width'], config['camera']['resolution_height'], config['paths']['save_dir'])
    plc = PLCCommunicator(config['plc']['ip'], config['plc']['port'])

    lock_start_time = None
    trigger_extraction = False

    print("[MAIN] Starting detection loop...")
    try:
        while True:
            # Capture frame from camera
            ret, depth_raw, color_raw = cam.get_raw_frame()
            if not ret:
                print("[WARNING] Failed to get frame")
                continue
            
            color_frame = np.asanyarray(color_raw.get_data())
            h, w = color_frame.shape[:2]

            # 1. Detect วัตถุแบบ Real-time
            detected_pixels, names, display_frame = detector.detect(color_frame, w, h)
            
            # อัปเดตพิกัดล่าสุดลง JSON (เพื่อการตรวจสอบหน้าจอ)
            current_detect = {"last_seen": names, "timestamp": time.time()}
            with open(config['paths']['save_dir'] + "/current_detect.json", "w") as f:
                json.dump(current_detect, f)

            # 2. ตรวจสอบ Trigger จาก PLC (เช่น อ่านค่าจาก M100)
            try:
                trigger = plc.batchread_bitunits(config['plc']['trigger_device'], 1)
            except Exception as e:
                print(f"[WARNING] PLC read error: {e}")
                trigger = [0]
            
            if trigger and trigger[0] == 1:  # เมื่อ PLC สั่ง Trigger
                print("[PLC] Trigger Received!")
                
                # 3. คำนวณ 6-DOF ชุดปัจจุบัน
                data_6dof = transformer.extract_3d_data(detected_pixels, names, show_3d=args.debug)
                
                if 'A' in data_6dof:
                    # 4. บันทึกลง position_mem.json (Snapshot ตอนโดน Trigger)
                    final_pos = data_6dof['A']
                    save_payload = {
                        "Position_X": float(final_pos[0]), 
                        "Position_Y": float(final_pos[1]), 
                        "Position_Z": float(final_pos[2]),
                        "Roll": float(final_pos[3]), 
                        "Pitch": float(final_pos[4]), 
                        "Yaw": float(final_pos[5]),
                        "timestamp": time.time()
                    }
                    with open(config['paths']['position_mem'], "w") as f:
                        json.dump(save_payload, f, indent=4)
                    
                    # 5. ส่งข้อมูลไปยัง PLC Device ตามที่ตั้งค่าไว้ใน config
                    try:
                        dev = config['plc']['devices']
                        plc.batchwrite_wordunits(dev['x'], float_to_scaled(final_pos[0]))
                        plc.batchwrite_wordunits(dev['y'], float_to_scaled(final_pos[1]))
                        plc.batchwrite_wordunits(dev['z'], float_to_scaled(final_pos[2]))
                        plc.batchwrite_wordunits(dev['roll'], float_to_scaled(final_pos[3]))
                        plc.batchwrite_wordunits(dev['pitch'], float_to_scaled(final_pos[4]))
                        plc.batchwrite_wordunits(dev['yaw'], float_to_scaled(final_pos[5]))
                        
                        # ส่งสัญญาณตอบกลับ (Handshake Done)
                        plc.batchwrite_bitunits(config['plc']['status_device'], [1])
                        print("[PLC] Data sent successfully")
                    except Exception as e:
                        print(f"[ERROR] PLC write error: {e}")

            # Display frame if debug mode
            if args.debug and display_frame is not None:
                cv2.imshow("Detection Debug", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted by user")
    except Exception as e:
        print(f"[ERROR] Main loop error: {e}")
    finally:
        print("[CLEANUP] Releasing resources...")
        cam.release()
        cv2.destroyAllWindows()
        print("[DONE] Program finished")

if __name__ == '__main__':
    main()