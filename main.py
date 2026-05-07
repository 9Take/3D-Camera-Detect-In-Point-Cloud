import cv2
import time
import yaml
import json
import os
from communication.realsense import DepthCamera
from communication.plc_comm import send_to_plc
from core.detector import ObjectDetector
from core.transformer import PointCloudTransformer

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    
    # สร้างโฟลเดอร์สำหรับ Data ถ้ายังไม่มี
    os.makedirs(os.path.dirname(config['paths']['position_mem']), exist_ok=True)
    os.makedirs(config['paths']['save_dir'], exist_ok=True)

    print("[INIT] Initializing Camera and Modules...")
    cam = DepthCamera(config['camera']['resolution_width'], config['camera']['resolution_height'])
    detector = ObjectDetector(config['paths']['template_dir'])
    transformer = PointCloudTransformer(cam, config['camera']['resolution_width'], config['camera']['resolution_height'])

    lock_start_time = None
    trigger_extraction = False

    while True:
        ret, depth_raw, color_raw = cam.get_raw_frame()
        if not ret: continue

        color_frame = color_raw.get_data()
        
        # 1. Detect 2D Image
        detected_pixels, detected_names, display_frame = detector.detect(
            color_frame, config['camera']['resolution_width'], config['camera']['resolution_height']
        )

        # 2. Timer Logic
        if 'A' in detected_names:
            if lock_start_time is None: lock_start_time = time.time()
            elapsed_time = time.time() - lock_start_time
            if elapsed_time >= config['app']['timer_lock_seconds']:
                trigger_extraction = True
        else:
            lock_start_time = None
            trigger_extraction = False

        # 3. 3D Extraction & External Comms
        if trigger_extraction:
            print("[PROCESSING] Timer Reached! Extracting Point Cloud...")
            # ดึง 6-DOF จาก Transformer
            extracted_6dof = transformer.extract_3d_data(config['app']['frames_for_averaging'])
            
            if 'A' in extracted_6dof:
                # บันทึกลง JSON
                with open(config['paths']['position_mem'], 'w') as f:
                    json.dump({"target_A": extracted_6dof['A'], "timestamp": time.time()}, f, indent=4)
                
                # ส่งเข้า PLC
                send_to_plc(
                    config['plc']['ip'], 
                    config['plc']['port'], 
                    config['plc']['start_d_reg'], 
                    extracted_6dof['A']
                )

            lock_start_time = None
            trigger_extraction = False

    cam.release()

if __name__ == '__main__':
    main()