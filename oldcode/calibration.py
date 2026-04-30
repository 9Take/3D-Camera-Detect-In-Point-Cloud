import numpy as np
import cv2
import json
import os
from realsense_depth import DepthCamera

class CameraCalibration:
    """
    Calibration class for RealSense D435 camera
    Uses checkerboard pattern for calibration
    """
    
    def __init__(self, checkerboard_size=(9, 6), square_size=0.025):
        """
        Args:
            checkerboard_size: (width, height) of checkerboard corners
            square_size: Size of each square in meters (default 2.5cm)
        """
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        self.calibration_file = "calibration_data.json"
        self.calibration_matrix = np.eye(4)  # 4x4 transformation matrix
        self.camera_offset = np.array([0.0, 0.0, 0.0])
        
    def capture_calibration_frames(self, num_frames=10):
        """
        Capture frames with checkerboard for calibration
        """
        resolution_width, resolution_height = (640, 480)
        camera = DepthCamera(resolution_width, resolution_height)
        
        print(f"\n{'='*60}")
        print("CAMERA CALIBRATION - Checkerboard Detection")
        print(f"{'='*60}")
        print(f"กำลังจับภาพ {num_frames} เฟรม...")
        print("แสดง checkerboard ({} x {}) ให้ชัดเจน".format(*self.checkerboard_size))
        print("หลังจากจับภาพเรียบร้อย ให้ทำให้ checkerboard หมดจากหน้าจอ")
        print("กดปุ่มใดก็ได้เพื่อเริ่ม...")
        input()
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objpoints = []  # 3D points in real world space
        imgpoints = []  # 2D points in image plane
        
        frame_count = 0
        while frame_count < num_frames:
            ret, depth_frame, color_frame = camera.get_frame()
            if not ret:
                continue
            
            gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
            ret_cb, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
            
            display_frame = color_frame.copy()
            
            if ret_cb:
                # Refine corner positions
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                # Draw checkerboard
                cv2.drawChessboardCorners(display_frame, self.checkerboard_size, corners2, ret_cb)
                
                # Prepare object points (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
                objp = np.zeros((self.checkerboard_size[0]*self.checkerboard_size[1], 3), np.float32)
                objp[:, :2] = np.mgrid[0:self.checkerboard_size[0], 0:self.checkerboard_size[1]].T.reshape(-1, 2)
                objp *= self.square_size
                
                objpoints.append(objp)
                imgpoints.append(corners2)
                frame_count += 1
                
                cv2.putText(display_frame, f"Captured: {frame_count}/{num_frames}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "No checkerboard detected", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("Calibration - Checkerboard Detection", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        camera.release()
        cv2.destroyAllWindows()
        
        return objpoints, imgpoints
    
    def compute_calibration(self, objpoints, imgpoints, image_size=(640, 480)):
        """
        Compute camera calibration matrix and distortion coefficients
        """
        print("\nการคำนวณเมทริกซ์การปรับเทียบ...")
        
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, image_size, None, None
        )
        
        print(f"Calibration RMS Error: {ret:.4f}")
        print(f"\nCamera Matrix (K):")
        print(mtx)
        print(f"\nDistortion Coefficients:")
        print(dist)
        
        # Store calibration data
        self.camera_matrix = mtx
        self.dist_coeffs = dist
        self.rvecs = rvecs
        self.tvecs = tvecs
        
        return mtx, dist
    
    def compute_offset(self, reference_point=None):
        """
        Compute camera offset from a reference point
        Args:
            reference_point: [x, y, z] coordinates of reference point
        """
        if reference_point is None:
            reference_point = [0, 0, 0]
        
        self.camera_offset = np.array(reference_point)
        print(f"\nกำหนด Camera Offset: {self.camera_offset}")
        
    def save_calibration(self):
        """
        Save calibration data to JSON file
        """
        calib_data = {
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coefficients": self.dist_coeffs.tolist(),
            "camera_offset": self.camera_offset.tolist(),
            "checkerboard_size": self.checkerboard_size,
            "square_size": self.square_size
        }
        
        with open(self.calibration_file, 'w') as f:
            json.dump(calib_data, f, indent=4)
        
        print(f"\n✓ บันทึกข้อมูลการปรับเทียบลงไฟล์: {self.calibration_file}")
        
    def load_calibration(self):
        """
        Load calibration data from JSON file
        """
        if not os.path.exists(self.calibration_file):
            print(f"ไฟล์การปรับเทียบไม่พบ: {self.calibration_file}")
            return False
        
        with open(self.calibration_file, 'r') as f:
            calib_data = json.load(f)
        
        self.camera_matrix = np.array(calib_data["camera_matrix"])
        self.dist_coeffs = np.array(calib_data["dist_coefficients"])
        self.camera_offset = np.array(calib_data["camera_offset"])
        
        print(f"\n✓ โหลดข้อมูลการปรับเทียบจากไฟล์: {self.calibration_file}")
        return True


def run_calibration():
    """
    Run the complete calibration process
    """
    calib = CameraCalibration(checkerboard_size=(9, 6), square_size=0.025)
    
    # Step 1: Capture frames
    objpoints, imgpoints = calib.capture_calibration_frames(num_frames=10)
    
    if len(objpoints) < 3:
        print("ไม่สามารถเก็บรวบรวมจุดสำหรับการปรับเทียบ")
        return
    
    # Step 2: Compute calibration
    mtx, dist = calib.compute_calibration(objpoints, imgpoints)
    
    # Step 3: Set offset (optional)
    print("\nต้องการตั้ง camera offset หรือไม่? (y/n): ", end="")
    if input().lower() == 'y':
        x = float(input("X offset (m): ") or "0")
        y = float(input("Y offset (m): ") or "0")
        z = float(input("Z offset (m): ") or "0")
        calib.compute_offset([x, y, z])
    
    # Step 4: Save calibration
    calib.save_calibration()
    
    print("\n" + "="*60)
    print("การปรับเทียบเสร็จสมบูรณ์!")
    print("="*60)


if __name__ == '__main__':
    run_calibration()
