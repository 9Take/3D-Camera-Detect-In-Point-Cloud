import math
import numpy as np
import open3d as o3d
import cv2

def rotation_matrix_to_euler_angles(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    if sy >= 1e-6:
        x, y, z = math.atan2(R[2, 1], R[2, 2]), math.atan2(-R[2, 0], sy), math.atan2(R[1, 0], R[0, 0])
    else:
        x, y, z = math.atan2(-R[1, 2], R[1, 1]), math.atan2(-R[2, 0], sy), 0
    return np.rad2deg([x, y, z])

class PointCloudTransformer:
    def __init__(self, camera, res_width, res_height):
        self.camera = camera
        self.res_width = res_width
        self.res_height = res_height

    def extract_3d_data(self, frames_count):
        # (นำ Logic while frames_captured < 70 ในการหา depth_sum เดิมมาใส่ที่นี่)
        # คำนวณ RGBD Image และสร้าง pcd 
        # คืนค่า pcd และ avg_depth เพื่อนำไปหาตำแหน่งเป้าหมายต่อ
        pass