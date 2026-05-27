import pyrealsense2 as rs
import numpy as np

class DepthCamera:
    def __init__(self, resolution_width, resolution_height):
        self.pipeline = rs.pipeline()
        config = rs.config()
        
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        depth_sensor = device.first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        # Set Visual Preset to "High Density" to help with shiny copper surfaces
        depth_sensor.set_option(rs.option.visual_preset, 4)

        # Initialize Filters
        self.spatial = rs.spatial_filter()       
        self.temporal = rs.temporal_filter()     
        self.hole_filling = rs.hole_filling_filter(2) 

        align_to = rs.stream.color
        self.align = rs.align(align_to)

        config.enable_stream(rs.stream.depth, resolution_width, resolution_height, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.bgr8, 30)
        
        self.pipeline.start(config)
       
    def get_frame(self):
        """Returns True/False and Numpy arrays for OpenCV"""
        ret, depth_frame, color_frame = self.get_raw_frame() # Re-use logic below
        if not ret:
            return False, None, None

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        return True, depth_image, color_image

    def get_raw_frame(self):
        """Returns True/False and RealSense frame objects for PointCloud/Templates"""
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            return False, None, None

        # Apply Filters to the raw frames
        depth_frame = self.spatial.process(depth_frame)
        depth_frame = self.temporal.process(depth_frame)
        depth_frame = self.hole_filling.process(depth_frame)
            
        return True, depth_frame, color_frame

    def get_color_intrinsics(self):
        """Returns (camera_matrix, dist_coeffs) for the color stream."""
        profile = self.pipeline.get_active_profile()
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        camera_matrix = np.array([[intr.fx, 0, intr.ppx],
                                  [0, intr.fy, intr.ppy],
                                  [0, 0, 1]], dtype=np.float64)
        dist_coeffs = np.array(intr.coeffs, dtype=np.float64)
        return camera_matrix, dist_coeffs

    def release(self):
        self.pipeline.stop()