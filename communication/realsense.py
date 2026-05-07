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
        
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        
        config.enable_stream(rs.stream.depth, resolution_width, resolution_height, rs.format.z16, 6)
        config.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.bgr8, 30)
        self.pipeline.start(config)

    def get_raw_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            return False, None, None
        return True, depth_frame, color_frame

    def get_depth_scale(self):
        return self.depth_scale

    def release(self):
        self.pipeline.stop()