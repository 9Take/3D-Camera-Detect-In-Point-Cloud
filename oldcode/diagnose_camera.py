# -*- coding: utf-8 -*-
"""
RealSense Camera Diagnostic Tool
Helps troubleshoot camera connection issues
"""

import sys
import os

print("\n" + "="*70)
print("REALSENSE CAMERA DIAGNOSTIC TOOL")
print("="*70 + "\n")

# Step 1: Check Python version
print("[1/5] Checking Python version...")
print("Python version: {}".format(sys.version))
if sys.version_info >= (3, 5):
    print("✓ Python version OK\n")
else:
    print("✗ Python version too old (need 3.5+)\n")

# Step 2: Check pyrealsense2 installation
print("[2/5] Checking pyrealsense2 module...")
try:
    import pyrealsense2 as rs
    print("✓ pyrealsense2 is installed")
    print("Version: {}\n".format(rs.__version__))
except ImportError as e:
    print("✗ pyrealsense2 NOT installed")
    print("Error: {}".format(str(e)))
    print("\nInstall with:")
    print("  pip install pyrealsense2\n")
    sys.exit(1)

# Step 3: Check for connected devices
print("[3/5] Scanning for connected RealSense devices...")
try:
    context = rs.context()
    devices = context.query_devices()
    
    num_devices = len(devices)
    print("Found {} device(s)".format(num_devices))
    
    if num_devices == 0:
        print("\n✗ NO DEVICES FOUND!")
        print("\nTroubleshooting steps:")
        print("  1. Check USB connection:")
        print("     - Ensure cable is firmly connected")
        print("     - Try different USB 3.0 port")
        print("     - Use original RealSense cable")
        print("\n  2. Check device visibility:")
        print("     - Linux: lsusb | grep Intel")
        print("     - Windows: Device Manager")
        print("     - Mac: System Report > USB")
        print("\n  3. Install firmware drivers:")
        print("     - Ubuntu: sudo apt-get install librealsense2-dkms")
        print("     - CentOS: sudo yum install kernel-devel")
        print("\n  4. Restart the system after installing drivers\n")
    else:
        print("\n✓ Device(s) found!\n")
        
        for i, device in enumerate(devices):
            print("Device {}: {}".format(i+1, device.get_info(rs.camera_info.name)))
            print("  Serial: {}".format(device.get_info(rs.camera_info.serial_number)))
            print("  Firmware: {}".format(device.get_info(rs.camera_info.firmware_version)))
            print("  USB Port: {}".format(device.get_info(rs.camera_info.usb_type_descriptor)))
            print()

except Exception as e:
    print("✗ Error scanning devices: {}".format(str(e)))
    sys.exit(1)

# Step 4: Test camera initialization
print("[4/5] Testing camera initialization...")
try:
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Configure streams
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    
    # Start pipeline
    profile = pipeline.start(config)
    print("✓ Camera initialized successfully!\n")
    
    # Get device info
    device = profile.get_device()
    print("Active Device: {}".format(device.get_info(rs.camera_info.name)))
    print("Serial: {}".format(device.get_info(rs.camera_info.serial_number)))
    print()
    
    # Test frame capture
    print("[5/5] Testing frame capture...")
    for i in range(5):
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        
        if color_frame and depth_frame:
            print("✓ Frame {}: Color {}x{}, Depth {}x{}".format(
                i+1,
                color_frame.width, color_frame.height,
                depth_frame.width, depth_frame.height
            ))
    
    pipeline.stop()
    print("\n" + "="*70)
    print("✓ ALL TESTS PASSED - CAMERA IS WORKING!")
    print("="*70 + "\n")
    print("You can now run: python src/copper_arm_targeting.py\n")

except Exception as e:
    print("✗ Camera initialization failed!")
    print("Error: {}".format(str(e)))
    print("\nTroubleshooting:")
    print("  - Check USB cable connection")
    print("  - Try rebooting system")
    print("  - Update RealSense firmware")
    print("  - Install RealSense SDK: https://github.com/IntelRealSense/librealsense\n")
    pipeline.stop()
    sys.exit(1)
