from camera_detector import CameraDetector
import logging

# Configure logging to print to console
logging.basicConfig(level=logging.DEBUG)

print("Detecting cameras...")
cameras = CameraDetector.detect_all_cameras()
CameraDetector.print_camera_list(cameras)

for cam in cameras:
    print(f"Device {cam.device_id}:")
    print(f"  Name: {cam.name}")
    print(f"  Resolution: {cam.resolution}")
    print(f"  Type: {cam.camera_type}")
    print(f"  Is Thermal: {cam.is_thermal()}")
    print(f"  Is RGB: {cam.is_rgb()}")
