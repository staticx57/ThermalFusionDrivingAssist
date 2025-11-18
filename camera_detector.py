"""
Camera Detection Utility
Automatically detect and identify FLIR Boson cameras
"""
import cv2
import subprocess
import re
import logging
from typing import List, Dict, Optional
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CameraType(Enum):
    """Camera type classification"""
    THERMAL = "Thermal"
    RGB = "RGB"
    UNKNOWN = "Unknown"


class CameraInfo:
    """Information about a detected camera"""

    def __init__(self, device_id: int, name: str, driver: str, resolution: tuple):
        self.device_id = device_id
        self.name = name
        self.driver = driver
        self.resolution = resolution
        self.is_flir = 'flir' in name.lower() or 'boson' in name.lower()
        self.camera_type = self._detect_camera_type()

    def _detect_camera_type(self) -> CameraType:
        """
        Detect camera type based on resolution and name

        Returns:
            CameraType enum
        """
        # FLIR Boson thermal camera resolutions
        thermal_resolutions = [
            (640, 512),  # Boson 640
            (320, 256),  # Boson 320
            (512, 640),  # Boson 640 (rotated)
            (256, 320),  # Boson 320 (rotated)
        ]

        # Check resolution first (most reliable)
        if self.resolution in thermal_resolutions:
            return CameraType.THERMAL

        # Check name for thermal keywords
        thermal_keywords = ['flir', 'boson', 'thermal', 'radiometric', 'lwir']
        name_lower = self.name.lower()
        if any(keyword in name_lower for keyword in thermal_keywords):
            return CameraType.THERMAL

        # RGB camera indicators
        rgb_keywords = ['rgb', 'webcam', 'usb', 'hd', 'firefly', 'color']
        if any(keyword in name_lower for keyword in rgb_keywords):
            return CameraType.RGB

        # Common RGB resolutions (HD, FHD, 4K, etc.)
        common_rgb_resolutions = [
            (640, 480),   # VGA
            (1280, 720),  # HD
            (1920, 1080), # Full HD
            (3840, 2160), # 4K
            (2592, 1944), # 5MP
        ]
        if self.resolution in common_rgb_resolutions:
            return CameraType.RGB

        # Unknown type
        return CameraType.UNKNOWN

    def is_thermal(self) -> bool:
        """Check if this camera is a thermal camera"""
        return self.camera_type == CameraType.THERMAL

    def is_rgb(self) -> bool:
        """Check if this camera is an RGB camera"""
        return self.camera_type == CameraType.RGB

    def __str__(self):
        type_str = f" [{self.camera_type.value}]"
        return f"[{self.device_id}] {self.name} ({self.resolution[0]}x{self.resolution[1]}){type_str}"


class CameraDetector:
    """Detect and identify connected cameras"""

    @staticmethod
    def detect_all_cameras() -> List[CameraInfo]:
        """
        Detect all available cameras (cross-platform: Linux/Windows/macOS)

        Returns:
            List of CameraInfo objects
        """
        import platform
        cameras = []
        system = platform.system()

        if system == "Linux":
            # Try v4l2-ctl first (Linux)
            v4l2_cameras = CameraDetector._detect_v4l2()
            if v4l2_cameras:
                cameras.extend(v4l2_cameras)
            else:
                # Fallback: probe video devices directly
                cameras = CameraDetector._probe_video_devices()
        elif system == "Windows":
            # Windows: probe cameras using DirectShow/MSMF
            cameras = CameraDetector._probe_windows_cameras()
        else:
            # macOS or other: probe using default backend
            cameras = CameraDetector._probe_video_devices()

        return cameras

    @staticmethod
    def _detect_v4l2() -> List[CameraInfo]:
        """Detect cameras using v4l2-ctl"""
        cameras = []

        try:
            # Get list of video devices
            result = subprocess.run(
                ['v4l2-ctl', '--list-devices'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                cameras = CameraDetector._parse_v4l2_output(result.stdout)

        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("v4l2-ctl not available or timed out")

        return cameras

    @staticmethod
    def _parse_v4l2_output(output: str) -> List[CameraInfo]:
        """Parse v4l2-ctl output"""
        cameras = []
        lines = output.strip().split('\n')

        current_name = ""
        for line in lines:
            line = line.strip()

            # Device name line (doesn't start with /)
            if line and not line.startswith('/dev/'):
                # Remove trailing colon and model info
                current_name = line.rstrip(':').strip()

            # Device path line
            elif line.startswith('/dev/video'):
                device_path = line.strip()
                # Extract device number
                match = re.search(r'/dev/video(\d+)', device_path)
                if match:
                    device_id = int(match.group(1))

                    # Get resolution for this device
                    resolution = CameraDetector._get_device_resolution(device_id)

                    cameras.append(CameraInfo(
                        device_id=device_id,
                        name=current_name if current_name else f"Camera {device_id}",
                        driver="v4l2",
                        resolution=resolution
                    ))

        return cameras

    @staticmethod
    def _get_device_resolution(device_id: int) -> tuple:
        """Get native resolution of a video device"""
        try:
            # Try to query formats
            result = subprocess.run(
                ['v4l2-ctl', '-d', f'/dev/video{device_id}', '--list-formats-ext'],
                capture_output=True,
                text=True,
                timeout=3
            )

            if result.returncode == 0:
                # Look for Size: line
                matches = re.findall(r'Size:\s+Discrete\s+(\d+)x(\d+)', result.stdout)
                if matches:
                    # Return the first (usually native) resolution
                    width, height = matches[0]
                    return (int(width), int(height))

            # Fallback: try opening with OpenCV
            return CameraDetector._probe_opencv_resolution(device_id)

        except:
            return CameraDetector._probe_opencv_resolution(device_id)

    @staticmethod
    def _probe_opencv_resolution(device_id: int) -> tuple:
        """Probe resolution using OpenCV"""
        try:
            cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                return (width, height)
        except:
            pass

        return (0, 0)

    @staticmethod
    def _probe_video_devices(max_devices: int = 10) -> List[CameraInfo]:
        """Probe /dev/video* devices directly (Linux)"""
        cameras = []

        for device_id in range(max_devices):
            try:
                cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)

                if cap.isOpened():
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()

                    cameras.append(CameraInfo(
                        device_id=device_id,
                        name=f"Camera {device_id}",
                        driver="opencv",
                        resolution=(width, height)
                    ))

            except:
                pass

        return cameras

    @staticmethod
    def _probe_windows_cameras(max_devices: int = 1) -> List[CameraInfo]:
        """
        Probe cameras on Windows using DirectShow/MSMF

        Note: Reduced max_devices to 1 to avoid driver-level crashes.
        Windows OpenCV drivers can cause fatal exceptions (0xc06d007e)
        when accessing non-existent camera indices that Python cannot catch.
        Users must manually specify camera ID if they have multiple cameras.
        """
        cameras = []
        consecutive_failures = 0
        MAX_CONSECUTIVE_FAILURES = 1  # Stop after 1 failure on Windows

        for device_id in range(max_devices):
            cap = None
            try:
                logger.debug(f"Probing Windows camera index {device_id}...")

                # Try DirectShow first (better compatibility with UVC devices)
                # Wrap in nested try to catch driver-level exceptions
                try:
                    cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
                except Exception as e:
                    logger.debug(f"DirectShow failed for camera {device_id}: {e}")
                    cap = None

                if cap is None or not cap.isOpened():
                    # Fall back to MSMF
                    try:
                        if cap is not None:
                            cap.release()
                        cap = cv2.VideoCapture(device_id, cv2.CAP_MSMF)
                    except Exception as e:
                        logger.debug(f"MSMF failed for camera {device_id}: {e}")
                        cap = None

                if cap is not None and cap.isOpened():
                    try:
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                        # Try to get camera name (backend-specific property)
                        name = f"Camera {device_id}"
                        try:
                            # Some backends support camera name property
                            backend_name = cap.getBackendName()
                            name = f"Camera {device_id} ({backend_name})"
                        except:
                            pass

                        # Detect if this looks like a FLIR Boson based on resolution
                        if (width, height) in [(640, 512), (320, 256)]:
                            name = f"FLIR Boson {width}x{height} (Camera {device_id})"

                        cap.release()
                        cap = None

                        cameras.append(CameraInfo(
                            device_id=device_id,
                            name=name,
                            driver="DirectShow/MSMF",
                            resolution=(width, height)
                        ))

                        logger.info(f"Detected: {name} at index {device_id}")
                        consecutive_failures = 0  # Reset failure counter

                    except Exception as e:
                        logger.debug(f"Failed to read properties from camera {device_id}: {e}")
                        consecutive_failures += 1
                else:
                    consecutive_failures += 1

            except Exception as e:
                logger.debug(f"Failed to probe camera {device_id}: {e}")
                consecutive_failures += 1
            finally:
                if cap is not None:
                    try:
                        cap.release()
                    except:
                        pass

            # Early termination if too many consecutive failures
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                logger.debug(f"Stopping camera probe after {consecutive_failures} consecutive failures")
                break

        logger.info(f"Windows camera probe complete: {len(cameras)} camera(s) found")
        return cameras

    @staticmethod
    def find_thermal_camera() -> Optional[CameraInfo]:
        """
        Find thermal camera automatically

        Returns:
            CameraInfo for thermal camera or None if not found
        """
        import time
        cameras = CameraDetector.detect_all_cameras()

        # Give cameras time to fully release after detection scan
        time.sleep(0.3)

        # First priority: cameras identified as thermal
        for cam in cameras:
            if cam.is_thermal():
                logger.info(f"Found thermal camera: {cam}")
                return cam

        # Second priority: cameras with FLIR/Boson in name
        for cam in cameras:
            if cam.is_flir:
                logger.info(f"Found FLIR camera (assumed thermal): {cam}")
                return cam

        # No thermal camera detected
        logger.warning("No thermal camera detected")
        return None

    @staticmethod
    def find_rgb_camera() -> Optional[CameraInfo]:
        """
        Find RGB camera automatically

        Returns:
            CameraInfo for RGB camera or None if not found
        """
        import time
        cameras = CameraDetector.detect_all_cameras()

        # Give cameras time to fully release after detection scan
        time.sleep(0.3)

        # First priority: cameras identified as RGB
        for cam in cameras:
            if cam.is_rgb():
                logger.info(f"Found RGB camera: {cam}")
                return cam

        # Second priority: any non-thermal camera
        for cam in cameras:
            if not cam.is_thermal() and cam.camera_type != CameraType.UNKNOWN:
                logger.info(f"Found camera (assumed RGB): {cam}")
                return cam

        # No RGB camera detected
        logger.warning("No RGB camera detected")
        return None

    @staticmethod
    def find_flir_boson() -> Optional[CameraInfo]:
        """
        Find FLIR Boson camera automatically (backward compatibility)

        Returns:
            CameraInfo for FLIR Boson or None if not found
        """
        # Use new thermal camera detection
        return CameraDetector.find_thermal_camera()

    @staticmethod
    def print_camera_list(cameras: List[CameraInfo]):
        """Print formatted list of cameras"""
        print("\n" + "="*60)
        print("DETECTED CAMERAS:")
        print("="*60)

        if not cameras:
            print("  No cameras detected")
        else:
            for cam in cameras:
                marker = " [FLIR]" if cam.is_flir else ""
                print(f"  {cam}{marker}")

        print("="*60 + "\n")