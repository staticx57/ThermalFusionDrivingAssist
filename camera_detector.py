"""
Camera Detection Utility
Automatically detect and identify FLIR Boson cameras
"""
import cv2
import subprocess
import re
import logging
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CameraInfo:
    """Information about a detected camera"""

    def __init__(self, device_id: int, name: str, driver: str, resolution: tuple):
        self.device_id = device_id
        self.name = name
        self.driver = driver
        self.resolution = resolution
        self.is_flir = 'flir' in name.lower() or 'boson' in name.lower()

    def __str__(self):
        return f"[{self.device_id}] {self.name} ({self.resolution[0]}x{self.resolution[1]})"


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
    def _probe_windows_cameras(max_devices: int = 5) -> List[CameraInfo]:
        """
        Probe cameras on Windows using DirectShow/MSMF

        Note: Reduced max_devices to 5 to avoid driver-level crashes
        when probing non-existent camera indices on Windows.
        """
        cameras = []

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

                    except Exception as e:
                        logger.debug(f"Failed to read properties from camera {device_id}: {e}")

            except Exception as e:
                logger.debug(f"Failed to probe camera {device_id}: {e}")
            finally:
                if cap is not None:
                    try:
                        cap.release()
                    except:
                        pass

        logger.info(f"Windows camera probe complete: {len(cameras)} camera(s) found")
        return cameras

    @staticmethod
    def find_flir_boson() -> Optional[CameraInfo]:
        """
        Find FLIR Boson camera automatically

        Returns:
            CameraInfo for FLIR Boson or None if not found
        """
        cameras = CameraDetector.detect_all_cameras()

        # First priority: cameras with FLIR/Boson in name
        for cam in cameras:
            if cam.is_flir:
                logger.info(f"Found FLIR Boson camera: {cam}")
                return cam

        # Second priority: cameras with Boson-typical resolutions
        boson_resolutions = [(640, 512), (320, 256)]
        for cam in cameras:
            if cam.resolution in boson_resolutions:
                logger.info(f"Found camera with Boson-like resolution: {cam}")
                return cam

        # No FLIR detected
        if cameras:
            logger.warning(f"No FLIR Boson detected. Available cameras: {[str(c) for c in cameras]}")
            return cameras[0]  # Return first camera as fallback
        else:
            logger.error("No cameras detected!")
            return None

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