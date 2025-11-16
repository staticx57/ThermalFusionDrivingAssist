"""
FLIR Boson Thermal Camera Interface
Optimized for Jetson Orin with GPU acceleration
"""
import cv2
import numpy as np
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FLIRBosonCamera:
    """Interface for FLIR Boson thermal camera"""

    def __init__(self, device_id: int = 0, resolution: Tuple[int, int] = (640, 512)):
        """
        Initialize FLIR Boson camera

        Args:
            device_id: Video device ID (usually /dev/video0)
            resolution: Camera resolution (640x512 for Boson 640)
        """
        self.device_id = device_id
        self.resolution = resolution
        self.cap = None
        self.is_opened = False

    def open(self) -> bool:
        """Open camera connection (cross-platform: Linux/Windows)"""
        try:
            import platform
            system = platform.system()

            # Try platform-specific backend first
            if system == "Linux":
                # Linux: Use V4L2 backend
                logger.info("Linux detected - trying V4L2 backend")
                self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_V4L2)
            elif system == "Windows":
                # Windows: Try DirectShow first, then MSMF
                logger.info("Windows detected - trying DirectShow backend")
                try:
                    self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_DSHOW)
                except:
                    logger.info("DirectShow failed, trying MSMF backend")
                    self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_MSMF)
            else:
                # macOS or other: use default backend
                logger.info(f"{system} detected - using default backend")
                self.cap = cv2.VideoCapture(self.device_id)

            # Fallback to default backend if platform-specific failed
            if not self.cap.isOpened():
                logger.warning(f"Platform-specific backend failed, trying default backend")
                self.cap = cv2.VideoCapture(self.device_id)

            if self.cap.isOpened():
                # Set camera properties for optimal performance
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency

                # Try to maximize frame rate
                self.cap.set(cv2.CAP_PROP_FPS, 60)

                self.is_opened = True
                logger.info(f"FLIR Boson camera opened successfully on device {self.device_id}")
                logger.info(f"Resolution: {self.get_actual_resolution()}")
                logger.info(f"FPS: {self.cap.get(cv2.CAP_PROP_FPS)}")
                return True
            else:
                logger.error(f"Failed to open camera on device {self.device_id}")
                return False

        except Exception as e:
            logger.error(f"Error opening camera: {e}")
            return False

    def read(self, flush_buffer: bool = False) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read frame from camera

        Args:
            flush_buffer: If True, grab and drop old frames to get the freshest frame

        Returns:
            Tuple of (success, frame) where frame is BGR format
        """
        if not self.is_opened or self.cap is None:
            return False, None

        # Flush old buffered frames to get fresher frames
        # This prevents lag when processing takes longer than frame interval
        if flush_buffer:
            # Only flush 1 frame - reduces latency by 16ms while still getting fresh frames
            # At 60 FPS, 1 frame = 16ms overhead
            for _ in range(1):
                if not self.cap.grab():
                    break
            # Retrieve the latest grabbed frame
            ret, frame = self.cap.retrieve()
        else:
            ret, frame = self.cap.read()

        if ret and frame is not None:
            # FLIR Boson outputs grayscale, convert to BGR for compatibility
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        return ret, frame

    def read_thermal(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read raw thermal frame (grayscale)

        Returns:
            Tuple of (success, frame) where frame is grayscale thermal data
        """
        if not self.is_opened or self.cap is None:
            return False, None

        ret, frame = self.cap.read()

        if ret and len(frame.shape) == 3:
            # Convert to grayscale if needed
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return ret, frame

    def get_actual_resolution(self) -> Tuple[int, int]:
        """Get actual camera resolution"""
        if self.cap is None:
            return (0, 0)

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)

    def release(self):
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()
            self.is_opened = False
            logger.info("FLIR Boson camera released")

    def __enter__(self):
        """Context manager entry"""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()