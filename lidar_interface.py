"""
LiDAR Interface - Abstract base class for LiDAR sensors
Supports Pandar 40P and future LiDAR sensors for sensor fusion
"""
import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LiDARStatus(Enum):
    """LiDAR connection status"""
    DISCONNECTED = "Disconnected"
    CONNECTING = "Connecting..."
    CONNECTED = "Connected"
    ERROR = "Error"
    NOT_CONFIGURED = "Not Configured"


class PointCloud:
    """LiDAR point cloud data structure"""
    def __init__(self, points: np.ndarray, intensities: Optional[np.ndarray] = None,
                 timestamp: float = 0.0):
        """
        Initialize point cloud

        Args:
            points: Nx3 array of (x, y, z) coordinates in meters
            intensities: N array of intensity values (0-255)
            timestamp: Timestamp of the scan
        """
        self.points = points
        self.intensities = intensities
        self.timestamp = timestamp
        self.num_points = len(points) if points is not None else 0

    def get_range_image(self, width: int = 1024, height: int = 64) -> np.ndarray:
        """Convert point cloud to range image for visualization"""
        # Placeholder implementation
        return np.zeros((height, width), dtype=np.float32)


class LiDARInterface(ABC):
    """Abstract base class for LiDAR sensors"""

    def __init__(self, name: str = "LiDAR"):
        self.name = name
        self.status = LiDARStatus.DISCONNECTED
        self.is_open = False
        self.specs = {
            "range_max": 0.0,  # meters
            "accuracy": 0.0,   # centimeters
            "fov_horizontal": 0.0,  # degrees
            "fov_vertical": 0.0,    # degrees
            "scan_rate": 0.0,   # Hz
        }

    @abstractmethod
    def connect(self, port: Optional[str] = None, ip: Optional[str] = None) -> bool:
        """
        Connect to LiDAR sensor

        Args:
            port: Serial port (if applicable)
            ip: IP address for network-based LiDAR

        Returns:
            True if connected successfully
        """
        pass

    @abstractmethod
    def disconnect(self):
        """Disconnect from LiDAR sensor"""
        pass

    @abstractmethod
    def read_point_cloud(self) -> Optional[PointCloud]:
        """
        Read point cloud from LiDAR

        Returns:
            PointCloud object or None if read failed
        """
        pass

    def get_status(self) -> LiDARStatus:
        """Get current connection status"""
        return self.status

    def get_specs(self) -> Dict[str, float]:
        """Get LiDAR specifications"""
        return self.specs

    def is_connected(self) -> bool:
        """Check if LiDAR is connected"""
        return self.is_open and self.status == LiDARStatus.CONNECTED


class PandarLiDAR(LiDARInterface):
    """
    Hesai Pandar 40P LiDAR Interface

    Specifications:
    - Range: 0.3m - 200m
    - Accuracy: ±2cm
    - FOV: 360° (H) x 40° (V)
    - Scan rate: 10Hz / 20Hz
    - Channels: 40
    - Point rate: 720,000 pts/sec
    """

    def __init__(self):
        super().__init__(name="Pandar 40P")
        self.specs = {
            "range_max": 200.0,      # 200 meters max range
            "accuracy": 2.0,         # ±2cm accuracy
            "fov_horizontal": 360.0, # 360° horizontal FOV
            "fov_vertical": 40.0,    # 40° vertical FOV
            "scan_rate": 10.0,       # 10Hz default scan rate
            "channels": 40,          # 40 laser channels
            "point_rate": 720000,    # 720k points/second
        }
        self.ip_address = None
        self.port = 2368  # Default Pandar UDP port

    def connect(self, port: Optional[str] = None, ip: Optional[str] = "192.168.1.201") -> bool:
        """
        Connect to Pandar 40P via UDP

        Args:
            port: UDP port (default 2368)
            ip: LiDAR IP address (default 192.168.1.201)

        Returns:
            True if connected
        """
        try:
            self.status = LiDARStatus.CONNECTING
            self.ip_address = ip
            if port:
                self.port = int(port)

            logger.info(f"Connecting to Pandar 40P at {self.ip_address}:{self.port}...")

            # TODO: Implement actual UDP connection
            # import socket
            # self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # self.socket.bind(('', self.port))
            # self.socket.settimeout(1.0)

            # Placeholder: Simulate connection failure
            logger.warning("Pandar 40P driver not implemented - placeholder only")
            self.status = LiDARStatus.NOT_CONFIGURED
            self.is_open = False
            return False

        except Exception as e:
            logger.error(f"Failed to connect to Pandar 40P: {e}")
            self.status = LiDARStatus.ERROR
            self.is_open = False
            return False

    def disconnect(self):
        """Disconnect from Pandar 40P"""
        if self.is_open:
            # TODO: Close UDP socket
            # self.socket.close()
            logger.info("Pandar 40P disconnected")

        self.is_open = False
        self.status = LiDARStatus.DISCONNECTED

    def read_point_cloud(self) -> Optional[PointCloud]:
        """
        Read point cloud from Pandar 40P

        Returns:
            PointCloud object or None
        """
        if not self.is_connected():
            return None

        try:
            # TODO: Implement actual UDP packet reading and parsing
            # data, addr = self.socket.recvfrom(65536)
            # points = self._parse_pandar_packet(data)
            # return PointCloud(points)

            # Placeholder: Return empty point cloud
            return PointCloud(
                points=np.zeros((0, 3), dtype=np.float32),
                intensities=None,
                timestamp=0.0
            )

        except Exception as e:
            logger.error(f"Error reading from Pandar 40P: {e}")
            return None


class MockLiDAR(LiDARInterface):
    """Mock LiDAR for testing (simulates point cloud data)"""

    def __init__(self):
        super().__init__(name="Mock LiDAR")
        self.specs = {
            "range_max": 100.0,
            "accuracy": 5.0,
            "fov_horizontal": 270.0,
            "fov_vertical": 30.0,
            "scan_rate": 10.0,
        }

    def connect(self, port: Optional[str] = None, ip: Optional[str] = None) -> bool:
        """Simulate connection"""
        logger.info("Mock LiDAR connected (simulation mode)")
        self.status = LiDARStatus.CONNECTED
        self.is_open = True
        return True

    def disconnect(self):
        """Simulate disconnection"""
        logger.info("Mock LiDAR disconnected")
        self.status = LiDARStatus.DISCONNECTED
        self.is_open = False

    def read_point_cloud(self) -> Optional[PointCloud]:
        """Generate simulated point cloud"""
        if not self.is_connected():
            return None

        # Generate random points for testing
        num_points = 10000
        points = np.random.rand(num_points, 3) * 50  # Random points within 50m
        intensities = np.random.randint(0, 256, num_points, dtype=np.uint8)

        return PointCloud(
            points=points,
            intensities=intensities,
            timestamp=0.0
        )


def create_lidar(lidar_type: str = "pandar") -> LiDARInterface:
    """
    Factory function to create LiDAR instance

    Args:
        lidar_type: Type of LiDAR ("pandar", "mock", etc.)

    Returns:
        LiDAR interface instance
    """
    lidar_type = lidar_type.lower()

    if lidar_type == "pandar" or lidar_type == "pandar_40p":
        return PandarLiDAR()
    elif lidar_type == "mock" or lidar_type == "test":
        return MockLiDAR()
    else:
        logger.warning(f"Unknown LiDAR type: {lidar_type}, using Pandar 40P")
        return PandarLiDAR()


if __name__ == "__main__":
    """Test LiDAR interface"""
    print("="*60)
    print("LiDAR Interface Test")
    print("="*60)

    # Test Pandar 40P
    print("\n1. Testing Pandar 40P...")
    pandar = PandarLiDAR()
    print(f"   Name: {pandar.name}")
    print(f"   Specs: {pandar.specs}")
    print(f"   Status: {pandar.get_status().value}")

    connected = pandar.connect(ip="192.168.1.201")
    print(f"   Connection attempt: {connected}")
    print(f"   Status after connection: {pandar.get_status().value}")

    # Test Mock LiDAR
    print("\n2. Testing Mock LiDAR...")
    mock = MockLiDAR()
    print(f"   Name: {mock.name}")
    print(f"   Status: {mock.get_status().value}")

    connected = mock.connect()
    print(f"   Connected: {connected}")
    print(f"   Status: {mock.get_status().value}")

    if connected:
        pc = mock.read_point_cloud()
        if pc:
            print(f"   Point cloud: {pc.num_points} points")

    mock.disconnect()
    print(f"   Status after disconnect: {mock.get_status().value}")

    print("\n" + "="*60)
    print("Test complete!")
