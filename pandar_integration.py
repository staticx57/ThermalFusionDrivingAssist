"""
Hesai Pandar 40P LiDAR Integration for Object Detection
Focused on distance measurement enhancement for thermal/RGB fusion

CRITICAL BUG FIXES (from real-world testing):
1. MIXED ENDIANNESS BUG:
   - Flag bytes (0xFF 0xEE) appear as big-endian markers
   - BUT all data fields (distance, azimuth) are LITTLE-ENDIAN
   - Must check flag bytes as RAW BYTES, not parse with struct
   - Example: Raw bytes 03 C5 = 965 (3.86m) little-endian, 50,436 (201.74m) big-endian WRONG!

2. DISTANCE RESOLUTION:
   - Spec says 2mm, but Hesai SDK (pandarGeneral_internal.cc) uses 4mm
   - Verified from user's real-world testing: 4mm is correct

3. FAR-RANGE NOISE FILTERING:
   - Points >100m with intensity <10 are typically noise
   - Must filter these out for reliable object detection

4. AZIMUTH WRAPAROUND:
   - Raw values can be -4.8° to 362.6°
   - Must normalize to 0-360° range

Specifications:
- 40-channel mechanical LiDAR
- UDP port 2368 (default)
- Source IP: 192.168.1.201 (default)
- Little-endian format for all multi-byte fields (except flag bytes!)
- 10 blocks per packet, 40 channels per block
- Distance resolution: 4mm (0.004m per unit)
- Accuracy: ±2cm
- Range: 0.3m - 200m

Reference: Hesai SDK pandarGeneral_internal.cc lines 1017-1023
"""

import socket
import struct
import threading
import time
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PandarPoint:
    """Single LiDAR point"""
    distance_m: float
    azimuth_deg: float
    elevation_deg: float
    intensity: int  # 0-255 (raw intensity, NOT reflectivity)
    channel_id: int


@dataclass
class RegionDistance:
    """Distance measurement for a region"""
    min_distance: float
    avg_distance: float
    max_distance: float
    point_count: int
    confidence: float  # 0.0-1.0


class PandarIntegration:
    """
    Hesai Pandar 40P LiDAR integration for object detection

    Phase 1: Distance override (use LiDAR distance when available)
    """

    # Pandar 40P vertical angles (degrees) for 40 channels
    # From bottom to top: -16° to +7°
    VERTICAL_ANGLES = [
        -16.0, -15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0,
        -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0,
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
        -16.0, -15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0,
        -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0
    ]

    # Pandar 40P FOV
    HORIZONTAL_FOV = 360.0  # degrees
    VERTICAL_FOV_MIN = -16.0  # degrees
    VERTICAL_FOV_MAX = 7.0  # degrees

    # Distance resolution: 4mm per unit (from Hesai SDK, pandarGeneral_internal.cc)
    # CRITICAL: User's testing shows 4mm, not 2mm!
    DISTANCE_UNIT = 0.004  # meters

    # Packet structure
    BLOCK_FLAG = 0xFFEE
    BLOCK_SIZE = 124  # bytes
    BLOCKS_PER_PACKET = 10
    CHANNELS_PER_BLOCK = 40
    CHANNEL_DATA_SIZE = 3  # bytes (2 distance + 1 reflectivity)

    def __init__(self, udp_port: int = 2368, buffer_size: int = 2000):
        """
        Initialize Pandar 40P integration

        Args:
            udp_port: UDP port for LiDAR data (default: 2368)
            buffer_size: Maximum points to keep in buffer
        """
        self.udp_port = udp_port
        self.buffer_size = buffer_size

        # Connection state
        self.socket = None
        self.running = False
        self.connected = False

        # Point cloud buffer (latest points)
        self.point_buffer = []
        self.buffer_lock = threading.Lock()

        # Reception thread
        self.receive_thread = None

        # Statistics
        self.packets_received = 0
        self.points_received = 0
        self.last_packet_time = 0

    def connect(self) -> bool:
        """
        Connect to Pandar 40P LiDAR

        Returns:
            True if connected successfully
        """
        try:
            # Create UDP socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            # Bind to port (listen for broadcast or directed packets)
            self.socket.bind(('', self.udp_port))

            # Set non-blocking mode with timeout
            self.socket.settimeout(1.0)

            self.connected = True
            logger.info(f"[OK] Pandar 40P connected on UDP port {self.udp_port}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Pandar 40P: {e}")
            self.connected = False
            return False

    def start(self):
        """Start receiving LiDAR data in background thread"""
        if not self.connected:
            logger.warning("Cannot start - not connected to LiDAR")
            return

        self.running = True
        self.receive_thread = threading.Thread(target=self._receive_worker, daemon=True)
        self.receive_thread.start()
        logger.info("Pandar 40P reception started")

    def stop(self):
        """Stop receiving LiDAR data"""
        self.running = False
        if self.receive_thread:
            self.receive_thread.join(timeout=2.0)
        logger.info("Pandar 40P reception stopped")

    def disconnect(self):
        """Disconnect from LiDAR"""
        self.stop()
        if self.socket:
            self.socket.close()
            self.socket = None
        self.connected = False
        logger.info("Pandar 40P disconnected")

    def _receive_worker(self):
        """Background thread to receive and parse UDP packets"""
        logger.info("Pandar 40P receiver thread started")

        while self.running:
            try:
                # Receive UDP packet
                data, addr = self.socket.recvfrom(2048)

                # Parse packet
                points = self._parse_packet(data)

                if points:
                    # Update buffer (keep latest points)
                    with self.buffer_lock:
                        self.point_buffer.extend(points)
                        if len(self.point_buffer) > self.buffer_size:
                            self.point_buffer = self.point_buffer[-self.buffer_size:]

                    self.packets_received += 1
                    self.points_received += len(points)
                    self.last_packet_time = time.time()

            except socket.timeout:
                # Normal timeout, continue
                continue
            except Exception as e:
                if self.running:
                    logger.error(f"Pandar packet reception error: {e}")
                    time.sleep(0.1)

        logger.info("Pandar 40P receiver thread stopped")

    def _parse_packet(self, data: bytes) -> List[PandarPoint]:
        """
        Parse Pandar 40P UDP packet

        CRITICAL ENDIANNESS BUG FIX (from user's real-world testing):
        - Flag bytes (0xFF 0xEE) appear as big-endian markers
        - BUT all data fields (distance, azimuth) are LITTLE-ENDIAN
        - DON'T parse flag bytes with struct - check raw bytes!
        - Example bug: Raw bytes 03 C5 = 965 (3.86m) when little-endian,
          but 50,436 (201.74m) when big-endian - WRONG!

        Packet structure (from Hesai SDK pandarGeneral_internal.cc):
        - 10 blocks × 124 bytes each = 1240 bytes
        Each block:
        - Block flag: 2 bytes (0xFF 0xEE raw bytes - NOT parsed!)
        - Azimuth: 2 bytes (little-endian uint16, × 0.01° units)
        - 40 channels × 3 bytes:
          - Distance: 2 bytes (little-endian uint16, × 0.004m = 4mm units)
          - Intensity: 1 byte (0-255)

        Args:
            data: Raw UDP packet data

        Returns:
            List of PandarPoint objects
        """
        points = []

        try:
            # Minimum expected size: 10 blocks × 124 bytes = 1240 bytes
            if len(data) < 1240:
                return points

            offset = 0

            # Parse 10 blocks
            for block_idx in range(self.BLOCKS_PER_PACKET):
                # CRITICAL: Check block flag as RAW BYTES (don't parse with endianness!)
                # Flag is 0xFF 0xEE in that exact order
                if data[offset] != 0xFF or data[offset + 1] != 0xEE:
                    logger.warning(f"Invalid block flag: 0x{data[offset]:02X}{data[offset+1]:02X} "
                                 f"(expected 0xFFEE)")
                    break
                offset += 2

                # Parse azimuth (LITTLE-ENDIAN uint16, 0.01° units)
                azimuth_raw = struct.unpack_from('<H', data, offset)[0]
                azimuth_deg = azimuth_raw * 0.01

                # Normalize azimuth to 0-360° range (handle wraparound)
                if azimuth_deg < 0:
                    azimuth_deg += 360.0
                elif azimuth_deg >= 360.0:
                    azimuth_deg -= 360.0

                offset += 2

                # Parse 40 channels
                for channel_idx in range(self.CHANNELS_PER_BLOCK):
                    # Distance: 2 bytes (LITTLE-ENDIAN uint16, 4mm units)
                    distance_raw = struct.unpack_from('<H', data, offset)[0]
                    offset += 2

                    # Intensity: 1 byte (NOT reflectivity - raw intensity value)
                    intensity = struct.unpack_from('B', data, offset)[0]
                    offset += 1

                    # Convert distance to meters (4mm resolution from Hesai SDK)
                    distance_m = distance_raw * self.DISTANCE_UNIT

                    # Filter invalid points (from user's real-world testing):
                    # 1. Distance = 0 (no return)
                    # 2. Distance < 0.1m (too close, likely noise)
                    # 3. Distance > 100m with intensity < 10 (far-range noise)
                    if distance_m < 0.1:
                        continue
                    if distance_m > 100.0 and intensity < 10:
                        continue  # Far-range noise
                    if distance_m > 200.0:
                        continue  # Beyond sensor range

                    # Get elevation angle for this channel
                    elevation_deg = self.VERTICAL_ANGLES[channel_idx]

                    # Create point
                    point = PandarPoint(
                        distance_m=distance_m,
                        azimuth_deg=azimuth_deg,
                        elevation_deg=elevation_deg,
                        intensity=intensity,
                        channel_id=channel_idx
                    )

                    points.append(point)

        except Exception as e:
            logger.error(f"Packet parsing error: {e}")

        return points

    def get_region_distance(self, azimuth_center: float, azimuth_width: float,
                           elevation_center: float = 0.0, elevation_width: float = 10.0,
                           min_intensity: int = 20) -> Optional[RegionDistance]:
        """
        Get distance measurement for angular region (matches camera detection ROI)

        Args:
            azimuth_center: Center azimuth angle in degrees (0-360)
            azimuth_width: Angular width in degrees
            elevation_center: Center elevation angle in degrees (-16 to +7)
            elevation_width: Elevation angular width in degrees
            min_intensity: Minimum intensity threshold (0-255) to filter noise

        Returns:
            RegionDistance object or None if no points found
        """
        # Define angular bounds
        azimuth_min = azimuth_center - azimuth_width / 2
        azimuth_max = azimuth_center + azimuth_width / 2
        elevation_min = elevation_center - elevation_width / 2
        elevation_max = elevation_center + elevation_width / 2

        # Handle azimuth wraparound (0-360)
        if azimuth_min < 0:
            azimuth_min += 360
        if azimuth_max > 360:
            azimuth_max -= 360

        # Query points in region
        region_points = []

        with self.buffer_lock:
            for point in self.point_buffer:
                # Check intensity threshold (filter low-intensity noise)
                if point.intensity < min_intensity:
                    continue

                # Check elevation bounds
                if not (elevation_min <= point.elevation_deg <= elevation_max):
                    continue

                # Check azimuth bounds (handle wraparound)
                if azimuth_min <= azimuth_max:
                    if azimuth_min <= point.azimuth_deg <= azimuth_max:
                        region_points.append(point.distance_m)
                else:
                    # Wraparound case
                    if point.azimuth_deg >= azimuth_min or point.azimuth_deg <= azimuth_max:
                        region_points.append(point.distance_m)

        if not region_points:
            return None

        # Calculate statistics
        distances = np.array(region_points)
        min_dist = float(np.min(distances))
        avg_dist = float(np.mean(distances))
        max_dist = float(np.max(distances))
        count = len(region_points)

        # Confidence based on point count (more points = higher confidence)
        confidence = min(count / 10.0, 1.0)  # 10+ points = full confidence

        return RegionDistance(
            min_distance=min_dist,
            avg_distance=avg_dist,
            max_distance=max_dist,
            point_count=count,
            confidence=confidence
        )

    def get_distance_for_bbox(self, bbox: Tuple[int, int, int, int],
                              image_width: int, image_height: int,
                              camera_fov_h: float = 60.0,
                              camera_fov_v: float = 45.0,
                              camera_pitch: float = 0.0) -> Optional[float]:
        """
        Get LiDAR distance for camera detection bounding box

        Args:
            bbox: Bounding box (x1, y1, x2, y2) in pixels
            image_width: Image width in pixels
            image_height: Image height in pixels
            camera_fov_h: Camera horizontal FOV in degrees
            camera_fov_v: Camera vertical FOV in degrees
            camera_pitch: Camera pitch angle in degrees (positive = up)

        Returns:
            Distance in meters or None if no LiDAR data in ROI
        """
        x1, y1, x2, y2 = bbox

        # Calculate bbox center and size in pixels
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width_px = x2 - x1
        height_px = y2 - y1

        # Convert pixel coordinates to angular coordinates
        # Azimuth: horizontal angle (assume camera faces forward, azimuth = 0)
        # Pixel offset from center → angle offset
        pixel_offset_x = center_x - (image_width / 2)
        azimuth_offset = (pixel_offset_x / image_width) * camera_fov_h
        azimuth_center = azimuth_offset  # Relative to camera forward direction

        # Elevation: vertical angle
        pixel_offset_y = (image_height / 2) - center_y  # Flip Y (image: top=0, world: up=positive)
        elevation_offset = (pixel_offset_y / image_height) * camera_fov_v
        elevation_center = elevation_offset + camera_pitch

        # Calculate angular widths
        azimuth_width = (width_px / image_width) * camera_fov_h
        elevation_width = (height_px / image_height) * camera_fov_v

        # Add margin for better coverage (20% margin)
        azimuth_width *= 1.2
        elevation_width *= 1.2

        # Query LiDAR region
        region = self.get_region_distance(
            azimuth_center=azimuth_center,
            azimuth_width=azimuth_width,
            elevation_center=elevation_center,
            elevation_width=elevation_width,
            min_intensity=20  # Filter low-intensity noise
        )

        if region and region.point_count >= 3:
            # Use minimum distance (closest point in ROI = object surface)
            return region.min_distance

        return None

    def get_stats(self) -> dict:
        """Get LiDAR statistics"""
        with self.buffer_lock:
            buffer_points = len(self.point_buffer)

        time_since_last = time.time() - self.last_packet_time if self.last_packet_time > 0 else 0

        return {
            'connected': self.connected,
            'running': self.running,
            'packets_received': self.packets_received,
            'points_received': self.points_received,
            'buffer_points': buffer_points,
            'time_since_last_packet': time_since_last,
            'receiving': time_since_last < 2.0  # Active if packet within 2 seconds
        }


# Simple test function
if __name__ == "__main__":
    print("Pandar 40P Integration Test")
    print("=" * 50)

    # Create integration
    pandar = PandarIntegration(udp_port=2368)

    # Try to connect
    if pandar.connect():
        print("✓ Connected to Pandar 40P")

        # Start receiving
        pandar.start()
        print("✓ Started reception")

        # Run for 10 seconds
        print("\nReceiving data for 10 seconds...")
        for i in range(10):
            time.sleep(1)
            stats = pandar.get_stats()
            print(f"  [{i+1}s] Packets: {stats['packets_received']}, "
                  f"Points: {stats['points_received']}, "
                  f"Buffer: {stats['buffer_points']}, "
                  f"Active: {stats['receiving']}")

        # Test region query
        print("\nTesting region query (forward, center)...")
        region = pandar.get_region_distance(
            azimuth_center=0.0,
            azimuth_width=10.0,
            elevation_center=0.0,
            elevation_width=5.0
        )

        if region:
            print(f"  Min distance: {region.min_distance:.2f}m")
            print(f"  Avg distance: {region.avg_distance:.2f}m")
            print(f"  Max distance: {region.max_distance:.2f}m")
            print(f"  Point count: {region.point_count}")
            print(f"  Confidence: {region.confidence:.2f}")
        else:
            print("  No points in region")

        # Cleanup
        pandar.disconnect()
        print("\n✓ Disconnected")
    else:
        print("✗ Failed to connect")
