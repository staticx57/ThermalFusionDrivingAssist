"""
Road Object Analysis and Alert System
Analyzes detected objects and generates driver alerts

New Features (v3.0):
- Distance estimation for all detected objects
- Time-to-collision (TTC) calculation
- Audio alert integration
- Enhanced proximity warnings with distance zones
"""
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import time
import logging

from object_detector import Detection

# Optional imports (graceful degradation if not available)
try:
    from distance_estimator import DistanceEstimator, DistanceEstimate
    DISTANCE_AVAILABLE = True
except ImportError:
    DISTANCE_AVAILABLE = False
    logger.warning("distance_estimator not available - distance estimation disabled")

try:
    from audio_alert_system import AudioAlertSystem
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    logger.warning("audio_alert_system not available - audio alerts disabled")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels"""
    NONE = 0
    INFO = 1
    WARNING = 2
    CRITICAL = 3


@dataclass
class Alert:
    """Alert for driver"""
    level: AlertLevel
    message: str
    object_type: str
    timestamp: float
    position: str  # "left", "center", "right"
    distance_m: Optional[float] = None  # Distance in meters (NEW)
    ttc: Optional[float] = None  # Time-to-collision in seconds (NEW)
    distance_zone: Optional[str] = None  # "IMMEDIATE", "VERY CLOSE", "CLOSE", etc. (NEW)


class RoadAnalyzer:
    """Analyzes road objects and generates alerts"""

    # Alert thresholds based on object size (as percentage of frame)
    CRITICAL_SIZE_THRESHOLDS = {
        'person': 0.15,      # Person taking up 15% of frame = very close
        'car': 0.30,
        'truck': 0.35,
        'bus': 0.35,
        'bicycle': 0.10,
        'motorcycle': 0.12,
        'dog': 0.08,
        'cat': 0.05,
        'motion': 0.12,      # Unknown motion (deer, animals, etc.)
    }

    WARNING_SIZE_THRESHOLDS = {
        'person': 0.08,
        'car': 0.15,
        'truck': 0.18,
        'bus': 0.18,
        'bicycle': 0.05,
        'motorcycle': 0.06,
        'dog': 0.04,
        'cat': 0.03,
        'motion': 0.05,      # Unknown motion
    }

    def __init__(self, frame_width: int = 640, frame_height: int = 512,
                 enable_distance: bool = True,
                 enable_audio: bool = True,
                 thermal_mode: bool = False):
        """
        Initialize road analyzer

        Args:
            frame_width: Camera frame width
            frame_height: Camera frame height
            enable_distance: Enable distance estimation
            enable_audio: Enable audio alerts
            thermal_mode: True if using thermal camera (affects distance estimation)
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_area = frame_width * frame_height
        self.alerts = []
        self.alert_history = []
        self.detection_counts = {}
        self.last_alert_time = {}

        # Alert cooldown to prevent spam (seconds)
        self.alert_cooldown = 1.0

        # Distance estimator (NEW)
        self.distance_estimator = None
        if enable_distance and DISTANCE_AVAILABLE:
            self.distance_estimator = DistanceEstimator(
                camera_focal_length=640.0,
                camera_height=1.2,
                frame_height=frame_height,
                thermal_mode=thermal_mode
            )
            logger.info("Distance estimation enabled")

        # Audio alert system (NEW)
        self.audio_system = None
        if enable_audio and AUDIO_AVAILABLE:
            self.audio_system = AudioAlertSystem()
            logger.info("Audio alerts enabled")

    def analyze(self, detections: List[Detection]) -> List[Alert]:
        """
        Analyze detections and generate alerts

        Args:
            detections: List of object detections

        Returns:
            List of alerts for driver
        """
        current_time = time.time()
        self.alerts = []

        # Update detection counts
        self._update_counts(detections)

        # Analyze each detection
        for det in detections:
            alert = self._evaluate_detection(det, current_time)
            if alert:
                self.alerts.append(alert)
                self.alert_history.append(alert)

        # Keep only recent alert history (last 100)
        if len(self.alert_history) > 100:
            self.alert_history = self.alert_history[-100:]

        return self.alerts

    def _evaluate_detection(self, det: Detection, current_time: float) -> Optional[Alert]:
        """Evaluate a single detection for alerts (ENHANCED with distance estimation)"""

        # Check alert cooldown
        cooldown_key = f"{det.class_name}_{det.get_center()}"
        if cooldown_key in self.last_alert_time:
            if current_time - self.last_alert_time[cooldown_key] < self.alert_cooldown:
                return None

        # Estimate distance (NEW)
        distance_estimate = None
        distance_m = None
        ttc = None
        distance_zone = None

        if self.distance_estimator:
            distance_estimate = self.distance_estimator.estimate_distance(det)
            if distance_estimate:
                distance_m = distance_estimate.distance_m
                ttc = distance_estimate.time_to_collision
                distance_zone, _ = self.distance_estimator.get_distance_zones(distance_m)

        # Calculate object size relative to frame (legacy method)
        obj_area = det.get_area()
        size_ratio = obj_area / self.frame_area

        # Determine position (left, center, right)
        center_x, center_y = det.get_center()
        position = self._get_position(center_x)

        # Alert level determination (distance-based if available, else size-based)
        alert_level = None
        message = ""

        if distance_m is not None:
            # Distance-based alerting (more accurate)
            if distance_m < 5.0:  # IMMEDIATE zone
                alert_level = AlertLevel.CRITICAL
                message = f"CRITICAL: {det.class_name.upper()} {distance_m:.1f}m ahead!"
            elif distance_m < 10.0:  # VERY CLOSE zone
                alert_level = AlertLevel.CRITICAL
                message = f"DANGER: {det.class_name} {distance_m:.1f}m on {position}!"
            elif distance_m < 20.0:  # CLOSE zone
                alert_level = AlertLevel.WARNING
                message = f"Warning: {det.class_name} {distance_m:.1f}m on {position}"
            elif distance_m < 40.0:  # MEDIUM zone
                if det.class_name in ['person', 'bicycle', 'motorcycle']:
                    alert_level = AlertLevel.INFO
                    message = f"{det.class_name} {distance_m:.0f}m ahead"

            # TTC-based urgent warning
            if ttc is not None and ttc < 3.0:
                alert_level = AlertLevel.CRITICAL
                message = f"COLLISION WARNING: {det.class_name} - TTC {ttc:.1f}s!"

        else:
            # Fallback to size-based alerting (legacy)
            critical_threshold = self.CRITICAL_SIZE_THRESHOLDS.get(det.class_name, 0.25)
            if size_ratio >= critical_threshold:
                alert_level = AlertLevel.CRITICAL
                message = f"CRITICAL: {det.class_name.upper()} VERY CLOSE ahead!"
            else:
                warning_threshold = self.WARNING_SIZE_THRESHOLDS.get(det.class_name, 0.12)
                if size_ratio >= warning_threshold:
                    alert_level = AlertLevel.WARNING
                    message = f"Warning: {det.class_name} approaching on {position}"

            # Info alert for certain object types
            if det.class_name in ['traffic light', 'stop sign']:
                alert_level = AlertLevel.INFO
                message = f"{det.class_name} detected ahead"

        # Create alert if needed
        if alert_level:
            self.last_alert_time[cooldown_key] = current_time

            alert = Alert(
                level=alert_level,
                message=message,
                object_type=det.class_name,
                timestamp=current_time,
                position=position,
                distance_m=distance_m,
                ttc=ttc,
                distance_zone=distance_zone
            )

            # Play audio alert (NEW)
            if self.audio_system and self.audio_system.is_enabled():
                if ttc is not None and ttc < 3.0:
                    # Urgent collision warning
                    self.audio_system.play_collision_warning(ttc, position)
                else:
                    # Standard alert
                    self.audio_system.play_alert(alert_level, position, det.class_name)

            return alert

        return None

    def _get_position(self, center_x: int) -> str:
        """Determine if object is on left, center, or right of frame"""
        third = self.frame_width // 3
        if center_x < third:
            return "left"
        elif center_x < 2 * third:
            return "center"
        else:
            return "right"

    def _update_counts(self, detections: List[Detection]):
        """Update detection counts by object type"""
        self.detection_counts = {}
        for det in detections:
            if det.class_name not in self.detection_counts:
                self.detection_counts[det.class_name] = 0
            self.detection_counts[det.class_name] += 1

    def get_statistics(self) -> Dict:
        """Get current statistics"""
        return {
            'current_detections': self.detection_counts,
            'active_alerts': len(self.alerts),
            'critical_alerts': len([a for a in self.alerts if a.level == AlertLevel.CRITICAL]),
            'warning_alerts': len([a for a in self.alerts if a.level == AlertLevel.WARNING]),
            'total_alerts_history': len(self.alert_history)
        }

    def get_alerts_by_level(self, level: AlertLevel) -> List[Alert]:
        """Get all current alerts of specified level"""
        return [a for a in self.alerts if a.level == level]

    def clear_old_alerts(self, max_age: float = 5.0):
        """Clear alerts older than max_age seconds"""
        current_time = time.time()
        self.alerts = [a for a in self.alerts if current_time - a.timestamp < max_age]

    def update_vehicle_speed(self, speed_kmh: float):
        """
        Update vehicle speed for TTC calculation (NEW)

        Args:
            speed_kmh: Vehicle speed in km/h
        """
        if self.distance_estimator:
            self.distance_estimator.update_vehicle_speed(speed_kmh)

    def set_audio_enabled(self, enabled: bool):
        """Enable/disable audio alerts (NEW)"""
        if self.audio_system:
            if enabled:
                self.audio_system.enable()
            else:
                self.audio_system.disable()

    def set_audio_volume(self, volume: float):
        """Set audio volume 0.0-1.0 (NEW)"""
        if self.audio_system:
            self.audio_system.set_volume(volume)

    def cleanup(self):
        """Cleanup resources (NEW)"""
        if self.audio_system:
            self.audio_system.cleanup()