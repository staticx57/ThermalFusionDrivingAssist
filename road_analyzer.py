"""
Road Object Analysis and Alert System
Analyzes detected objects and generates driver alerts
"""
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import time
import logging

from object_detector import Detection

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

    def __init__(self, frame_width: int = 640, frame_height: int = 512):
        """
        Initialize road analyzer

        Args:
            frame_width: Camera frame width
            frame_height: Camera frame height
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
        """Evaluate a single detection for alerts"""

        # Check alert cooldown
        cooldown_key = f"{det.class_name}_{det.get_center()}"
        if cooldown_key in self.last_alert_time:
            if current_time - self.last_alert_time[cooldown_key] < self.alert_cooldown:
                return None

        # Calculate object size relative to frame
        obj_area = det.get_area()
        size_ratio = obj_area / self.frame_area

        # Determine position (left, center, right)
        center_x, center_y = det.get_center()
        position = self._get_position(center_x)

        # Critical alert check
        critical_threshold = self.CRITICAL_SIZE_THRESHOLDS.get(det.class_name, 0.25)
        if size_ratio >= critical_threshold:
            self.last_alert_time[cooldown_key] = current_time
            return Alert(
                level=AlertLevel.CRITICAL,
                message=f"CRITICAL: {det.class_name.upper()} VERY CLOSE ahead!",
                object_type=det.class_name,
                timestamp=current_time,
                position=position
            )

        # Warning alert check
        warning_threshold = self.WARNING_SIZE_THRESHOLDS.get(det.class_name, 0.12)
        if size_ratio >= warning_threshold:
            self.last_alert_time[cooldown_key] = current_time
            return Alert(
                level=AlertLevel.WARNING,
                message=f"Warning: {det.class_name} approaching on {position}",
                object_type=det.class_name,
                timestamp=current_time,
                position=position
            )

        # Info alert for certain object types
        if det.class_name in ['traffic light', 'stop sign']:
            return Alert(
                level=AlertLevel.INFO,
                message=f"{det.class_name} detected ahead",
                object_type=det.class_name,
                timestamp=current_time,
                position=position
            )

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