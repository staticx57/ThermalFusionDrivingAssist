"""
Road Analyzer - Alert System
Minimal implementation for alert handling in driver GUI
"""
from enum import Enum
from typing import Optional


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    DANGER = "danger"
    CRITICAL = "critical"


class Alert:
    """
    Alert object for road safety warnings
    """
    def __init__(self, 
                 message: str,
                 level: AlertLevel = AlertLevel.WARNING,
                 object_class: Optional[str] = None,
                 distance: Optional[float] = None,
                 position: Optional[str] = None):
        """
        Initialize alert
        
        Args:
            message: Alert message text
            level: Alert severity level
            object_class: Detected object class (e.g., 'person', 'vehicle')
            distance: Distance to object in meters
            position: Position relative to vehicle ('left', 'right', 'center')
        """
        self.message = message
        self.level = level
        self.object_class = object_class
        self.distance = distance
        self.position = position
        self.timestamp = None  # Can be set by caller
    
    def __str__(self):
        return f"Alert({self.level.value}): {self.message}"
    
    def __repr__(self):
        return self.__str__()
