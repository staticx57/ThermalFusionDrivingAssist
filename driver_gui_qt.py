#!/usr/bin/env python3
"""
Qt-based GUI for Thermal Fusion Driving Assist
Professional interface with proper theming, font rendering, and performance
Optimized for Jetson Orin and x86-64 platforms
"""
import sys
import numpy as np
from typing import List, Optional
import logging

try:
    from PyQt5.QtWidgets import (QMainWindow, QWidget, QLabel, QPushButton,
                                  QVBoxLayout, QHBoxLayout, QApplication,
                                  QSizePolicy)
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QSize
    from PyQt5.QtGui import QImage, QPixmap, QFont
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    print("WARNING: PyQt5 not available. Install with: pip3 install PyQt5")

import cv2
from detection import Detection
from alerts import Alert
from view_mode import ViewMode

logger = logging.getLogger(__name__)


# ============================================================================
# Qt Styling (QSS - CSS-like)
# ============================================================================

DARK_THEME = """
QMainWindow {
    background-color: #1a1a1a;
}

QPushButton {
    background-color: #2d2d2d;
    color: #e0e0e0;
    border: 2px solid #404040;
    border-radius: 8px;
    padding: 10px 20px;
    font-size: 14px;
    font-weight: bold;
    min-width: 120px;
}

QPushButton:hover {
    background-color: #3d3d3d;
    border: 2px solid #00aaff;
}

QPushButton:pressed {
    background-color: #1d1d1d;
}

QPushButton:checked {
    background-color: #00aaff;
    color: #ffffff;
    border: 2px solid #00ddff;
}

QPushButton:disabled {
    background-color: #1a1a1a;
    color: #666666;
    border: 2px solid #2d2d2d;
}

QLabel#info_panel {
    background-color: rgba(0, 0, 0, 180);
    color: #00ff00;
    border: 1px solid #404040;
    border-radius: 5px;
    padding: 10px;
    font-size: 12px;
    font-family: monospace;
}
"""

LIGHT_THEME = """
QMainWindow {
    background-color: #f5f5f5;
}

QPushButton {
    background-color: #ffffff;
    color: #202020;
    border: 2px solid #c0c0c0;
    border-radius: 8px;
    padding: 10px 20px;
    font-size: 14px;
    font-weight: bold;
    min-width: 120px;
}

QPushButton:hover {
    background-color: #f0f0f0;
    border: 2px solid #0066cc;
}

QPushButton:pressed {
    background-color: #e0e0e0;
}

QPushButton:checked {
    background-color: #0066cc;
    color: #ffffff;
    border: 2px solid #0088ff;
}

QPushButton:disabled {
    background-color: #f5f5f5;
    color: #999999;
    border: 2px solid #d0d0d0;
}

QLabel#info_panel {
    background-color: rgba(255, 255, 255, 220);
    color: #006600;
    border: 1px solid #c0c0c0;
    border-radius: 5px;
    padding: 10px;
    font-size: 12px;
    font-family: monospace;
}
"""


# ============================================================================
# Video Display Widget
# ============================================================================

class VideoWidget(QLabel):
    """
    High-performance video display widget
    Optimized for Jetson with frame buffer reuse
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setScaledContents(False)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #000000;")

        # Performance optimization: Reuse buffers
        self._frame_buffer = None
        self._qimage_buffer = None

        # Placeholder
        self._show_placeholder()

    def _show_placeholder(self):
        """Show placeholder when no video available"""
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Waiting for video...", (150, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
        self.update_frame(placeholder)

    def update_frame(self, frame: np.ndarray):
        """
        Update displayed frame efficiently

        Args:
            frame: BGR numpy array (from OpenCV)
        """
        if frame is None or frame.size == 0:
            return

        try:
            # Convert BGR to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape

            # Create QImage from numpy array
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # Convert to pixmap and scale to fit widget
            pixmap = QPixmap.fromImage(qimg)
            scaled_pixmap = pixmap.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation  # High quality scaling
            )

            self.setPixmap(scaled_pixmap)

        except Exception as e:
            logger.error(f"Error updating video frame: {e}")

    def sizeHint(self):
        """Suggest reasonable default size"""
        return QSize(1280, 1024)


# ============================================================================
# Control Panel Widget
# ============================================================================

class ControlPanel(QWidget):
    """
    Control buttons for driving mode
    Emits signals for button clicks
    """

    # Signals for button actions
    view_mode_clicked = pyqtSignal()
    yolo_toggled = pyqtSignal(bool)
    audio_toggled = pyqtSignal(bool)
    info_toggled = pyqtSignal(bool)
    theme_clicked = pyqtSignal()
    retry_sensors_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        # Create buttons
        self.view_btn = QPushButton("🎥 View: Thermal")
        self.yolo_btn = QPushButton("🎯 YOLO: OFF")
        self.audio_btn = QPushButton("🔊 Audio: ON")
        self.info_btn = QPushButton("ℹ️ Info")
        self.theme_btn = QPushButton("🎨 Theme: Auto")
        self.retry_btn = QPushButton("🔄 Retry Sensors")

        # Make toggle buttons checkable
        self.yolo_btn.setCheckable(True)
        self.audio_btn.setCheckable(True)
        self.info_btn.setCheckable(True)

        # Set initial checked states
        self.audio_btn.setChecked(True)  # Audio on by default

        # Connect signals
        self.view_btn.clicked.connect(self.view_mode_clicked.emit)
        self.yolo_btn.toggled.connect(self._on_yolo_toggled)
        self.audio_btn.toggled.connect(self._on_audio_toggled)
        self.info_btn.toggled.connect(self._on_info_toggled)
        self.theme_btn.clicked.connect(self.theme_clicked.emit)
        self.retry_btn.clicked.connect(self.retry_sensors_clicked.emit)

        # Layout
        layout = QHBoxLayout()
        layout.addWidget(self.view_btn)
        layout.addWidget(self.yolo_btn)
        layout.addWidget(self.audio_btn)
        layout.addWidget(self.info_btn)
        layout.addStretch()  # Push theme/retry to right
        layout.addWidget(self.theme_btn)
        layout.addWidget(self.retry_btn)

        self.setLayout(layout)

    def _on_yolo_toggled(self, checked: bool):
        """Update button text when toggled"""
        self.yolo_btn.setText(f"🎯 YOLO: {'ON' if checked else 'OFF'}")
        self.yolo_toggled.emit(checked)

    def _on_audio_toggled(self, checked: bool):
        """Update button text when toggled"""
        self.audio_btn.setText(f"{'🔊' if checked else '🔇'} Audio: {'ON' if checked else 'OFF'}")
        self.audio_toggled.emit(checked)

    def _on_info_toggled(self, checked: bool):
        """Update button text when toggled"""
        self.info_btn.setText(f"ℹ️ Info{'✓' if checked else ''}")
        self.info_toggled.emit(checked)

    def update_view_mode(self, mode: ViewMode):
        """Update view mode button text"""
        mode_text = {
            ViewMode.THERMAL_ONLY: "Thermal",
            ViewMode.RGB_ONLY: "RGB",
            ViewMode.FUSION: "Fusion",
            ViewMode.SIDE_BY_SIDE: "Side-by-Side",
            ViewMode.PICTURE_IN_PICTURE: "PiP"
        }.get(mode, "Unknown")
        self.view_btn.setText(f"🎥 View: {mode_text}")

    def update_theme_mode(self, theme: str):
        """Update theme button text"""
        self.theme_btn.setText(f"🎨 Theme: {theme.title()}")

    def set_yolo_enabled(self, enabled: bool):
        """Set YOLO button state"""
        self.yolo_btn.setChecked(enabled)

    def set_audio_enabled(self, enabled: bool):
        """Set audio button state"""
        self.audio_btn.setChecked(enabled)


# ============================================================================
# Info Panel Overlay Widget
# ============================================================================

class InfoPanel(QLabel):
    """
    Semi-transparent overlay showing metrics
    Positioned over video widget
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("info_panel")
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.setWordWrap(True)

        # Set minimum size
        self.setMinimumSize(250, 150)

        # Position in top-right corner (will be adjusted in parent resize)
        self.move(parent.width() - 270 if parent else 0, 10)

        # Initially hidden
        self.hide()

    def update_info(self, fps: float, detections: int,
                    thermal_connected: bool, rgb_connected: bool,
                    lidar_connected: bool = False):
        """Update information display"""
        info_text = f"""
<b>Performance</b><br>
FPS: {fps:.1f}<br>
Detections: {detections}<br>
<br>
<b>Sensors</b><br>
Thermal: {'✓ Connected' if thermal_connected else '✗ Disconnected'}<br>
RGB: {'✓ Connected' if rgb_connected else '✗ Disconnected'}<br>
LiDAR: {'✓ Connected' if lidar_connected else '✗ Not Available'}<br>
"""
        self.setText(info_text.strip())


# ============================================================================
# Main Application Window
# ============================================================================

class DriverAppWindow(QMainWindow):
    """
    Main Qt window for Thermal Fusion Driving Assist
    Handles video display, controls, and application state
    """

    def __init__(self):
        super().__init__()

        if not PYQT_AVAILABLE:
            raise ImportError("PyQt5 is required. Install with: pip3 install PyQt5")

        self.setWindowTitle("Thermal Fusion Driving Assist - Qt Edition")

        # Application state
        self.current_theme = 'dark'
        self.view_mode = ViewMode.THERMAL_ONLY
        self.show_info_panel = False

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Video display (90% of space)
        self.video_widget = VideoWidget()
        main_layout.addWidget(self.video_widget, stretch=9)

        # Control panel (10% of space)
        self.control_panel = ControlPanel()
        main_layout.addWidget(self.control_panel, stretch=1)

        central_widget.setLayout(main_layout)

        # Info panel overlay (on top of video widget)
        self.info_panel = InfoPanel(self.video_widget)

        # Connect control panel signals (will be connected by main.py)
        # self.control_panel.view_mode_clicked.connect(...)
        # etc.

        # Apply initial theme
        self.apply_theme(self.current_theme)

        # Set window size
        self.resize(1280, 960)

        logger.info("Qt GUI initialized successfully")

    def apply_theme(self, theme_name: str):
        """Apply color theme"""
        self.current_theme = theme_name

        if theme_name == 'dark':
            self.setStyleSheet(DARK_THEME)
        elif theme_name == 'light':
            self.setStyleSheet(LIGHT_THEME)
        else:
            # Auto mode - default to dark
            self.setStyleSheet(DARK_THEME)

        self.control_panel.update_theme_mode(theme_name)
        logger.debug(f"Applied {theme_name} theme")

    def update_frame(self, frame: np.ndarray):
        """Update video display (called from main loop)"""
        self.video_widget.update_frame(frame)

    def update_metrics(self, fps: float, detections: int,
                       thermal_connected: bool, rgb_connected: bool):
        """Update info panel metrics"""
        if self.show_info_panel:
            self.info_panel.update_info(fps, detections,
                                        thermal_connected, rgb_connected)

    def toggle_info_panel(self, show: bool):
        """Show/hide info panel"""
        self.show_info_panel = show
        if show:
            self.info_panel.show()
        else:
            self.info_panel.hide()

    def set_view_mode(self, mode: ViewMode):
        """Update view mode"""
        self.view_mode = mode
        self.control_panel.update_view_mode(mode)

    def resizeEvent(self, event):
        """Handle window resize - reposition info panel"""
        super().resizeEvent(event)
        if self.info_panel:
            # Position in top-right corner of video widget
            self.info_panel.move(
                self.video_widget.width() - self.info_panel.width() - 20,
                10
            )

    def closeEvent(self, event):
        """Handle window close"""
        logger.info("Qt GUI closing...")
        event.accept()


# ============================================================================
# Standalone Test
# ============================================================================

def main():
    """Test Qt GUI with dummy video"""
    app = QApplication(sys.argv)

    window = DriverAppWindow()
    window.show()

    # Dummy video timer
    def update_test_frame():
        import time
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, f"Test Frame {int(time.time())}", (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        window.update_frame(frame)
        window.update_metrics(fps=30.0, detections=5,
                              thermal_connected=True, rgb_connected=False)

    timer = QTimer()
    timer.timeout.connect(update_test_frame)
    timer.start(33)  # ~30 FPS

    sys.exit(app.exec_())


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
