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
from object_detector import Detection
from road_analyzer import Alert, AlertLevel
from view_mode import ViewMode
from developer_panel import DeveloperPanel
from alert_overlay import AlertOverlayWidget

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
    color: #006600;
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
    color: #00ff00;
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
    High-performance video display widget with ADAS alert overlay
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

        # Alert overlay (ADAS-compliant)
        self.alert_overlay = AlertOverlayWidget(self)
        self.alert_overlay.setGeometry(self.rect())  # Cover entire video widget

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

    def update_alerts(self, alerts: List[Alert], detections: List[Detection]):
        """
        Update ADAS alert overlay

        Args:
            alerts: List of Alert objects from RoadAnalyzer
            detections: List of Detection objects for proximity zones
        """
        if self.alert_overlay:
            self.alert_overlay.update_alerts(alerts, detections)

    def resizeEvent(self, event):
        """Handle resize to keep alert overlay sized correctly"""
        super().resizeEvent(event)
        if hasattr(self, 'alert_overlay') and self.alert_overlay:
            self.alert_overlay.setGeometry(self.rect())

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

    # Developer control signals
    buffer_flush_toggled = pyqtSignal(bool)
    frame_skip_clicked = pyqtSignal()
    palette_clicked = pyqtSignal()
    detection_toggled = pyqtSignal(bool)
    device_clicked = pyqtSignal()
    model_clicked = pyqtSignal()
    fusion_mode_clicked = pyqtSignal()
    fusion_alpha_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        # Create buttons
        self.view_btn = QPushButton("🎥 View: Thermal")
        self.yolo_btn = QPushButton("🎯 YOLO: OFF")
        self.audio_btn = QPushButton("🔊 Audio: ON")
        self.info_btn = QPushButton("ℹ️ Info")
        self.theme_btn = QPushButton("🎨 Theme: Auto")
        self.retry_btn = QPushButton("🔄 Retry Sensors")

        # Developer control buttons (hidden by default)
        self.buffer_flush_btn = QPushButton("💾 Flush: OFF")
        self.frame_skip_btn = QPushButton("⏩ Skip: 1")
        self.palette_btn = QPushButton("🎨 PAL: IRONBOW")
        self.detection_btn = QPushButton("📦 BOX: ON")
        self.device_btn = QPushButton("🖥️ DEV: CPU")
        self.model_btn = QPushButton("🤖 MDL: V8N")
        self.fusion_mode_btn = QPushButton("🔀 FUS: ALPHA")
        self.fusion_alpha_btn = QPushButton("⚖️ α: 0.50")

        # Hide all developer controls by default
        self.buffer_flush_btn.hide()
        self.frame_skip_btn.hide()
        self.palette_btn.hide()
        self.detection_btn.hide()
        self.device_btn.hide()
        self.model_btn.hide()
        self.fusion_mode_btn.hide()
        self.fusion_alpha_btn.hide()

        # Make toggle buttons checkable
        self.yolo_btn.setCheckable(True)
        self.audio_btn.setCheckable(True)
        self.info_btn.setCheckable(True)
        self.buffer_flush_btn.setCheckable(True)
        self.detection_btn.setCheckable(True)

        # Set initial checked states
        self.audio_btn.setChecked(True)  # Audio on by default

        # Connect signals
        self.view_btn.clicked.connect(self.view_mode_clicked.emit)
        self.yolo_btn.toggled.connect(self._on_yolo_toggled)
        self.audio_btn.toggled.connect(self._on_audio_toggled)
        self.info_btn.toggled.connect(self._on_info_toggled)
        self.theme_btn.clicked.connect(self.theme_clicked.emit)
        self.retry_btn.clicked.connect(self.retry_sensors_clicked.emit)

        # Developer control connections
        self.buffer_flush_btn.toggled.connect(self._on_buffer_flush_toggled)
        self.frame_skip_btn.clicked.connect(self.frame_skip_clicked.emit)
        self.palette_btn.clicked.connect(self.palette_clicked.emit)
        self.detection_btn.toggled.connect(self._on_detection_toggled)
        self.device_btn.clicked.connect(self.device_clicked.emit)
        self.model_btn.clicked.connect(self.model_clicked.emit)
        self.fusion_mode_btn.clicked.connect(self.fusion_mode_clicked.emit)
        self.fusion_alpha_btn.clicked.connect(self.fusion_alpha_clicked.emit)

        # Layout
        layout = QHBoxLayout()
        layout.addWidget(self.view_btn)
        layout.addWidget(self.yolo_btn)
        layout.addWidget(self.audio_btn)
        layout.addWidget(self.info_btn)

        # Developer controls row 1 (camera & detection)
        layout.addWidget(self.palette_btn)
        layout.addWidget(self.detection_btn)
        layout.addWidget(self.device_btn)
        layout.addWidget(self.model_btn)

        # Developer controls row 2 (performance & view)
        layout.addWidget(self.buffer_flush_btn)
        layout.addWidget(self.frame_skip_btn)
        layout.addWidget(self.fusion_mode_btn)
        layout.addWidget(self.fusion_alpha_btn)

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

    def _on_buffer_flush_toggled(self, checked: bool):
        """Update buffer flush button text when toggled"""
        self.buffer_flush_btn.setText(f"💾 Flush: {'ON' if checked else 'OFF'}")
        self.buffer_flush_toggled.emit(checked)

    def set_buffer_flush_enabled(self, enabled: bool):
        """Set buffer flush button state"""
        self.buffer_flush_btn.setChecked(enabled)

    def set_frame_skip_value(self, value: int):
        """Update frame skip button text"""
        self.frame_skip_btn.setText(f"⏩ Skip: {value}")

    def _on_detection_toggled(self, checked: bool):
        """Update detection toggle button text when toggled"""
        self.detection_btn.setText(f"📦 BOX: {'ON' if checked else 'OFF'}")
        self.detection_toggled.emit(checked)

    def set_palette(self, palette_name: str):
        """Update palette button text"""
        # Truncate palette name to fit button (8 chars max)
        short_name = palette_name[:8].upper()
        self.palette_btn.setText(f"🎨 PAL: {short_name}")

    def set_detection_enabled(self, enabled: bool):
        """Set detection toggle button state"""
        self.detection_btn.setChecked(enabled)

    def set_device(self, device: str):
        """Update device button text"""
        self.device_btn.setText(f"🖥️ DEV: {device.upper()}")

    def set_model(self, model_name: str):
        """Update model button text"""
        # Convert yolov8n.pt -> V8N
        short_name = model_name.replace('.pt', '').replace('yolov8', 'V8').upper()
        self.model_btn.setText(f"🤖 MDL: {short_name}")

    def set_fusion_mode(self, mode: str):
        """Update fusion mode button text"""
        # Truncate mode name (max 6 chars)
        short_name = mode[:6].upper()
        self.fusion_mode_btn.setText(f"🔀 FUS: {short_name}")

    def set_fusion_alpha(self, alpha: float):
        """Update fusion alpha button text"""
        self.fusion_alpha_btn.setText(f"⚖️ α: {alpha:.2f}")

    def show_developer_controls(self, show: bool):
        """Show or hide developer control buttons"""
        if show:
            self.buffer_flush_btn.show()
            self.frame_skip_btn.show()
            self.palette_btn.show()
            self.detection_btn.show()
            self.device_btn.show()
            self.model_btn.show()
            self.fusion_mode_btn.show()
            self.fusion_alpha_btn.show()
            logger.info("Developer controls shown (all 8 controls)")
        else:
            self.buffer_flush_btn.hide()
            self.frame_skip_btn.hide()
            self.palette_btn.hide()
            self.detection_btn.hide()
            self.device_btn.hide()
            self.model_btn.hide()
            self.fusion_mode_btn.hide()
            self.fusion_alpha_btn.hide()
            logger.info("Developer controls hidden")


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

    def __init__(self, app=None):
        super().__init__()

        if not PYQT_AVAILABLE:
            raise ImportError("PyQt5 is required. Install with: pip3 install PyQt5")

        self.setWindowTitle("Thermal Fusion Driving Assist - Qt Edition")

        # Reference to main application (for button callbacks)
        self.app = app

        # Application state
        self.current_theme = 'dark'
        self.view_mode = ViewMode.THERMAL_ONLY
        self.show_info_panel = False
        self.developer_mode = False  # Developer panel hidden by default

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Horizontal layout: Main content | Developer Panel
        horizontal_layout = QHBoxLayout()
        horizontal_layout.setContentsMargins(0, 0, 0, 0)
        horizontal_layout.setSpacing(0)

        # Left side: Video + Controls (vertical layout)
        left_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Video display (90% of space)
        self.video_widget = VideoWidget()
        main_layout.addWidget(self.video_widget, stretch=9)

        # Control panel (10% of space)
        self.control_panel = ControlPanel()
        main_layout.addWidget(self.control_panel, stretch=1)

        left_widget.setLayout(main_layout)
        horizontal_layout.addWidget(left_widget)

        # Right side: Developer Panel (initially hidden)
        self.developer_panel = DeveloperPanel()
        self.developer_panel.set_app_reference(app)
        self.developer_panel.hide()  # Hidden by default
        horizontal_layout.addWidget(self.developer_panel)

        central_widget.setLayout(horizontal_layout)

        # Info panel overlay (on top of video widget)
        self.info_panel = InfoPanel(self.video_widget)

        # Connect control panel signals
        self._connect_controls()

        # Apply initial theme
        self.apply_theme(self.current_theme)

        # Set window size
        self.resize(1280, 960)

        logger.info("Qt GUI initialized successfully")

    def set_app(self, app):
        """Set reference to main application after initialization"""
        self.app = app
        self._initialize_control_states()
        logger.info("Application reference set in Qt GUI")

    def _initialize_control_states(self):
        """Initialize all control button states from app configuration"""
        if not self.app:
            return

        # Standard controls
        self.control_panel.set_yolo_enabled(getattr(self.app, 'yolo_enabled', True))
        self.control_panel.set_audio_enabled(getattr(self.app, 'audio_enabled', True))
        self.control_panel.update_view_mode(getattr(self.app, 'view_mode', ViewMode.THERMAL_ONLY))

        # Developer controls - initial states
        self.control_panel.set_buffer_flush_enabled(getattr(self.app, 'buffer_flush_enabled', False))
        self.control_panel.set_frame_skip_value(getattr(self.app, 'frame_skip_value', 1))
        self.control_panel.set_detection_enabled(getattr(self.app, 'show_detections', True))
        self.control_panel.set_device(getattr(self.app, 'device', 'cpu'))

        # Get palette from detector if available
        if self.app.detector and hasattr(self.app.detector, 'palette_name'):
            self.control_panel.set_palette(self.app.detector.palette_name)
        else:
            self.control_panel.set_palette('ironbow')

        # Get model name
        model_name = getattr(self.app, 'model_name', 'yolov8n.pt')
        self.control_panel.set_model(model_name)

        # Get fusion settings
        fusion_mode = getattr(self.app, 'fusion_mode', 'alpha_blend')
        fusion_alpha = getattr(self.app, 'fusion_alpha', 0.5)
        self.control_panel.set_fusion_mode(fusion_mode)
        self.control_panel.set_fusion_alpha(fusion_alpha)

        logger.info("All control states initialized from app configuration")

    def _connect_controls(self):
        """Connect control panel button signals to handlers"""
        self.control_panel.view_mode_clicked.connect(self._on_view_mode_cycle)
        self.control_panel.yolo_toggled.connect(self._on_yolo_toggle)
        self.control_panel.audio_toggled.connect(self._on_audio_toggle)
        self.control_panel.info_toggled.connect(self._on_info_toggle)
        self.control_panel.theme_clicked.connect(self._on_theme_toggle)
        self.control_panel.retry_sensors_clicked.connect(self._on_retry_sensors)
        # Developer controls
        self.control_panel.buffer_flush_toggled.connect(self._on_buffer_flush_toggle)
        self.control_panel.frame_skip_clicked.connect(self._on_frame_skip_cycle)
        self.control_panel.palette_clicked.connect(self._on_palette_cycle)
        self.control_panel.detection_toggled.connect(self._on_detection_toggle)
        self.control_panel.device_clicked.connect(self._on_device_toggle)
        self.control_panel.model_clicked.connect(self._on_model_cycle)
        self.control_panel.fusion_mode_clicked.connect(self._on_fusion_mode_cycle)
        self.control_panel.fusion_alpha_clicked.connect(self._on_fusion_alpha_adjust)
        logger.info("Control panel signals connected (all 14 controls)")

    def _on_view_mode_cycle(self):
        """Cycle through view modes"""
        if not self.app:
            return

        view_modes = [ViewMode.THERMAL_ONLY, ViewMode.RGB_ONLY, ViewMode.FUSION,
                      ViewMode.SIDE_BY_SIDE, ViewMode.PICTURE_IN_PICTURE]
        current_idx = view_modes.index(self.app.view_mode) if self.app.view_mode in view_modes else 0
        next_idx = (current_idx + 1) % len(view_modes)
        self.app.view_mode = view_modes[next_idx]
        self.set_view_mode(view_modes[next_idx])
        logger.info(f"View mode changed to: {view_modes[next_idx]}")

    def _on_yolo_toggle(self, enabled: bool):
        """Toggle YOLO detection"""
        if not self.app:
            return
        self.app.yolo_enabled = enabled
        logger.info(f"YOLO detection: {'enabled' if enabled else 'disabled'}")

    def _on_audio_toggle(self, enabled: bool):
        """Toggle audio alerts"""
        if not self.app:
            return
        self.app.audio_enabled = enabled
        logger.info(f"Audio alerts: {'enabled' if enabled else 'disabled'}")

    def _on_info_toggle(self, show: bool):
        """Toggle info panel"""
        self.show_info_panel = show
        self.toggle_info_panel(show)
        logger.info(f"Info panel: {'shown' if show else 'hidden'}")

    def _on_theme_toggle(self):
        """Cycle through themes"""
        themes = ['dark', 'light', 'auto']
        current_idx = themes.index(self.current_theme) if self.current_theme in themes else 0
        next_idx = (current_idx + 1) % len(themes)
        self.apply_theme(themes[next_idx])
        logger.info(f"Theme changed to: {themes[next_idx]}")

    def _on_retry_sensors(self):
        """Retry sensor connections"""
        if not self.app:
            return
        logger.info("Retry sensors requested from GUI")
        # Reset scan timers to force immediate retry
        self.app.last_thermal_scan_time = 0
        if hasattr(self.app, 'frame_count'):
            self.app.frame_count = 0  # Trigger RGB retry on next frame

    def _on_buffer_flush_toggle(self, enabled: bool):
        """Toggle buffer flush"""
        if not self.app:
            return
        self.app.buffer_flush_enabled = enabled
        logger.info(f"Buffer flush: {'enabled' if enabled else 'disabled'}")

    def _on_frame_skip_cycle(self):
        """Cycle through frame skip values (1, 2, 3, 5, back to 1)"""
        if not self.app:
            return
        skip_values = [1, 2, 3, 5]
        current_idx = skip_values.index(self.app.frame_skip_value) if self.app.frame_skip_value in skip_values else 0
        next_idx = (current_idx + 1) % len(skip_values)
        self.app.frame_skip_value = skip_values[next_idx]
        self.control_panel.set_frame_skip_value(skip_values[next_idx])
        logger.info(f"Frame skip set to: {skip_values[next_idx]}")

    def _on_palette_cycle(self):
        """Cycle through thermal color palettes"""
        if not self.app or not self.app.detector:
            return
        # Available palettes in ThermalObjectDetector
        palettes = ['ironbow', 'whitehot', 'blackhot', 'rainbow', 'arctic', 'grayscale']
        current = getattr(self.app.detector, 'palette_name', 'ironbow')
        current_idx = palettes.index(current) if current in palettes else 0
        next_idx = (current_idx + 1) % len(palettes)
        next_palette = palettes[next_idx]

        # Update detector palette
        if hasattr(self.app.detector, 'set_palette'):
            self.app.detector.set_palette(next_palette)
        self.control_panel.set_palette(next_palette)
        logger.info(f"Thermal palette: {next_palette}")

    def _on_detection_toggle(self, enabled: bool):
        """Toggle detection bounding box display"""
        if not self.app:
            return
        self.app.show_detections = enabled
        logger.info(f"Detection bounding boxes: {'shown' if enabled else 'hidden'}")

    def _on_device_toggle(self):
        """Toggle between CPU and CUDA for detection"""
        if not self.app:
            return
        current_device = getattr(self.app, 'device', 'cpu')
        new_device = 'cuda' if current_device == 'cpu' else 'cpu'

        # Update device (requires detector reload)
        self.app.device = new_device
        if self.app.detector and hasattr(self.app.detector, 'model'):
            try:
                self.app.detector.model.to(new_device)
                logger.info(f"Detection device switched to: {new_device.upper()}")
            except Exception as e:
                logger.error(f"Failed to switch device to {new_device}: {e}")
                # Revert on failure
                self.app.device = current_device
                new_device = current_device

        self.control_panel.set_device(new_device)

    def _on_model_cycle(self):
        """Cycle through YOLO model variants (n, s, m, l)"""
        if not self.app:
            return
        models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt']
        current = getattr(self.app, 'model_name', 'yolov8n.pt')
        current_idx = models.index(current) if current in models else 0
        next_idx = (current_idx + 1) % len(models)
        next_model = models[next_idx]

        # Update model (requires detector reload)
        self.app.model_name = next_model
        logger.info(f"YOLO model changed to: {next_model} (requires restart to apply)")
        # Note: Actual model reload happens on next detector initialization
        self.control_panel.set_model(next_model)

    def _on_fusion_mode_cycle(self):
        """Cycle through fusion blend modes"""
        if not self.app:
            return
        # Fusion modes from FusionProcessor
        modes = ['alpha_blend', 'weighted', 'overlay', 'highlight']
        current = getattr(self.app, 'fusion_mode', 'alpha_blend')
        current_idx = modes.index(current) if current in modes else 0
        next_idx = (current_idx + 1) % len(modes)
        next_mode = modes[next_idx]

        self.app.fusion_mode = next_mode
        if self.app.fusion_processor and hasattr(self.app.fusion_processor, 'set_mode'):
            self.app.fusion_processor.set_mode(next_mode)

        self.control_panel.set_fusion_mode(next_mode)
        logger.info(f"Fusion mode: {next_mode}")

    def _on_fusion_alpha_adjust(self):
        """Adjust fusion alpha value (0.3, 0.5, 0.7, 0.9, back to 0.3)"""
        if not self.app:
            return
        alpha_values = [0.3, 0.5, 0.7, 0.9]
        current = getattr(self.app, 'fusion_alpha', 0.5)
        # Find closest value
        current_idx = min(range(len(alpha_values)), key=lambda i: abs(alpha_values[i] - current))
        next_idx = (current_idx + 1) % len(alpha_values)
        next_alpha = alpha_values[next_idx]

        self.app.fusion_alpha = next_alpha
        if self.app.fusion_processor and hasattr(self.app.fusion_processor, 'set_alpha'):
            self.app.fusion_processor.set_alpha(next_alpha)

        self.control_panel.set_fusion_alpha(next_alpha)
        logger.info(f"Fusion alpha: {next_alpha}")

    def toggle_developer_mode(self):
        """
        Toggle developer mode panel and controls (Ctrl+D)
        Note: Developer panel not intended for use while driving
        """
        self.developer_mode = not self.developer_mode
        if self.developer_mode:
            self.developer_panel.show()
            self.control_panel.show_developer_controls(True)
            logger.info("✓ Developer mode ENABLED (panel + controls visible)")
        else:
            self.developer_panel.hide()
            self.control_panel.show_developer_controls(False)
            logger.info("✗ Developer mode DISABLED")

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        key = event.key()
        modifiers = event.modifiers()

        # Ctrl+D: Toggle developer mode
        if key == Qt.Key_D and modifiers & Qt.ControlModifier:
            self.toggle_developer_mode()
            return

        if key == Qt.Key_Q or key == Qt.Key_Escape:
            # Quit
            if self.app:
                self.app.running = False
            self.close()

        elif key == Qt.Key_V:
            # Cycle view mode
            self._on_view_mode_cycle()

        elif key == Qt.Key_Y:
            # Toggle YOLO
            if self.app:
                self.app.yolo_enabled = not self.app.yolo_enabled
                self.control_panel.set_yolo_enabled(self.app.yolo_enabled)

        elif key == Qt.Key_A:
            # Toggle audio
            if self.app:
                self.app.audio_enabled = not self.app.audio_enabled
                self.control_panel.set_audio_enabled(self.app.audio_enabled)

        elif key == Qt.Key_I:
            # Toggle info panel
            self.show_info_panel = not self.show_info_panel
            self.toggle_info_panel(self.show_info_panel)

        elif key == Qt.Key_T:
            # Cycle theme
            self._on_theme_toggle()

        elif key == Qt.Key_D:
            # Toggle detections display
            if self.app:
                self.app.show_detections = not self.app.show_detections
                logger.info(f"Show detections: {self.app.show_detections}")

        elif key == Qt.Key_R:
            # Retry sensors
            self._on_retry_sensors()

        elif key == Qt.Key_F:
            # Toggle fullscreen
            self.toggle_fullscreen()

        else:
            # Pass to parent
            super().keyPressEvent(event)

    def apply_theme(self, theme_name: str):
        """
        Apply color theme

        Design principle for ADAS driving safety:
        - 'light' mode = BRIGHT display for DAYTIME (offsets glare)
        - 'dark' mode = DARK display for NIGHTTIME (preserves night vision)
        """
        self.current_theme = theme_name

        if theme_name == 'dark':
            # Night mode: DARK display to preserve night vision
            self.setStyleSheet(DARK_THEME)
        elif theme_name == 'light':
            # Day mode: BRIGHT display to offset glare
            self.setStyleSheet(LIGHT_THEME)
        else:
            # Auto mode - default to dark for safety
            self.setStyleSheet(DARK_THEME)

        self.control_panel.update_theme_mode(theme_name)
        logger.info(f"Applied '{theme_name}' theme (Day=bright, Night=dark)")

    def connect_worker_signals(self, worker):
        """
        Connect VideoProcessorWorker signals to GUI slots (Phase 3)
        This enables thread-safe communication from worker to GUI
        """
        worker.frame_ready.connect(self._on_frame_ready)
        worker.metrics_update.connect(self._on_metrics_update)
        logger.info("Worker signals connected to GUI")

    def _on_frame_ready(self, frame):
        """Handle frame_ready signal from worker thread (runs in main thread)"""
        self.update_frame(frame)

    def _on_metrics_update(self, metrics):
        """Handle metrics_update signal from worker thread (runs in main thread)"""
        self.update_metrics(
            fps=metrics.get('fps', 0.0),
            detections=metrics.get('detections', 0),
            thermal_connected=metrics.get('thermal_connected', False),
            rgb_connected=metrics.get('rgb_connected', False)
        )
        # Update control panel states
        if self.app:
            self.set_view_mode(self.app.view_mode)
            self.control_panel.set_yolo_enabled(self.app.yolo_enabled)
            self.control_panel.set_audio_enabled(self.app.audio_enabled)

        # Update ADAS alert overlay
        alerts = metrics.get('alerts', [])
        detections_list = metrics.get('detections_list', [])
        if alerts or detections_list:
            self.video_widget.update_alerts(alerts, detections_list)

        # Update developer panel if enabled
        if self.developer_mode and self.developer_panel:
            self.developer_panel.update_metrics(metrics)

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
