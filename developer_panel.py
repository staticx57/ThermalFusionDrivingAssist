#!/usr/bin/env python3
"""
Developer Panel for Thermal Fusion Driving Assist
Real-time performance metrics, diagnostics, and debugging overlays
"""
import logging
from collections import deque
from typing import Optional, Dict
import time

try:
    from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                                  QFrame, QScrollArea, QGridLayout, QPushButton)
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtGui import QFont, QPalette, QColor
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# Developer Panel Styling
# ============================================================================

DEVELOPER_PANEL_STYLE = """
QFrame#DeveloperPanel {
    background-color: #0d0d0d;
    border-left: 2px solid #00aaff;
}

QLabel#SectionHeader {
    color: #00aaff;
    font-size: 12px;
    font-weight: bold;
    padding: 5px 0px;
    border-bottom: 1px solid #404040;
}

QLabel#MetricLabel {
    color: #b0b0b0;
    font-size: 10px;
    padding: 2px 5px;
}

QLabel#MetricValue {
    color: #ffffff;
    font-size: 11px;
    font-weight: bold;
    padding: 2px 5px;
}

QLabel#StatusGood {
    color: #00ff00;
    font-size: 10px;
    font-weight: bold;
}

QLabel#StatusWarning {
    color: #ffaa00;
    font-size: 10px;
    font-weight: bold;
}

QLabel#StatusError {
    color: #ff0000;
    font-size: 10px;
    font-weight: bold;
}

QPushButton#DevButton {
    background-color: #1a1a1a;
    color: #00aaff;
    border: 1px solid #404040;
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 10px;
    min-width: 80px;
}

QPushButton#DevButton:hover {
    background-color: #2a2a2a;
    border: 1px solid #00aaff;
}
"""


class FPSGraph(QWidget):
    """Mini FPS graph widget"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.fps_history = deque(maxlen=60)  # Last 60 FPS samples
        self.setMinimumHeight(60)
        self.setMaximumHeight(60)

    def update_fps(self, fps: float):
        """Update FPS history"""
        self.fps_history.append(fps)
        self.update()  # Trigger repaint


class DeveloperPanel(QFrame):
    """
    Developer mode panel with comprehensive diagnostics and metrics
    Shows real-time performance, camera status, detection stats, and system info
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("DeveloperPanel")
        self.setFixedWidth(300)
        self.setStyleSheet(DEVELOPER_PANEL_STYLE)

        # State
        self.app = None  # Reference to main app
        self.start_time = time.time()

        # Build UI
        self._build_ui()

        # Update timer (10 FPS for developer panel)
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._auto_update)
        self.update_timer.start(100)  # 10 Hz update

    def _build_ui(self):
        """Build developer panel UI"""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Scroll area for metrics
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout()
        scroll_layout.setSpacing(15)

        # === Performance Section ===
        perf_section = self._create_section("PERFORMANCE")
        self.fps_label = self._create_metric_row("FPS:", "0.0")
        self.frame_time_label = self._create_metric_row("Frame Time:", "0.0 ms")
        self.frame_count_label = self._create_metric_row("Frames:", "0")
        self.uptime_label = self._create_metric_row("Uptime:", "00:00:00")
        perf_section.layout().addWidget(self.fps_label)
        perf_section.layout().addWidget(self.frame_time_label)
        perf_section.layout().addWidget(self.frame_count_label)
        perf_section.layout().addWidget(self.uptime_label)
        scroll_layout.addWidget(perf_section)

        # === System Resources Section ===
        sys_section = self._create_section("SYSTEM RESOURCES")
        self.cpu_label = self._create_metric_row("CPU:", "0%")
        self.gpu_label = self._create_metric_row("GPU:", "0%")
        self.memory_label = self._create_metric_row("Memory:", "0 MB")
        self.temp_label = self._create_metric_row("Temp:", "0°C")
        self.power_label = self._create_metric_row("Power:", "0.0 W")
        sys_section.layout().addWidget(self.cpu_label)
        sys_section.layout().addWidget(self.gpu_label)
        sys_section.layout().addWidget(self.memory_label)
        sys_section.layout().addWidget(self.temp_label)
        sys_section.layout().addWidget(self.power_label)
        scroll_layout.addWidget(sys_section)

        # === Camera Status Section ===
        cam_section = self._create_section("CAMERAS")
        self.thermal_status_label = self._create_status_row("Thermal:", "DISCONNECTED")
        self.thermal_res_label = self._create_metric_row("  Resolution:", "N/A")
        self.rgb_status_label = self._create_status_row("RGB:", "DISCONNECTED")
        self.rgb_res_label = self._create_metric_row("  Resolution:", "N/A")
        cam_section.layout().addWidget(self.thermal_status_label)
        cam_section.layout().addWidget(self.thermal_res_label)
        cam_section.layout().addWidget(self.rgb_status_label)
        cam_section.layout().addWidget(self.rgb_res_label)
        scroll_layout.addWidget(cam_section)

        # === Detection Section ===
        det_section = self._create_section("DETECTION")
        self.yolo_status_label = self._create_status_row("YOLO:", "DISABLED")
        self.detection_count_label = self._create_metric_row("Detections:", "0")
        self.inference_time_label = self._create_metric_row("Inference:", "0.0 ms")
        self.frame_skip_label = self._create_metric_row("Frame Skip:", "1")
        self.device_label = self._create_metric_row("Device:", "CPU")
        det_section.layout().addWidget(self.yolo_status_label)
        det_section.layout().addWidget(self.detection_count_label)
        det_section.layout().addWidget(self.inference_time_label)
        det_section.layout().addWidget(self.frame_skip_label)
        det_section.layout().addWidget(self.device_label)
        scroll_layout.addWidget(det_section)

        # === View Mode Section ===
        view_section = self._create_section("VIEW")
        self.view_mode_label = self._create_metric_row("Mode:", "thermal")
        self.fusion_mode_label = self._create_metric_row("Fusion:", "overlay")
        self.fusion_alpha_label = self._create_metric_row("Alpha:", "0.5")
        view_section.layout().addWidget(self.view_mode_label)
        view_section.layout().addWidget(self.fusion_mode_label)
        view_section.layout().addWidget(self.fusion_alpha_label)
        scroll_layout.addWidget(view_section)

        # === Threading Section ===
        thread_section = self._create_section("THREADING")
        self.gui_type_label = self._create_metric_row("GUI:", "Qt")
        self.worker_status_label = self._create_status_row("Worker:", "RUNNING")
        self.detection_thread_label = self._create_status_row("Detection Thread:", "RUNNING")
        thread_section.layout().addWidget(self.gui_type_label)
        thread_section.layout().addWidget(self.worker_status_label)
        thread_section.layout().addWidget(self.detection_thread_label)
        scroll_layout.addWidget(thread_section)

        # === Camera Management Section ===
        cam_section = self._create_section("CAMERA MANAGEMENT")
        
        # Camera list widget
        from PyQt5.QtWidgets import QListWidget
        self.camera_list = QListWidget()
        self.camera_list.setStyleSheet("""
            QListWidget {
                background-color: #0d0d0d;
                border: 1px solid #404040;
                color: #ffffff;
                font-size: 10px;
                padding: 4px;
            }
            QListWidget::item {
                padding: 3px;
            }
            QListWidget::item:selected {
                background-color: #2a2a2a;
                border: 1px solid #00aaff;
            }
        """)
        self.camera_list.setMaximumHeight(120)
        cam_section.layout().addWidget(self.camera_list)
        
        # Buttons row 1
        btn_row1 = QWidget()
        btn_layout1 = QHBoxLayout()
        btn_layout1.setContentsMargins(0, 0, 0, 0)
        btn_layout1.setSpacing(4)
        
        self.btn_assign_thermal = QPushButton("Thermal")
        self.btn_assign_thermal.setObjectName("DevButton")
        self.btn_assign_thermal.setToolTip("Assign selected camera as thermal")
        self.btn_assign_thermal.clicked.connect(self._assign_thermal)
        
        self.btn_assign_rgb = QPushButton("RGB")
        self.btn_assign_rgb.setObjectName("DevButton")
        self.btn_assign_rgb.setToolTip("Assign selected camera as RGB")
        self.btn_assign_rgb.clicked.connect(self._assign_rgb)
        
        btn_layout1.addWidget(self.btn_assign_thermal)
        btn_layout1.addWidget(self.btn_assign_rgb)
        btn_row1.setLayout(btn_layout1)
        cam_section.layout().addWidget(btn_row1)
        
        # Buttons row 2
        btn_row2 = QWidget()
        btn_layout2 = QHBoxLayout()
        btn_layout2.setContentsMargins(0, 0, 0, 0)
        btn_layout2.setSpacing(4)
        
        self.btn_rescan = QPushButton("Rescan")
        self.btn_rescan.setObjectName("DevButton")
        self.btn_rescan.setToolTip("Force camera rescan")
        self.btn_rescan.clicked.connect(self._rescan_cameras)
        
        self.btn_clear = QPushButton("Clear All")
        self.btn_clear.setObjectName("DevButton")
        self.btn_clear.setToolTip("Clear all manual assignments")
        self.btn_clear.clicked.connect(self._clear_assignments)
        
        btn_layout2.addWidget(self.btn_rescan)
        btn_layout2.addWidget(self.btn_clear)
        btn_row2.setLayout(btn_layout2)
        cam_section.layout().addWidget(btn_row2)
        
        scroll_layout.addWidget(cam_section)

        # Spacer
        scroll_layout.addStretch()

        scroll_content.setLayout(scroll_layout)
        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)

        self.setLayout(main_layout)

    def _create_section(self, title: str) -> QFrame:
        """Create a section frame with header"""
        frame = QFrame()
        frame.setStyleSheet("QFrame { background-color: #1a1a1a; border: 1px solid #303030; border-radius: 4px; }")
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        header = QLabel(title)
        header.setObjectName("SectionHeader")
        layout.addWidget(header)

        frame.setLayout(layout)
        return frame

    def _create_metric_row(self, label_text: str, value_text: str) -> QWidget:
        """Create a metric row (label + value)"""
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        label = QLabel(label_text)
        label.setObjectName("MetricLabel")
        label.setMinimumWidth(100)

        value = QLabel(value_text)
        value.setObjectName("MetricValue")
        value.setAlignment(Qt.AlignRight)

        layout.addWidget(label)
        layout.addWidget(value)

        widget.setLayout(layout)
        widget._value_label = value  # Store reference for updates
        return widget

    def _create_status_row(self, label_text: str, status_text: str) -> QWidget:
        """Create a status row with color-coded status"""
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        label = QLabel(label_text)
        label.setObjectName("MetricLabel")
        label.setMinimumWidth(100)

        status = QLabel(status_text)
        status.setObjectName("StatusError")  # Default to error state
        status.setAlignment(Qt.AlignRight)

        layout.addWidget(label)
        layout.addWidget(status)

        widget.setLayout(layout)
        widget._status_label = status  # Store reference for updates
        return widget

    def set_app_reference(self, app):
        """Set reference to main application"""
        self.app = app
        logger.info("Developer panel connected to main app")

    def _auto_update(self):
        """Auto-update developer panel metrics"""
        if not self.app:
            return

        # Update uptime
        uptime_sec = int(time.time() - self.start_time)
        hours = uptime_sec // 3600
        minutes = (uptime_sec % 3600) // 60
        seconds = uptime_sec % 60
        self.uptime_label._value_label.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")

    def update_metrics(self, metrics: Dict):
        """
        Update all metrics from main app

        Args:
            metrics: Dictionary with all performance and status metrics
        """
        # Performance
        fps = metrics.get('fps', 0.0)
        frame_time = (1000.0 / fps) if fps > 0 else 0.0
        self.fps_label._value_label.setText(f"{fps:.1f}")
        self.frame_time_label._value_label.setText(f"{frame_time:.1f} ms")
        self.frame_count_label._value_label.setText(f"{metrics.get('frame_count', 0)}")

        # System resources
        self.cpu_label._value_label.setText(f"{metrics.get('cpu_usage', 0.0):.1f}%")
        self.gpu_label._value_label.setText(f"{metrics.get('gpu_usage', 0.0):.1f}%")
        mem_used = metrics.get('memory_used_mb', 0)
        mem_total = metrics.get('memory_total_mb', 0)
        self.memory_label._value_label.setText(f"{mem_used:.0f}/{mem_total:.0f} MB")

        temp = metrics.get('temperature', 0.0)
        self.temp_label._value_label.setText(f"{temp:.1f}°C")
        # Color code temperature
        if temp > 80:
            self.temp_label._value_label.setObjectName("StatusError")
        elif temp > 70:
            self.temp_label._value_label.setObjectName("StatusWarning")
        else:
            self.temp_label._value_label.setObjectName("StatusGood")
        self.temp_label._value_label.setStyleSheet(DEVELOPER_PANEL_STYLE)

        self.power_label._value_label.setText(f"{metrics.get('power_watts', 0.0):.1f} W")

        # Camera status
        thermal_connected = metrics.get('thermal_connected', False)
        self._update_status(self.thermal_status_label, "CONNECTED" if thermal_connected else "DISCONNECTED", thermal_connected)
        if thermal_connected:
            thermal_res = metrics.get('thermal_resolution', 'N/A')
            self.thermal_res_label._value_label.setText(str(thermal_res))
        else:
            self.thermal_res_label._value_label.setText("N/A")

        rgb_connected = metrics.get('rgb_connected', False)
        self._update_status(self.rgb_status_label, "CONNECTED" if rgb_connected else "DISCONNECTED", rgb_connected)
        if rgb_connected:
            rgb_res = metrics.get('rgb_resolution', 'N/A')
            self.rgb_res_label._value_label.setText(str(rgb_res))
        else:
            self.rgb_res_label._value_label.setText("N/A")

        # Detection
        yolo_enabled = metrics.get('yolo_enabled', False)
        self._update_status(self.yolo_status_label, "ENABLED" if yolo_enabled else "DISABLED", yolo_enabled)
        self.detection_count_label._value_label.setText(f"{metrics.get('detections', 0)}")
        self.inference_time_label._value_label.setText(f"{metrics.get('inference_time_ms', 0.0):.1f} ms")
        self.frame_skip_label._value_label.setText(f"{metrics.get('frame_skip', 1)}")
        self.device_label._value_label.setText(metrics.get('device', 'CPU'))

        # View mode
        self.view_mode_label._value_label.setText(metrics.get('view_mode', 'thermal'))
        self.fusion_mode_label._value_label.setText(metrics.get('fusion_mode', 'overlay'))
        self.fusion_alpha_label._value_label.setText(f"{metrics.get('fusion_alpha', 0.5):.2f}")

        # Threading
        self.gui_type_label._value_label.setText(metrics.get('gui_type', 'Qt'))
        worker_running = metrics.get('worker_running', True)
        self._update_status(self.worker_status_label, "RUNNING" if worker_running else "STOPPED", worker_running)
        detection_thread_alive = metrics.get('detection_thread_alive', True)
        self._update_status(self.detection_thread_label, "RUNNING" if detection_thread_alive else "STOPPED", detection_thread_alive)

    def _update_status(self, widget: QWidget, text: str, is_good: bool):
        """Update status label with color coding"""
        widget._status_label.setText(text)
        if is_good:
            widget._status_label.setObjectName("StatusGood")
        else:
            widget._status_label.setObjectName("StatusError")
        widget._status_label.setStyleSheet(DEVELOPER_PANEL_STYLE)
    
    # ========================================================================
    # Camera Management Methods
    # ========================================================================
    
    def update_camera_list(self):
        """Refresh camera list from registry"""
        from camera_registry import get_camera_registry, CameraRole
        
        self.camera_list.clear()
        registry = get_camera_registry()
        cameras = registry.get_all_cameras()
        
        for cam in cameras:
            status = "●" if cam.is_connected else "○"
            role_text = cam.role.value.upper()
            
            # Color based on connection
            color = "#ffffff" if cam.is_connected else "#666666"
            
            # Build display text
            text = f"{status} Dev {cam.device_id} - {cam.name} [{role_text}]"
            if cam.manual_override:
                text += " (manual)"
            
            from PyQt5.QtWidgets import QListWidgetItem
            from PyQt5.QtGui import QColor
            item = QListWidgetItem(text)
            item.setForeground(QColor(color))
            item.setData(Qt.UserRole, cam.device_id)
            self.camera_list.addItem(item)
        
        logger.debug(f"Camera list updated: {len(cameras)} cameras")
    
    def _assign_thermal(self):
        """Assign selected camera as thermal"""
        from camera_registry import get_camera_registry, CameraRole
        
        selected_items = self.camera_list.selectedItems()
        if selected_items:
            device_id = selected_items[0].data(Qt.UserRole)
            registry = get_camera_registry()
            registry.set_camera_role(device_id, CameraRole.THERMAL, manual=True)
            logger.info(f"Manually assigned camera {device_id} as THERMAL")
            self.update_camera_list()
            
            # Notify main app
            if self.app and hasattr(self.app, '_on_camera_assignment_changed'):
                self.app._on_camera_assignment_changed()
    
    def _assign_rgb(self):
        """Assign selected camera as RGB"""
        from camera_registry import get_camera_registry, CameraRole
        
        selected_items = self.camera_list.selectedItems()
        if selected_items:
            device_id = selected_items[0].data(Qt.UserRole)
            registry = get_camera_registry()
            registry.set_camera_role(device_id, CameraRole.RGB, manual=True)
            logger.info(f"Manually assigned camera {device_id} as RGB")
            self.update_camera_list()
            
            # Notify main app
            if self.app and hasattr(self.app, '_on_camera_assignment_changed'):
                self.app._on_camera_assignment_changed()
    
    def _rescan_cameras(self):
        """Force camera rescan"""
        from PyQt5.QtCore import QTimer
        logger.info("Forcing camera rescan...")
        
        try:
            from camera_monitor import get_camera_monitor
            monitor = get_camera_monitor()
            monitor.force_scan()
            logger.info("Camera rescan triggered")
        except Exception as e:
            logger.error(f"Failed to trigger camera rescan: {e}")
        
        # Update list after short delay
        QTimer.singleShot(500, self.update_camera_list)
    
    def _clear_assignments(self):
        """Clear all manual camera assignments"""
        from camera_registry import get_camera_registry
        
        registry = get_camera_registry()
        registry.clear_manual_assignments()
        logger.info("Cleared all manual camera assignments")
        self.update_camera_list()
        
        # Notify main app
        if self.app and hasattr(self.app, '_on_camera_assignment_changed'):
            self.app._on_camera_assignment_changed()
