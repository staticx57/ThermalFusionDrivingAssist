"""
Enhanced Real-Time Driver GUI with Multi-View Support and Smart Alerts
Optimized for low-latency display on Jetson Orin and x86-64 workstations
Features: Thermal/RGB/Fusion views, smart proximity alerts, directional warnings
"""
import cv2
import numpy as np
from typing import List, Optional, Dict
import time
import math
from datetime import datetime

from object_detector import Detection
from road_analyzer import Alert, AlertLevel


class ViewMode:
    """Display view modes"""
    THERMAL_ONLY = "thermal"
    RGB_ONLY = "rgb"
    FUSION = "fusion"
    SIDE_BY_SIDE = "side_by_side"
    PICTURE_IN_PICTURE = "pip"


class DriverGUI:
    """Enhanced real-time GUI for driver assistance with multi-view and smart alerts"""

    def __init__(self, window_name: str = "Thermal Fusion Driving Assist", scale_factor: float = 2.0):
        """
        Initialize enhanced driver GUI

        Args:
            window_name: OpenCV window name
            scale_factor: Display scale factor for better visibility
        """
        self.window_name = window_name
        self.is_fullscreen = False
        self.scale_factor = scale_factor
        self.view_mode = ViewMode.THERMAL_ONLY  # Default to thermal

        # Modern color scheme
        self.colors = {
            'critical': (30, 30, 255),    # Red
            'warning': (0, 180, 255),     # Orange
            'info': (255, 220, 0),        # Cyan
            'success': (100, 255, 100),   # Green
            'text': (255, 255, 255),      # White
            'text_dim': (200, 200, 200),  # Dim white
            'bg': (0, 0, 0),              # Black
            'panel_bg': (35, 35, 45),     # Dark blue-gray
            'panel_accent': (45, 50, 65), # Lighter blue-gray

            # Button colors
            'button_bg': (40, 45, 55),
            'button_inactive': (50, 50, 60),
            'button_hover': (65, 70, 90),
            'button_active': (20, 180, 100),  # Green
            'button_active_alt': (100, 150, 255),  # Blue

            # Smart alert colors (pulsing)
            'danger_pulse': (0, 0, 255),     # Red pulse for critical
            'warning_pulse': (0, 165, 255),  # Orange pulse for warnings
            'accent_cyan': (255, 200, 50),
            'accent_green': (100, 255, 150),
            'accent_purple': (200, 100, 255),
        }

        # UI dimensions
        self.alert_panel_height = int(150 * scale_factor)
        self.metrics_panel_width = int(300 * scale_factor)
        self.margin = int(15 * scale_factor)

        # Font scales
        self.font_scale_small = 0.5 * scale_factor
        self.font_scale_medium = 0.7 * scale_factor
        self.font_scale_large = 1.0 * scale_factor
        self.font_thickness = max(1, int(2 * scale_factor))
        self.font_thickness_bold = max(2, int(3 * scale_factor))

        # Persistent notifications
        self.persistent_notifications = {}
        self.notification_persistence_seconds = 5.0

        # Persistent alerts
        self.persistent_alerts = {}
        self.alert_persistence_seconds = 3.0

        # GUI controls
        self.control_buttons = {}
        self.mouse_callback_registered = False

        # Smart proximity alerts (NEW)
        self.proximity_zones = {
            'left': [],     # Objects on left side
            'right': [],    # Objects on right side
            'center': [],   # Objects in center
        }
        self.pulse_phase = 0.0  # For pulsing effect
        self.last_pulse_update = time.time()

    def set_view_mode(self, mode: str):
        """Change display view mode"""
        if mode in [ViewMode.THERMAL_ONLY, ViewMode.RGB_ONLY, ViewMode.FUSION,
                   ViewMode.SIDE_BY_SIDE, ViewMode.PICTURE_IN_PICTURE]:
            self.view_mode = mode
        else:
            print(f"Invalid view mode: {mode}")

    def create_window(self, fullscreen: bool = False, window_width: int = None, window_height: int = None):
        """Create display window"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        if window_width and window_height and not fullscreen:
            cv2.resizeWindow(self.window_name, window_width, window_height)

        if fullscreen:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            self.is_fullscreen = True

    def _update_proximity_zones(self, detections: List[Detection], frame_width: int):
        """
        Update proximity zones for smart directional alerts

        Args:
            detections: List of detections
            frame_width: Frame width for zone calculation
        """
        self.proximity_zones = {'left': [], 'right': [], 'center': []}

        # Define zones (left 1/3, center 1/3, right 1/3)
        left_boundary = frame_width / 3
        right_boundary = 2 * frame_width / 3

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            center_x = (x1 + x2) / 2

            # Determine zone
            if center_x < left_boundary:
                zone = 'left'
            elif center_x > right_boundary:
                zone = 'right'
            else:
                zone = 'center'

            # Add to zone if high priority object
            if det.class_name in ['person', 'bicycle', 'motorcycle', 'motion']:
                self.proximity_zones[zone].append(det)

    def _draw_smart_proximity_alerts(self, canvas: np.ndarray):
        """
        Draw pulsing side alerts for pedestrians/animals in proximity zones
        RED PULSE on left/right sides when critical objects detected

        Args:
            canvas: Frame to draw on
        """
        h, w = canvas.shape[:2]

        # Update pulse phase (0.0 to 1.0, cycling)
        current_time = time.time()
        dt = current_time - self.last_pulse_update
        self.last_pulse_update = current_time
        self.pulse_phase = (self.pulse_phase + dt * 2.0) % 1.0  # 2 Hz pulsing

        # Calculate pulse intensity (sine wave for smooth pulsing)
        pulse_intensity = (math.sin(self.pulse_phase * 2 * math.pi) + 1.0) / 2.0  # 0.0 to 1.0

        alert_width = int(40 * self.scale_factor)
        alert_margin = int(10 * self.scale_factor)

        # LEFT SIDE ALERT
        if len(self.proximity_zones['left']) > 0:
            # Determine alert level
            has_critical = any(d.class_name in ['person', 'bicycle', 'motorcycle']
                             for d in self.proximity_zones['left'])

            if has_critical:
                color = self.colors['danger_pulse']
                alpha = 0.3 + (pulse_intensity * 0.5)  # Pulse between 0.3 and 0.8
            else:
                color = self.colors['warning_pulse']
                alpha = 0.2 + (pulse_intensity * 0.3)

            # Draw pulsing bar on left side
            overlay = canvas.copy()
            cv2.rectangle(overlay,
                         (alert_margin, alert_margin),
                         (alert_margin + alert_width, h - alert_margin),
                         color, -1)
            cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)

            # Draw icon/text
            text_y = h // 2
            icon = "<!!"  # Alert icon
            cv2.putText(canvas, icon,
                       (alert_margin + 5, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, self.font_scale_large,
                       (255, 255, 255), self.font_thickness_bold)

            # Count
            count = len(self.proximity_zones['left'])
            cv2.putText(canvas, str(count),
                       (alert_margin + 10, text_y + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, self.font_scale_medium,
                       (255, 255, 255), self.font_thickness_bold)

        # RIGHT SIDE ALERT
        if len(self.proximity_zones['right']) > 0:
            has_critical = any(d.class_name in ['person', 'bicycle', 'motorcycle']
                             for d in self.proximity_zones['right'])

            if has_critical:
                color = self.colors['danger_pulse']
                alpha = 0.3 + (pulse_intensity * 0.5)
            else:
                color = self.colors['warning_pulse']
                alpha = 0.2 + (pulse_intensity * 0.3)

            # Draw pulsing bar on right side
            overlay = canvas.copy()
            cv2.rectangle(overlay,
                         (w - alert_margin - alert_width, alert_margin),
                         (w - alert_margin, h - alert_margin),
                         color, -1)
            cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)

            # Draw icon/text
            text_y = h // 2
            icon = "!!>"
            cv2.putText(canvas, icon,
                       (w - alert_margin - alert_width + 5, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, self.font_scale_large,
                       (255, 255, 255), self.font_thickness_bold)

            # Count
            count = len(self.proximity_zones['right'])
            cv2.putText(canvas, str(count),
                       (w - alert_margin - alert_width + 10, text_y + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, self.font_scale_medium,
                       (255, 255, 255), self.font_thickness_bold)

        # CENTER WARNING (top of screen)
        if len(self.proximity_zones['center']) > 0:
            has_critical = any(d.class_name in ['person', 'bicycle', 'motorcycle']
                             for d in self.proximity_zones['center'])

            if has_critical:
                color = self.colors['danger_pulse']
                alpha = 0.4 + (pulse_intensity * 0.4)
                text = "!!! COLLISION WARNING !!!"
            else:
                color = self.colors['warning_pulse']
                alpha = 0.3 + (pulse_intensity * 0.3)
                text = "OBJECT AHEAD"

            # Draw pulsing bar at top
            bar_height = int(60 * self.scale_factor)
            overlay = canvas.copy()
            cv2.rectangle(overlay,
                         (w // 4, alert_margin),
                         (3 * w // 4, alert_margin + bar_height),
                         color, -1)
            cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)

            # Warning text
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                       self.font_scale_large, self.font_thickness_bold)[0]
            text_x = (w - text_size[0]) // 2
            text_y = alert_margin + bar_height // 2 + 10

            cv2.putText(canvas, text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, self.font_scale_large,
                       (255, 255, 255), self.font_thickness_bold)

    def render_multi_view(self, thermal_frame: Optional[np.ndarray],
                         rgb_frame: Optional[np.ndarray],
                         fusion_frame: Optional[np.ndarray],
                         detections: List[Detection],
                         alerts: List[Alert],
                         metrics: dict,
                         show_detections: bool = True,
                         gui_params: dict = None) -> np.ndarray:
        """
        Render frame with selected view mode

        Args:
            thermal_frame: Thermal camera frame
            rgb_frame: RGB camera frame (optional)
            fusion_frame: Pre-fused frame (optional)
            detections: Object detections
            alerts: Current alerts
            metrics: Performance metrics
            show_detections: Draw detection boxes
            gui_params: Additional GUI parameters (palette, YOLO status, etc.)

        Returns:
            Rendered frame
        """
        # Select display frame based on view mode
        if self.view_mode == ViewMode.THERMAL_ONLY:
            display_frame = thermal_frame
        elif self.view_mode == ViewMode.RGB_ONLY:
            display_frame = rgb_frame if rgb_frame is not None else thermal_frame
        elif self.view_mode == ViewMode.FUSION:
            display_frame = fusion_frame if fusion_frame is not None else thermal_frame
        elif self.view_mode == ViewMode.SIDE_BY_SIDE:
            if rgb_frame is not None:
                # Ensure same height
                h = thermal_frame.shape[0]
                if rgb_frame.shape[0] != h:
                    rgb_frame = cv2.resize(rgb_frame, (rgb_frame.shape[1], h))
                display_frame = np.hstack([thermal_frame, rgb_frame])
            else:
                display_frame = thermal_frame
        elif self.view_mode == ViewMode.PICTURE_IN_PICTURE:
            if rgb_frame is not None and thermal_frame is not None:
                display_frame = self._create_pip(thermal_frame, rgb_frame)
            else:
                display_frame = thermal_frame
        else:
            display_frame = thermal_frame

        # Scale up
        orig_h, orig_w = display_frame.shape[:2]
        scaled_w = int(orig_w * self.scale_factor)
        scaled_h = int(orig_h * self.scale_factor)
        frame_scaled = cv2.resize(display_frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

        canvas = frame_scaled.copy()

        # Update proximity zones for smart alerts
        self._update_proximity_zones(detections, orig_w)

        # Draw smart proximity alerts FIRST (behind everything)
        self._draw_smart_proximity_alerts(canvas)

        # Draw detections if enabled
        if show_detections:
            scaled_detections = self._scale_detections(detections, self.scale_factor)
            canvas = self._draw_detections(canvas, scaled_detections)

        # Overlay metrics panel (top-right)
        metrics_overlay = self._create_metrics_overlay(metrics, detections)
        self._overlay_panel(canvas, metrics_overlay, position='top-right')

        # Create detection alerts
        detection_alerts = self._create_detection_alerts(detections)
        persistent_road_alerts = self._persist_road_alerts(alerts)
        all_alerts = persistent_road_alerts + detection_alerts

        # Overlay alert panel (bottom) - NOW SHOWS MORE ALERTS
        if len(all_alerts) > 0:
            alert_overlay = self._create_alert_overlay(all_alerts, scaled_w)
            self._overlay_panel(canvas, alert_overlay, position='bottom')

        # Add timestamp (bottom-left)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(canvas, timestamp,
                   (int(15 * self.scale_factor), scaled_h - int(15 * self.scale_factor)),
                   cv2.FONT_HERSHEY_SIMPLEX, self.font_scale_small,
                   self.colors['text'], self.font_thickness)

        # Add view mode indicator (top-left)
        view_text = f"VIEW: {self.view_mode.upper()}"
        cv2.putText(canvas, view_text,
                   (int(15 * self.scale_factor), int(25 * self.scale_factor)),
                   cv2.FONT_HERSHEY_SIMPLEX, self.font_scale_small,
                   self.colors['accent_cyan'], self.font_thickness)

        return canvas

    def _create_pip(self, thermal: np.ndarray, rgb: np.ndarray,
                   pip_size: float = 0.25) -> np.ndarray:
        """Create picture-in-picture view"""
        main = thermal.copy()
        inset = rgb

        # Calculate inset size
        main_h, main_w = main.shape[:2]
        inset_w = int(main_w * pip_size)
        inset_h = int(main_h * pip_size)
        inset_resized = cv2.resize(inset, (inset_w, inset_h))

        # Position: bottom-right corner
        margin = 10
        x, y = main_w - inset_w - margin, main_h - inset_h - margin

        # Add border
        inset_bordered = cv2.copyMakeBorder(inset_resized, 3, 3, 3, 3,
                                           cv2.BORDER_CONSTANT, value=[255, 255, 255])

        # Overlay
        ih, iw = inset_bordered.shape[:2]
        main[y:y+ih, x:x+iw] = inset_bordered

        return main

    def render_frame_with_controls(self, thermal_frame: Optional[np.ndarray],
                                   rgb_frame: Optional[np.ndarray],
                                   fusion_frame: Optional[np.ndarray],
                                   detections: List[Detection],
                                   alerts: List[Alert],
                                   metrics: dict,
                                   show_detections: bool = True,
                                   current_palette: str = "ironbow",
                                   yolo_enabled: bool = True,
                                   buffer_flush_enabled: bool = False,
                                   frame_skip_value: int = 1,
                                   device: str = "cuda",
                                   current_model: str = "yolov8s.pt",
                                   fusion_mode: str = "alpha_blend",
                                   fusion_alpha: float = 0.5) -> np.ndarray:
        """
        Render frame with enhanced GUI controls

        Args:
            thermal_frame: Thermal frame
            rgb_frame: RGB frame (optional)
            fusion_frame: Fused frame (optional)
            detections: Detections
            alerts: Alerts
            metrics: Metrics
            show_detections: Show boxes
            current_palette: Thermal palette
            yolo_enabled: YOLO on/off
            buffer_flush_enabled: Buffer flush on/off
            frame_skip_value: Frame skip setting
            device: CUDA/CPU
            current_model: YOLO model
            fusion_mode: Fusion algorithm
            fusion_alpha: Fusion blend ratio

        Returns:
            Rendered frame with controls
        """
        # Render with multi-view support
        canvas = self.render_multi_view(
            thermal_frame, rgb_frame, fusion_frame,
            detections, alerts, metrics, show_detections
        )

        # Add enhanced control buttons
        self._draw_enhanced_controls(canvas, {
            'palette': current_palette,
            'yolo': yolo_enabled,
            'boxes': show_detections,
            'device': device,
            'model': current_model,
            'buffer_flush': buffer_flush_enabled,
            'frame_skip': frame_skip_value,
            'view_mode': self.view_mode,
            'fusion_mode': fusion_mode,
            'fusion_alpha': fusion_alpha,
            'rgb_available': rgb_frame is not None
        })

        return canvas

    def _draw_enhanced_controls(self, canvas: np.ndarray, params: dict):
        """
        Draw enhanced control panel with better layout

        Args:
            canvas: Frame to draw on
            params: GUI parameters dictionary
        """
        gui_scale = self.scale_factor * 0.85  # Slightly larger than before
        button_height = int(35 * gui_scale)   # Taller buttons
        button_spacing = int(10 * gui_scale)  # More spacing
        top_margin = int(50 * gui_scale)      # Lower to avoid view mode text
        left_margin = int(15 * gui_scale)

        self.control_buttons = {}
        current_x = left_margin
        button_y = top_margin

        # ROW 1: Camera and Detection Controls
        buttons_row1 = [
            ('palette_cycle', f"PAL: {params['palette'].upper()}", 130, None),
            ('yolo_toggle', f"YOLO: {'ON' if params['yolo'] else 'OFF'}", 90,
             self.colors['button_active'] if params['yolo'] else self.colors['button_bg']),
            ('detection_toggle', f"BOX: {'ON' if params['boxes'] else 'OFF'}", 85,
             self.colors['button_active'] if params['boxes'] else self.colors['button_bg']),
            ('device_toggle', f"DEV: {params['device'].upper()}", 100,
             self.colors['button_active'] if params['device'] == 'cuda' else self.colors['button_bg']),
            ('model_cycle', f"MODEL: {params['model'].replace('.pt', '').replace('yolov8', 'V8').upper()}", 110, None),
        ]

        for btn_id, text, width, bg_color in buttons_row1:
            w = int(width * gui_scale)
            self._draw_button_simple(canvas, current_x, button_y, w, button_height,
                                    text, btn_id, gui_scale, bg_color)
            current_x += w + button_spacing

        # ROW 2: Performance and View Controls
        button_y += button_height + button_spacing
        current_x = left_margin

        buttons_row2 = [
            ('buffer_flush_toggle', f"FLUSH: {'ON' if params['buffer_flush'] else 'OFF'}", 95,
             self.colors['button_active'] if params['buffer_flush'] else self.colors['button_bg']),
            ('audio_toggle', f"AUDIO: {'ON' if params.get('audio_enabled', True) else 'OFF'}", 95,
             self.colors['button_active'] if params.get('audio_enabled', True) else self.colors['button_bg']),
            ('frame_skip_cycle', f"SKIP: 1/{params['frame_skip']+1}", 85, None),
            ('view_mode_cycle', f"VIEW: {params['view_mode'][:4].upper()}", 100, self.colors['button_active_alt']),
        ]

        # Only show fusion controls if RGB available
        if params['rgb_available']:
            buttons_row2.extend([
                ('fusion_mode_cycle', f"FUS: {params['fusion_mode'][:5].upper()}", 110, None),
                ('fusion_alpha_adjust', f"α: {params['fusion_alpha']:.1f}", 70, None),
            ])

        for btn_id, text, width, bg_color in buttons_row2:
            w = int(width * gui_scale)
            self._draw_button_simple(canvas, current_x, button_y, w, button_height,
                                    text, btn_id, gui_scale, bg_color)
            current_x += w + button_spacing

    def _draw_button_simple(self, canvas: np.ndarray, x: int, y: int, w: int, h: int,
                           text: str, button_id: str, gui_scale: float, bg_color: tuple = None):
        """Draw a single button (simplified)"""
        if bg_color is None:
            bg_color = self.colors['button_bg']

        self.control_buttons[button_id] = (x, y, w, h)

        is_active = bg_color == self.colors['button_active'] or bg_color == self.colors['button_active_alt']

        # Draw button with transparency
        overlay = canvas[y:y+h, x:x+w].copy()
        button_bg = np.full((h, w, 3), bg_color, dtype=np.uint8)
        cv2.addWeighted(button_bg, 0.75, overlay, 0.25, 0, overlay)
        canvas[y:y+h, x:x+w] = overlay

        # Border
        border_color = self.colors['accent_green'] if is_active else self.colors['panel_accent']
        border_thickness = max(2, int(2 * gui_scale)) if is_active else max(1, int(1 * gui_scale))
        cv2.rectangle(canvas, (x, y), (x + w, y + h), border_color, border_thickness)

        # Text (centered)
        text_color = self.colors['text'] if is_active else self.colors['text_dim']
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                   self.font_scale_small * 0.75, self.font_thickness)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (h + text_size[1]) // 2

        cv2.putText(canvas, text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, self.font_scale_small * 0.75,
                   text_color, max(1, int(self.font_thickness * 0.8)))

    def check_button_click(self, x: int, y: int) -> Optional[str]:
        """Check if click hits a button"""
        for button_id, (bx, by, bw, bh) in self.control_buttons.items():
            if bx <= x <= bx + bw and by <= y <= by + bh:
                return button_id
        return None

    # Keep existing helper methods from original GUI
    def _scale_detections(self, detections: List[Detection], scale: float) -> List[Detection]:
        """Scale detection bounding boxes"""
        scaled_dets = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            scaled_bbox = (int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale))
            scaled_det = Detection(bbox=scaled_bbox, confidence=det.confidence,
                                  class_id=det.class_id, class_name=det.class_name)
            scaled_dets.append(scaled_det)
        return scaled_dets

    def _draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw detection boxes and labels"""
        identified_objects = [d for d in detections if d.class_name != 'motion']
        motion_objects = [d for d in detections if d.class_name == 'motion']

        # Color map
        color_map = {
            'person': (0, 255, 255), 'car': (0, 255, 0), 'truck': (0, 200, 0),
            'bus': (0, 180, 0), 'bicycle': (255, 0, 255), 'motorcycle': (200, 0, 255),
            'traffic light': (0, 0, 255), 'stop sign': (0, 0, 255),
        }

        # Draw identified objects
        for det in identified_objects:
            x1, y1, x2, y2 = det.bbox
            color = color_map.get(det.class_name, (0, 255, 255))
            box_thickness = max(3, int(4 * self.scale_factor))

            # Color-code box based on distance (NEW)
            if det.distance_estimate is not None:
                distance_m = det.distance_estimate
                if distance_m < 5.0:
                    color = (0, 0, 255)  # RED - IMMEDIATE
                    box_thickness = max(4, int(6 * self.scale_factor))
                elif distance_m < 10.0:
                    color = (0, 165, 255)  # ORANGE - VERY CLOSE
                    box_thickness = max(3, int(5 * self.scale_factor))
                elif distance_m < 20.0:
                    color = (0, 255, 255)  # YELLOW - CLOSE
                else:
                    color = (0, 255, 0)  # GREEN - SAFE

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)

            # Label with distance info (NEW)
            if det.distance_estimate is not None:
                label = f"{det.class_name.upper()}: {det.distance_estimate:.1f}m ({det.confidence:.0%})"
            else:
                label = f"{det.class_name.upper()}: {det.confidence:.0%}"

            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale_medium, self.font_thickness_bold)
            padding = int(8 * self.scale_factor)

            cv2.rectangle(frame, (x1, y1 - label_h - baseline - padding * 2),
                         (x1 + label_w + padding * 2, y1), color, -1)
            cv2.putText(frame, label, (x1 + padding, y1 - padding),
                       cv2.FONT_HERSHEY_SIMPLEX, self.font_scale_medium,
                       (0, 0, 0), self.font_thickness_bold)

        # Draw motion (dashed boxes)
        for det in motion_objects:
            x1, y1, x2, y2 = det.bbox
            color = (255, 165, 0)  # Orange
            self._draw_dashed_rectangle(frame, (x1, y1), (x2, y2), color, max(2, int(2 * self.scale_factor)))

            label = "MOTION"
            cv2.putText(frame, label, (x1 + 5, y2 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, self.font_scale_small,
                       color, self.font_thickness)

        return frame

    def _draw_dashed_rectangle(self, frame: np.ndarray, pt1: tuple, pt2: tuple,
                               color: tuple, thickness: int, dash_length: int = None):
        """Draw dashed rectangle"""
        if dash_length is None:
            dash_length = max(10, int(15 * self.scale_factor))

        x1, y1 = pt1
        x2, y2 = pt2

        for side in [(x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1)]:
            x_start, y_start, x_end, y_end = side
            length = int(np.sqrt((x_end - x_start)**2 + (y_end - y_start)**2))

            for i in range(0, length, dash_length * 2):
                start_ratio = i / length
                end_ratio = min((i + dash_length) / length, 1.0)
                dash_x1 = int(x_start + (x_end - x_start) * start_ratio)
                dash_y1 = int(y_start + (y_end - y_start) * start_ratio)
                dash_x2 = int(x_start + (x_end - x_start) * end_ratio)
                dash_y2 = int(y_start + (y_end - y_start) * end_ratio)
                cv2.line(frame, (dash_x1, dash_y1), (dash_x2, dash_y2), color, thickness)

    def _create_detection_alerts(self, detections: List[Detection]) -> List[Alert]:
        """Create persistent alerts for identified objects"""
        current_time = time.time()
        alerts = []

        identified = [d for d in detections if d.class_name != 'motion']
        current_counts = {}
        for det in identified:
            current_counts[det.class_name] = current_counts.get(det.class_name, 0) + 1

        for obj_type, count in current_counts.items():
            if obj_type in ['person', 'bicycle', 'motorcycle']:
                level = AlertLevel.WARNING
                message = f"{count} {obj_type.upper()}{'S' if count > 1 else ''} IDENTIFIED"
            else:
                level = AlertLevel.INFO
                message = f"{count} {obj_type.capitalize()}{'s' if count > 1 else ''} identified"

            self.persistent_notifications[obj_type] = {
                'count': count, 'timestamp': current_time, 'level': level, 'message': message}

        expired_types = [obj_type for obj_type, data in self.persistent_notifications.items()
                        if current_time - data['timestamp'] > self.notification_persistence_seconds]
        for obj_type in expired_types:
            del self.persistent_notifications[obj_type]

        for obj_type, data in self.persistent_notifications.items():
            alert = Alert(level=data['level'], message=data['message'],
                         object_type=obj_type, timestamp=data['timestamp'], position="center")
            alerts.append(alert)

        return alerts

    def _persist_road_alerts(self, alerts: List[Alert]) -> List[Alert]:
        """Persist critical/warning alerts"""
        current_time = time.time()

        for alert in alerts:
            if alert.level in [AlertLevel.CRITICAL, AlertLevel.WARNING]:
                alert_key = f"{alert.object_type}_{alert.position}_{alert.level.name}"
                self.persistent_alerts[alert_key] = alert

        expired_keys = [key for key, alert in self.persistent_alerts.items()
                       if current_time - alert.timestamp > self.alert_persistence_seconds]
        for key in expired_keys:
            del self.persistent_alerts[key]

        return list(self.persistent_alerts.values())

    def _overlay_panel(self, canvas: np.ndarray, panel: np.ndarray, position: str = 'top-right'):
        """Overlay panel on canvas"""
        panel_h, panel_w = panel.shape[:2]
        canvas_h, canvas_w = canvas.shape[:2]

        if position == 'top-right':
            x = canvas_w - panel_w - int(10 * self.scale_factor)
            y = int(110 * self.scale_factor)  # Below buttons
        elif position == 'bottom':
            x = int(10 * self.scale_factor)
            y = canvas_h - panel_h - int(10 * self.scale_factor)
        else:
            x, y = 0, 0

        x = max(0, min(x, canvas_w - panel_w))
        y = max(0, min(y, canvas_h - panel_h))

        roi = canvas[y:y+panel_h, x:x+panel_w]
        cv2.addWeighted(panel, 0.75, roi, 0.25, 0, roi)
        canvas[y:y+panel_h, x:x+panel_w] = roi

    def _create_metrics_overlay(self, metrics: dict, detections: List[Detection]) -> np.ndarray:
        """Create metrics panel"""
        gui_scale = self.scale_factor * 0.75
        panel_width = int(220 * gui_scale)

        detection_counts = {}
        for det in detections:
            detection_counts[det.class_name] = detection_counts.get(det.class_name, 0) + 1

        num_lines = 3 + len(detection_counts)
        padding_top = int(18 * gui_scale)
        padding_bottom = int(12 * gui_scale)
        line_spacing = int(26 * gui_scale)
        panel_height = padding_top + (num_lines * line_spacing) + padding_bottom

        panel = np.full((panel_height, panel_width, 3), self.colors['panel_bg'], dtype=np.uint8)
        accent_height = int(3 * gui_scale)
        panel[:accent_height, :] = self.colors['accent_cyan']

        y = padding_top + int(15 * gui_scale)
        spacing = line_spacing

        fps = metrics.get('fps', 0)
        fps_color = self.colors['success'] if fps > 20 else self.colors['warning']
        self._draw_compact_metric(panel, "FPS:", f"{fps:.1f}", y, fps_color)
        y += spacing

        inf_time = metrics.get('inference_time_ms', 0)
        self._draw_compact_metric(panel, "Inference:", f"{inf_time:.1f}ms", y)
        y += spacing

        total = len(detections)
        total_color = self.colors['critical'] if total > 0 else self.colors['text']
        self._draw_compact_metric(panel, "Objects:", str(total), y, total_color)
        y += spacing

        if detection_counts:
            for obj_type, count in sorted(detection_counts.items()):
                self._draw_compact_metric(panel, f"  {obj_type}:", str(count), y, self.colors['warning'])
                y += spacing

        return panel

    def _create_alert_overlay(self, alerts: List[Alert], max_width: int) -> np.ndarray:
        """Create alert panel - NOW SHOWS UP TO 4 ALERTS (increased from 2)"""
        gui_scale = self.scale_factor * 0.75
        num_alerts = min(len(alerts), 4)  # INCREASED FROM 2
        padding = int(18 * gui_scale)
        line_height = int(32 * gui_scale)
        panel_height = padding * 2 + (num_alerts * line_height)

        panel_width = min(max_width - int(20 * gui_scale), int(800 * gui_scale))
        panel = np.full((panel_height, panel_width, 3), self.colors['panel_bg'], dtype=np.uint8)

        sorted_alerts = sorted(alerts, key=lambda a: a.level.value, reverse=True)

        y_offset = padding + int(20 * gui_scale)
        alert_spacing = line_height
        circle_radius = int(8 * gui_scale)
        circle_offset = int(15 * gui_scale)
        text_offset = int(35 * gui_scale)

        for i, alert in enumerate(sorted_alerts[:4]):  # Show top 4
            color = self._get_alert_color(alert.level)

            cv2.circle(panel, (circle_offset, y_offset), circle_radius, color, -1)
            cv2.circle(panel, (circle_offset, y_offset), circle_radius, self.colors['text'], 1)

            message = alert.message
            cv2.putText(panel, message, (text_offset, y_offset + int(5 * gui_scale)),
                       cv2.FONT_HERSHEY_SIMPLEX, self.font_scale_medium * 0.75,
                       color, max(1, int(self.font_thickness_bold * 0.75)))

            y_offset += alert_spacing

        if len(sorted_alerts) > 4:
            count_text = f"+{len(sorted_alerts) - 4} more alerts"
            cv2.putText(panel, count_text, (text_offset, y_offset + int(5 * gui_scale)),
                       cv2.FONT_HERSHEY_SIMPLEX, self.font_scale_small * 0.75,
                       self.colors['info'], max(1, int(self.font_thickness * 0.75)))

        return panel

    def _draw_compact_metric(self, panel: np.ndarray, label: str, value: str, y: int, color: tuple = None):
        """Draw metric line"""
        if color is None:
            color = self.colors['text']

        gui_scale = self.scale_factor * 0.75
        label_x = int(10 * gui_scale)
        value_x = int(140 * gui_scale)

        cv2.putText(panel, label, (label_x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, self.font_scale_small * 0.75,
                   self.colors['text'], max(1, int(self.font_thickness * 0.75)))
        cv2.putText(panel, value, (value_x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, self.font_scale_small * 0.75,
                   color, max(1, int(self.font_thickness_bold * 0.75)))

    def _get_alert_color(self, level: AlertLevel) -> tuple:
        """Get color for alert level"""
        if level == AlertLevel.CRITICAL:
            return self.colors['critical']
        elif level == AlertLevel.WARNING:
            return self.colors['warning']
        elif level == AlertLevel.INFO:
            return self.colors['info']
        else:
            return self.colors['text']

    def display(self, frame: np.ndarray) -> int:
        """Display frame and handle keyboard input"""
        cv2.imshow(self.window_name, frame)
        return cv2.waitKey(1) & 0xFF

    def close(self):
        """Close GUI window"""
        cv2.destroyWindow(self.window_name)

    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        if self.is_fullscreen:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            self.is_fullscreen = False
        else:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            self.is_fullscreen = True
