"""
Real-Time Driver GUI with Alerts and Metrics
Optimized for low-latency display on Jetson Orin
"""
import cv2
import numpy as np
from typing import List, Optional
import time
from datetime import datetime

from object_detector import Detection
from road_analyzer import Alert, AlertLevel


class DriverGUI:
    """Real-time GUI for driver with alerts and metrics"""

    def __init__(self, window_name: str = "FLIR Thermal Road Monitor", scale_factor: float = 2.0):
        """
        Initialize driver GUI

        Args:
            window_name: OpenCV window name
            scale_factor: Display scale factor for better visibility (2.0 = 2x size)
        """
        self.window_name = window_name
        self.is_fullscreen = False
        self.scale_factor = scale_factor

        # Modern color scheme with highly visible toggles
        self.colors = {
            'critical': (30, 30, 255),    # Bright red with slight blue tint
            'warning': (0, 180, 255),     # Vibrant orange
            'info': (255, 220, 0),        # Bright cyan
            'success': (100, 255, 100),   # Soft green
            'text': (255, 255, 255),      # Pure white
            'text_dim': (200, 200, 200),  # Slightly brighter dimmed text
            'bg': (0, 0, 0),              # Deep black
            'panel_bg': (35, 35, 45),     # Dark blue-gray
            'panel_accent': (45, 50, 65), # Slightly lighter blue-gray

            # Enhanced button colors for visibility
            'button_bg': (40, 45, 55),          # Darker inactive button
            'button_inactive': (50, 50, 60),    # Clear inactive state
            'button_hover': (65, 70, 90),       # Button hover state
            'button_active': (20, 180, 100),    # Bright green active (was blue)
            'button_active_alt': (100, 150, 255), # Bright blue alternative

            # Accent colors
            'accent_cyan': (255, 200, 50),      # Cyan accent
            'accent_green': (100, 255, 150),    # Bright green accent
            'accent_purple': (200, 100, 255),   # Purple accent
            'glow_active': (80, 255, 150),      # Bright green glow
            'glow_active_alt': (120, 180, 255), # Blue glow alternative
        }

        # UI dimensions (scaled)
        self.alert_panel_height = int(150 * scale_factor)
        self.metrics_panel_width = int(300 * scale_factor)
        self.margin = int(15 * scale_factor)

        # Font scale
        self.font_scale_small = 0.5 * scale_factor
        self.font_scale_medium = 0.7 * scale_factor
        self.font_scale_large = 1.0 * scale_factor
        self.font_thickness = max(1, int(2 * scale_factor))
        self.font_thickness_bold = max(2, int(3 * scale_factor))

        # Persistent notifications for identified objects
        self.persistent_notifications = {}  # {obj_type: {'count': N, 'timestamp': time, 'level': AlertLevel}}
        self.notification_persistence_seconds = 5.0  # Keep notifications for 5 seconds

        # Persistent alerts from road analyzer (critical/warning alerts)
        self.persistent_alerts = {}  # {alert_key: Alert object}
        self.alert_persistence_seconds = 3.0  # Keep critical/warning alerts for 3 seconds

        # GUI control buttons
        self.control_buttons = {}  # {button_name: (x, y, w, h)}
        self.mouse_callback_registered = False

    def create_window(self, fullscreen: bool = False, window_width: int = None, window_height: int = None):
        """Create display window

        Args:
            fullscreen: Start in fullscreen mode
            window_width: Window width (if None, uses auto size)
            window_height: Window height (if None, uses auto size)
        """
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        # Set window size if specified
        if window_width and window_height and not fullscreen:
            cv2.resizeWindow(self.window_name, window_width, window_height)

        if fullscreen:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            self.is_fullscreen = True

    def render_frame(self, frame: np.ndarray, detections: List[Detection],
                    alerts: List[Alert], metrics: dict,
                    show_detections: bool = True) -> np.ndarray:
        """
        Render complete GUI frame with video, alerts, and metrics overlaid

        Args:
            frame: Camera frame
            detections: Object detections
            alerts: Current alerts
            metrics: Performance metrics dictionary
            show_detections: Draw detection boxes

        Returns:
            Rendered frame
        """
        # Scale up the video frame for better visibility
        orig_h, orig_w = frame.shape[:2]
        scaled_w = int(orig_w * self.scale_factor)
        scaled_h = int(orig_h * self.scale_factor)
        frame_scaled = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

        # Get scaled frame dimensions
        frame_h, frame_w = frame_scaled.shape[:2]

        # Use the video frame as the canvas (full window)
        canvas = frame_scaled.copy()

        # 1. Draw detections if enabled
        if show_detections:
            # Scale detection boxes to match scaled frame
            scaled_detections = self._scale_detections(detections, self.scale_factor)
            canvas = self._draw_detections(canvas, scaled_detections)

        # 2. Overlay metrics panel (top-right corner)
        metrics_overlay = self._create_metrics_overlay(metrics, detections)
        self._overlay_panel(canvas, metrics_overlay, position='top-right')

        # 3. Create detection alerts
        detection_alerts = self._create_detection_alerts(detections)

        # 4. Persist critical/warning alerts from road analyzer for 3 seconds
        persistent_road_alerts = self._persist_road_alerts(alerts)

        all_alerts = persistent_road_alerts + detection_alerts

        # 5. Overlay alert panel (bottom) if there are alerts
        if len(all_alerts) > 0:
            alert_overlay = self._create_alert_overlay(all_alerts, frame_w)
            self._overlay_panel(canvas, alert_overlay, position='bottom')

        # 6. Add timestamp (bottom-left corner)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(canvas, timestamp,
                   (int(15 * self.scale_factor), frame_h - int(15 * self.scale_factor)),
                   cv2.FONT_HERSHEY_SIMPLEX, self.font_scale_small,
                   self.colors['text'], self.font_thickness)

        return canvas

    def render_frame_with_controls(self, frame: np.ndarray, detections: List[Detection],
                                   alerts: List[Alert], metrics: dict,
                                   show_detections: bool = True,
                                   current_palette: str = "ironbow",
                                   yolo_enabled: bool = True,
                                   buffer_flush_enabled: bool = False,
                                   frame_skip_value: int = 1,
                                   device: str = "cuda",
                                   current_model: str = "yolov8s.pt") -> np.ndarray:
        """
        Render frame with GUI control buttons

        Args:
            frame: Camera frame
            detections: Object detections
            alerts: Current alerts
            metrics: Performance metrics
            show_detections: Draw detection boxes
            current_palette: Current thermal palette name
            yolo_enabled: Whether YOLO detection is enabled
            buffer_flush_enabled: Whether buffer flushing is enabled
            frame_skip_value: Frame skip setting (0-3)
            device: Processing device (cuda or cpu)
            current_model: Current YOLO model name

        Returns:
            Rendered frame with controls
        """
        # First render the normal frame
        canvas = self.render_frame(frame, detections, alerts, metrics, show_detections)

        # Add control buttons at the top-left
        self._draw_control_buttons(canvas, current_palette, yolo_enabled, show_detections,
                                   buffer_flush_enabled, frame_skip_value, device, current_model)

        return canvas

    def _draw_control_buttons(self, canvas: np.ndarray, current_palette: str,
                              yolo_enabled: bool, show_detections: bool,
                              buffer_flush_enabled: bool = False, frame_skip_value: int = 1,
                              device: str = "cuda", current_model: str = "yolov8s.pt"):
        """Draw interactive control buttons on the canvas - compact horizontal layout"""
        # Compact buttons aligned horizontally at the top (scaled to 75% for less obstruction)
        gui_scale = self.scale_factor * 0.75
        button_height = int(28 * gui_scale)
        button_spacing = int(8 * gui_scale)
        top_margin = int(10 * gui_scale)
        left_margin = int(10 * gui_scale)

        # Clear old button positions
        self.control_buttons = {}

        # Calculate button widths based on text
        button_y = top_margin
        current_x = left_margin

        # Button 1: Palette Cycle (wider for text)
        palette_text = current_palette.upper()
        button_width = int(110 * gui_scale)
        self._draw_compact_button(canvas, current_x, button_y, button_width, button_height,
                                  "PAL", palette_text, 'palette_cycle', gui_scale)
        current_x += button_width + button_spacing

        # Button 2: YOLO Toggle (compact)
        button_width = int(75 * gui_scale)
        yolo_text = 'ON' if yolo_enabled else 'OFF'
        yolo_color = self.colors['button_active'] if yolo_enabled else self.colors['button_bg']
        self._draw_compact_button(canvas, current_x, button_y, button_width, button_height,
                                  "YOLO", yolo_text, 'yolo_toggle', gui_scale, bg_color=yolo_color)
        current_x += button_width + button_spacing

        # Button 3: Detection Boxes Toggle (compact)
        button_width = int(75 * gui_scale)
        det_text = 'ON' if show_detections else 'OFF'
        det_color = self.colors['button_active'] if show_detections else self.colors['button_bg']
        self._draw_compact_button(canvas, current_x, button_y, button_width, button_height,
                                  "BOX", det_text, 'detection_toggle', gui_scale, bg_color=det_color)
        current_x += button_width + button_spacing

        # Button 4: Device Toggle (CUDA/CPU)
        button_width = int(85 * gui_scale)
        device_text = device.upper()
        device_color = self.colors['button_active'] if device == 'cuda' else self.colors['button_bg']
        self._draw_compact_button(canvas, current_x, button_y, button_width, button_height,
                                  "DEV", device_text, 'device_toggle', gui_scale, bg_color=device_color)
        current_x += button_width + button_spacing

        # Button 5: Buffer Flush Toggle
        button_width = int(85 * gui_scale)
        flush_text = 'ON' if buffer_flush_enabled else 'OFF'
        flush_color = self.colors['button_active'] if buffer_flush_enabled else self.colors['button_bg']
        self._draw_compact_button(canvas, current_x, button_y, button_width, button_height,
                                  "FLUSH", flush_text, 'buffer_flush_toggle', gui_scale, bg_color=flush_color)
        current_x += button_width + button_spacing

        # Button 6: Frame Skip Cycle
        button_width = int(75 * gui_scale)
        skip_text = f"1/{frame_skip_value+1}"  # Shows "1/1", "1/2", "1/3", "1/4"
        self._draw_compact_button(canvas, current_x, button_y, button_width, button_height,
                                  "SKIP", skip_text, 'frame_skip_cycle', gui_scale)
        current_x += button_width + button_spacing

        # Button 7: Model Switcher
        button_width = int(95 * gui_scale)
        model_name = current_model.replace('.pt', '').replace('yolov8', 'v8').upper()  # "V8N", "V8S", "V8M"
        self._draw_compact_button(canvas, current_x, button_y, button_width, button_height,
                                  "MODEL", model_name, 'model_cycle', gui_scale)

    def _draw_compact_button(self, canvas: np.ndarray, x: int, y: int, w: int, h: int,
                            label: str, value: str, button_id: str, gui_scale: float = None, bg_color: tuple = None):
        """Draw a simplified compact button for low latency"""
        if gui_scale is None:
            gui_scale = self.scale_factor * 0.75
        if bg_color is None:
            bg_color = self.colors['button_bg']

        # Store button position for click detection
        self.control_buttons[button_id] = (x, y, w, h)

        # Determine if button is active
        is_active = bg_color == self.colors['button_active']

        # Create semi-transparent overlay for button
        overlay = canvas[y:y+h, x:x+w].copy()
        button_bg = np.full((h, w, 3), bg_color, dtype=np.uint8)
        # 70% opacity for buttons
        cv2.addWeighted(button_bg, 0.7, overlay, 0.3, 0, overlay)
        canvas[y:y+h, x:x+w] = overlay

        # Draw simple border
        if is_active:
            border_color = self.colors['accent_green']
            border_thickness = max(2, int(2 * gui_scale))
        else:
            border_color = self.colors['panel_accent']
            border_thickness = max(1, int(1 * gui_scale))

        cv2.rectangle(canvas, (x, y), (x + w, y + h), border_color, border_thickness)

        # Draw label and value side by side (compact)
        padding = int(6 * gui_scale)
        text_y = y + h // 2 + int(4 * gui_scale)

        # Label (left-aligned, brighter for active)
        label_color = self.colors['text'] if is_active else self.colors['text_dim']
        cv2.putText(canvas, label, (x + padding, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, self.font_scale_small * 0.65 * 0.75,
                   label_color, max(1, int(self.font_thickness * 0.75)))

        # Value (right-aligned, very bright for active)
        if is_active:
            value_color = self.colors['accent_green']  # Bright green for active
        else:
            value_color = self.colors['text_dim']  # Dim for inactive

        (value_w, value_h), _ = cv2.getTextSize(
            value, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale_small * 0.9 * 0.75, max(1, int(self.font_thickness_bold * 0.75))
        )
        value_x = x + w - value_w - padding
        cv2.putText(canvas, value, (value_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, self.font_scale_small * 0.9 * 0.75,
                   value_color, max(1, int(self.font_thickness_bold * 0.75)))

    def _draw_rounded_rect(self, canvas: np.ndarray, x: int, y: int, w: int, h: int,
                          radius: int, color: tuple, thickness: int = -1, alpha: float = 1.0):
        """Draw a rounded rectangle with optional transparency"""
        if thickness == -1:
            # Filled rectangle
            overlay = canvas.copy()

            # Draw rounded corners and edges
            cv2.ellipse(overlay, (x + radius, y + radius), (radius, radius), 180, 0, 90, color, -1)
            cv2.ellipse(overlay, (x + w - radius, y + radius), (radius, radius), 270, 0, 90, color, -1)
            cv2.ellipse(overlay, (x + radius, y + h - radius), (radius, radius), 90, 0, 90, color, -1)
            cv2.ellipse(overlay, (x + w - radius, y + h - radius), (radius, radius), 0, 0, 90, color, -1)

            # Fill the body
            cv2.rectangle(overlay, (x + radius, y), (x + w - radius, y + h), color, -1)
            cv2.rectangle(overlay, (x, y + radius), (x + w, y + h - radius), color, -1)

            # Blend with alpha
            cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
        else:
            # Draw outline only
            # Top and bottom lines
            cv2.line(canvas, (x + radius, y), (x + w - radius, y), color, thickness)
            cv2.line(canvas, (x + radius, y + h), (x + w - radius, y + h), color, thickness)
            # Left and right lines
            cv2.line(canvas, (x, y + radius), (x, y + h - radius), color, thickness)
            cv2.line(canvas, (x + w, y + radius), (x + w, y + h - radius), color, thickness)
            # Corners
            cv2.ellipse(canvas, (x + radius, y + radius), (radius, radius), 180, 0, 90, color, thickness)
            cv2.ellipse(canvas, (x + w - radius, y + radius), (radius, radius), 270, 0, 90, color, thickness)
            cv2.ellipse(canvas, (x + radius, y + h - radius), (radius, radius), 90, 0, 90, color, thickness)
            cv2.ellipse(canvas, (x + w - radius, y + h - radius), (radius, radius), 0, 0, 90, color, thickness)

    def _draw_button(self, canvas: np.ndarray, x: int, y: int, w: int, h: int,
                     text: str, button_id: str, bg_color: tuple = None):
        """Draw a single button (legacy, kept for compatibility)"""
        if bg_color is None:
            bg_color = self.colors['button_bg']

        # Store button position for click detection
        self.control_buttons[button_id] = (x, y, w, h)

        # Draw button background
        cv2.rectangle(canvas, (x, y), (x + w, y + h), bg_color, -1)

        # Draw button border
        cv2.rectangle(canvas, (x, y), (x + w, y + h), self.colors['text'],
                     max(2, int(2 * self.scale_factor)))

        # Draw button text (centered)
        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale_small, self.font_thickness
        )
        text_x = x + (w - text_w) // 2
        text_y = y + (h + text_h) // 2
        cv2.putText(canvas, text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, self.font_scale_small,
                   self.colors['text'], self.font_thickness)

    def check_button_click(self, x: int, y: int) -> Optional[str]:
        """
        Check if a click position hits any button

        Args:
            x, y: Click coordinates

        Returns:
            Button ID if hit, None otherwise
        """
        for button_id, (bx, by, bw, bh) in self.control_buttons.items():
            if bx <= x <= bx + bw and by <= y <= by + bh:
                return button_id
        return None

    def _create_detection_alerts(self, detections: List[Detection]) -> List[Alert]:
        """Create persistent alerts for identified objects only (excludes motion to reduce clutter)"""
        current_time = time.time()
        alerts = []

        # Separate identified objects from motion
        identified = [d for d in detections if d.class_name != 'motion']

        # Count currently identified objects by type
        current_counts = {}
        for det in identified:
            current_counts[det.class_name] = current_counts.get(det.class_name, 0) + 1

        # Update persistent notifications for currently detected objects
        for obj_type, count in current_counts.items():
            if obj_type in ['person', 'bicycle', 'motorcycle']:
                level = AlertLevel.WARNING
                message = f"{count} {obj_type.upper()}{'S' if count > 1 else ''} IDENTIFIED"
            elif obj_type in ['car', 'truck', 'bus', 'vehicle']:
                level = AlertLevel.INFO
                message = f"{count} {obj_type.capitalize()}{'s' if count > 1 else ''} identified"
            else:
                level = AlertLevel.INFO
                message = f"{count} {obj_type.capitalize()}{'s' if count > 1 else ''} identified"

            # Update or create persistent notification
            self.persistent_notifications[obj_type] = {
                'count': count,
                'timestamp': current_time,
                'level': level,
                'message': message
            }

        # Clean up old notifications (older than 5 seconds)
        expired_types = []
        for obj_type, data in self.persistent_notifications.items():
            if current_time - data['timestamp'] > self.notification_persistence_seconds:
                expired_types.append(obj_type)

        for obj_type in expired_types:
            del self.persistent_notifications[obj_type]

        # Create alerts from all persistent notifications
        for obj_type, data in self.persistent_notifications.items():
            alert = Alert(
                level=data['level'],
                message=data['message'],
                object_type=obj_type,
                timestamp=data['timestamp'],
                position="center"
            )
            alerts.append(alert)

        # Note: Motion detections are NOT included in notifications to reduce clutter
        # They are still visible as dashed boxes on the video feed

        return alerts

    def _persist_road_alerts(self, alerts: List[Alert]) -> List[Alert]:
        """Persist critical/warning alerts from road analyzer for 3 seconds"""
        current_time = time.time()

        # Update persistent alerts with new critical/warning alerts
        for alert in alerts:
            if alert.level in [AlertLevel.CRITICAL, AlertLevel.WARNING]:
                # Create a unique key for this alert type/position
                alert_key = f"{alert.object_type}_{alert.position}_{alert.level.name}"
                self.persistent_alerts[alert_key] = alert

        # Clean up expired alerts (older than 3 seconds)
        expired_keys = []
        for alert_key, alert in self.persistent_alerts.items():
            if current_time - alert.timestamp > self.alert_persistence_seconds:
                expired_keys.append(alert_key)

        for key in expired_keys:
            del self.persistent_alerts[key]

        # Return all persistent alerts
        return list(self.persistent_alerts.values())

    def _scale_detections(self, detections: List[Detection], scale: float) -> List[Detection]:
        """Scale detection bounding boxes"""
        scaled_dets = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            scaled_bbox = (
                int(x1 * scale),
                int(y1 * scale),
                int(x2 * scale),
                int(y2 * scale)
            )
            scaled_det = Detection(
                bbox=scaled_bbox,
                confidence=det.confidence,
                class_id=det.class_id,
                class_name=det.class_name
            )
            scaled_dets.append(scaled_det)
        return scaled_dets

    def _draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw detection boxes and labels on frame with separation between identified and motion"""

        # Separate identified objects from motion detections
        identified_objects = [d for d in detections if d.class_name != 'motion']
        motion_objects = [d for d in detections if d.class_name == 'motion']

        # Color scheme for identified objects (brighter, more specific)
        identified_color_map = {
            'person': (0, 255, 255),      # Yellow - high priority
            'car': (0, 255, 0),           # Green
            'truck': (0, 200, 0),         # Dark green
            'bus': (0, 180, 0),           # Darker green
            'bicycle': (255, 0, 255),     # Magenta
            'motorcycle': (200, 0, 255),  # Purple
            'traffic light': (0, 0, 255), # Red
            'stop sign': (0, 0, 255),     # Red
        }

        # Draw identified objects first (priority display)
        for det in identified_objects:
            x1, y1, x2, y2 = det.bbox
            color = identified_color_map.get(det.class_name, (0, 255, 255))

            # Thicker, solid box for identified objects
            box_thickness = max(3, int(4 * self.scale_factor))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)

            # Draw label with solid background
            label = f"{det.class_name.upper()}: {det.confidence:.0%}"
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale_medium, self.font_thickness_bold
            )
            padding = int(8 * self.scale_factor)

            # Solid background for label
            cv2.rectangle(frame, (x1, y1 - label_h - baseline - padding * 2),
                         (x1 + label_w + padding * 2, y1), color, -1)
            cv2.putText(frame, label, (x1 + padding, y1 - padding),
                       cv2.FONT_HERSHEY_SIMPLEX, self.font_scale_medium,
                       (0, 0, 0), self.font_thickness_bold)

        # Draw motion detections with different style (lower priority, less intrusive)
        for det in motion_objects:
            x1, y1, x2, y2 = det.bbox

            # Dashed/dotted style for motion (less prominent)
            box_thickness = max(2, int(2 * self.scale_factor))
            color = (255, 165, 0)  # Orange for motion

            # Draw dashed rectangle
            self._draw_dashed_rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)

            # Smaller, semi-transparent label for motion
            label = "MOTION"
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale_small, self.font_thickness
            )
            padding = int(4 * self.scale_factor)

            # Semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y2 - label_h - baseline - padding * 2),
                         (x1 + label_w + padding * 2, y2), color, -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            cv2.putText(frame, label, (x1 + padding, y2 - padding),
                       cv2.FONT_HERSHEY_SIMPLEX, self.font_scale_small,
                       (0, 0, 0), self.font_thickness)

        return frame

    def _draw_dashed_rectangle(self, frame: np.ndarray, pt1: tuple, pt2: tuple,
                               color: tuple, thickness: int, dash_length: int = None):
        """Draw a dashed rectangle"""
        if dash_length is None:
            dash_length = max(10, int(15 * self.scale_factor))

        x1, y1 = pt1
        x2, y2 = pt2

        # Draw dashed lines for each side
        for side in [(x1, y1, x2, y1), (x2, y1, x2, y2),
                     (x2, y2, x1, y2), (x1, y2, x1, y1)]:
            x_start, y_start, x_end, y_end = side

            # Calculate line length
            length = int(np.sqrt((x_end - x_start)**2 + (y_end - y_start)**2))

            # Draw dashes
            for i in range(0, length, dash_length * 2):
                start_ratio = i / length
                end_ratio = min((i + dash_length) / length, 1.0)

                dash_x1 = int(x_start + (x_end - x_start) * start_ratio)
                dash_y1 = int(y_start + (y_end - y_start) * start_ratio)
                dash_x2 = int(x_start + (x_end - x_start) * end_ratio)
                dash_y2 = int(y_start + (y_end - y_start) * end_ratio)

                cv2.line(frame, (dash_x1, dash_y1), (dash_x2, dash_y2),
                        color, thickness)

    def _overlay_panel(self, canvas: np.ndarray, panel: np.ndarray, position: str = 'top-right'):
        """
        Overlay panel directly on canvas (no blending for speed)

        Args:
            canvas: Main canvas to overlay on
            panel: Panel to overlay
            position: Position of overlay ('top-right', 'bottom', 'top-left')
        """
        panel_h, panel_w = panel.shape[:2]
        canvas_h, canvas_w = canvas.shape[:2]

        # Calculate position
        if position == 'top-right':
            x = canvas_w - panel_w - int(10 * self.scale_factor)
            # Place metrics panel below the control buttons (button height ~56px + margin)
            y = int(70 * self.scale_factor)
        elif position == 'bottom':
            x = int(10 * self.scale_factor)
            y = canvas_h - panel_h - int(10 * self.scale_factor)
        elif position == 'top-left':
            x = int(10 * self.scale_factor)
            y = int(10 * self.scale_factor)
        else:
            x, y = 0, 0

        # Ensure panel fits within canvas
        if x + panel_w > canvas_w:
            x = canvas_w - panel_w
        if y + panel_h > canvas_h:
            y = canvas_h - panel_h
        if x < 0:
            x = 0
        if y < 0:
            y = 0

        # Alpha blend for transparency (75% panel, 25% video for better see-through)
        roi = canvas[y:y+panel_h, x:x+panel_w]
        cv2.addWeighted(panel, 0.75, roi, 0.25, 0, roi)
        canvas[y:y+panel_h, x:x+panel_w] = roi

    def _create_metrics_overlay(self, metrics: dict, detections: List[Detection]) -> np.ndarray:
        """Create simplified metrics overlay panel for low latency (scaled to 75%)"""
        gui_scale = self.scale_factor * 0.75
        panel_width = int(220 * gui_scale)

        # Count detection types
        detection_counts = {}
        for det in detections:
            detection_counts[det.class_name] = detection_counts.get(det.class_name, 0) + 1

        # Calculate height based on content (more compact)
        num_lines = 3  # FPS, Inference, Objects
        if detection_counts:
            num_lines += len(detection_counts)

        padding_top = int(18 * gui_scale)
        padding_bottom = int(12 * gui_scale)
        line_spacing = int(26 * gui_scale)
        panel_height = padding_top + (num_lines * line_spacing) + padding_bottom

        # Create panel with solid background (no gradient for speed)
        panel = np.full((panel_height, panel_width, 3), self.colors['panel_bg'], dtype=np.uint8)

        # Add simple top accent line
        accent_height = int(3 * gui_scale)
        panel[:accent_height, :] = self.colors['accent_cyan']

        y = padding_top + int(15 * gui_scale)
        spacing = line_spacing

        # FPS
        fps = metrics.get('fps', 0)
        fps_color = self.colors['success'] if fps > 20 else self.colors['warning']
        self._draw_compact_metric(panel, "FPS:", f"{fps:.1f}", y, fps_color)
        y += spacing

        # Inference time
        inf_time = metrics.get('inference_time_ms', 0)
        self._draw_compact_metric(panel, "Inference:", f"{inf_time:.1f}ms", y)
        y += spacing

        # Detection count
        total = len(detections)
        total_color = self.colors['critical'] if total > 0 else self.colors['text']
        self._draw_compact_metric(panel, "Objects:", str(total), y, total_color)
        y += spacing

        # Individual detection counts (if any)
        if detection_counts:
            for obj_type, count in sorted(detection_counts.items()):
                self._draw_compact_metric(panel, f"  {obj_type}:", str(count), y, self.colors['warning'])
                y += spacing

        return panel

    def _create_alert_overlay(self, alerts: List[Alert], max_width: int) -> np.ndarray:
        """Create simplified alert overlay panel for low latency (scaled to 75%)"""
        gui_scale = self.scale_factor * 0.75
        # Determine panel height based on number of alerts
        num_alerts = min(len(alerts), 2)  # Show max 2 alerts
        padding = int(18 * gui_scale)
        line_height = int(32 * gui_scale)
        panel_height = padding * 2 + (num_alerts * line_height)

        # Make panel width more reasonable
        panel_width = min(max_width - int(20 * gui_scale), int(800 * gui_scale))

        # Create panel with solid background (no gradient for speed)
        panel = np.full((panel_height, panel_width, 3), self.colors['panel_bg'], dtype=np.uint8)

        # Sort alerts by level (critical first)
        sorted_alerts = sorted(alerts, key=lambda a: a.level.value, reverse=True)

        y_offset = padding + int(20 * gui_scale)
        alert_spacing = line_height
        circle_radius = int(8 * gui_scale)
        circle_offset = int(15 * gui_scale)
        text_offset = int(35 * gui_scale)

        # Show top alerts with simplified styling
        for i, alert in enumerate(sorted_alerts[:2]):
            color = self._get_alert_color(alert.level)

            # Simple indicator circle with border (no glow for speed)
            cv2.circle(panel, (circle_offset, y_offset), circle_radius, color, -1)
            cv2.circle(panel, (circle_offset, y_offset), circle_radius, self.colors['text'], 1)

            # Alert message
            message = alert.message
            cv2.putText(panel, message, (text_offset, y_offset + int(5 * gui_scale)),
                       cv2.FONT_HERSHEY_SIMPLEX, self.font_scale_medium * 0.75,
                       color, max(1, int(self.font_thickness_bold * 0.75)))

            y_offset += alert_spacing

        # Show count if more alerts
        if len(sorted_alerts) > 2:
            count_text = f"+{len(sorted_alerts) - 2} more alerts"
            cv2.putText(panel, count_text,
                       (text_offset, y_offset + int(5 * gui_scale)),
                       cv2.FONT_HERSHEY_SIMPLEX, self.font_scale_small * 0.75,
                       self.colors['info'], max(1, int(self.font_thickness * 0.75)))

        return panel

    def _draw_compact_metric(self, panel: np.ndarray, label: str, value: str,
                            y: int, color: tuple = None):
        """Draw a compact metric line on panel (scaled to 75%)"""
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

    def _create_alert_panel(self, alerts: List[Alert], width: int) -> np.ndarray:
        """Create bottom alert panel"""
        panel = np.full((self.alert_panel_height, width, 3),
                       self.colors['panel_bg'], dtype=np.uint8)

        # Title
        title_y = int(30 * self.scale_factor)
        cv2.putText(panel, "ALERTS", (self.margin, title_y),
                   cv2.FONT_HERSHEY_SIMPLEX, self.font_scale_large,
                   self.colors['text'], self.font_thickness_bold)

        # Sort alerts by level (critical first)
        sorted_alerts = sorted(alerts, key=lambda a: a.level.value, reverse=True)

        # Display up to 3 most important alerts
        y_offset = int(60 * self.scale_factor)
        alert_spacing = int(35 * self.scale_factor)
        circle_radius = int(10 * self.scale_factor)
        circle_offset = int(15 * self.scale_factor)
        text_offset = int(40 * self.scale_factor)

        for i, alert in enumerate(sorted_alerts[:3]):
            color = self._get_alert_color(alert.level)

            # Alert level indicator (circle)
            cv2.circle(panel, (self.margin + circle_offset, y_offset), circle_radius, color, -1)

            # Alert message
            message = alert.message
            cv2.putText(panel, message, (self.margin + text_offset, y_offset + int(5 * self.scale_factor)),
                       cv2.FONT_HERSHEY_SIMPLEX, self.font_scale_medium, color, self.font_thickness_bold)

            y_offset += alert_spacing

        # If no alerts, show "ALL CLEAR"
        if len(alerts) == 0:
            clear_y = int(80 * self.scale_factor)
            cv2.putText(panel, "ALL CLEAR", (self.margin, clear_y),
                       cv2.FONT_HERSHEY_SIMPLEX, self.font_scale_large * 1.2,
                       self.colors['success'], self.font_thickness_bold)

        return panel

    def _create_metrics_panel(self, metrics: dict, detections: List[Detection],
                             height: int) -> np.ndarray:
        """Create right-side metrics panel"""
        panel = np.full((height, self.metrics_panel_width, 3),
                       self.colors['panel_bg'], dtype=np.uint8)

        y = int(40 * self.scale_factor)
        spacing = int(35 * self.scale_factor)

        # Title
        cv2.putText(panel, "METRICS", (self.margin, y),
                   cv2.FONT_HERSHEY_SIMPLEX, self.font_scale_large,
                   self.colors['text'], self.font_thickness_bold)
        y += spacing + int(10 * self.scale_factor)

        # FPS
        fps = metrics.get('fps', 0)
        fps_color = self.colors['success'] if fps > 20 else self.colors['warning']
        self._draw_metric(panel, "FPS:", f"{fps:.1f}", y, fps_color)
        y += spacing

        # Inference time
        inf_time = metrics.get('inference_time_ms', 0)
        self._draw_metric(panel, "Inference:", f"{inf_time:.1f}ms", y)
        y += spacing

        # GPU usage (if available)
        if 'gpu_usage' in metrics:
            gpu = metrics['gpu_usage']
            self._draw_metric(panel, "GPU:", f"{gpu:.0f}%", y)
            y += spacing

        # Draw separator
        y += int(10 * self.scale_factor)
        line_thickness = max(1, int(self.scale_factor))
        cv2.line(panel, (self.margin, y), (self.metrics_panel_width - self.margin, y),
                self.colors['text'], line_thickness)
        y += int(30 * self.scale_factor)

        # Detection counts
        cv2.putText(panel, "DETECTIONS", (self.margin, y),
                   cv2.FONT_HERSHEY_SIMPLEX, self.font_scale_medium,
                   self.colors['text'], self.font_thickness)
        y += int(30 * self.scale_factor)

        # Count by type
        detection_counts = {}
        for det in detections:
            detection_counts[det.class_name] = detection_counts.get(det.class_name, 0) + 1

        item_spacing = int(30 * self.scale_factor)
        if detection_counts:
            for obj_type, count in sorted(detection_counts.items()):
                self._draw_metric(panel, f"{obj_type}:", str(count), y, self.colors['info'])
                y += item_spacing
                if y > height - int(60 * self.scale_factor):  # Prevent overflow
                    break
        else:
            cv2.putText(panel, "None", (self.margin, y),
                       cv2.FONT_HERSHEY_SIMPLEX, self.font_scale_small,
                       self.colors['text'], self.font_thickness)

        # Total at bottom
        y = height - int(40 * self.scale_factor)
        total = len(detections)
        self._draw_metric(panel, "TOTAL:", str(total), y, self.colors['success'])

        return panel

    def _draw_metric(self, panel: np.ndarray, label: str, value: str,
                    y: int, color: tuple = None):
        """Draw a metric line on panel"""
        if color is None:
            color = self.colors['text']

        label_x_offset = int(130 * self.scale_factor)
        cv2.putText(panel, label, (self.margin, y),
                   cv2.FONT_HERSHEY_SIMPLEX, self.font_scale_small,
                   self.colors['text'], self.font_thickness)
        cv2.putText(panel, value, (self.margin + label_x_offset, y),
                   cv2.FONT_HERSHEY_SIMPLEX, self.font_scale_medium, color, self.font_thickness_bold)

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
        """
        Display frame and handle keyboard input

        Args:
            frame: Frame to display

        Returns:
            Key code pressed (or -1 if none)
        """
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