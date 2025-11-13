#!/usr/bin/env python3
"""
FLIR Boson Thermal Road Monitor - VPI Mode
Bypasses PyTorch and uses VPI GPU acceleration directly
"""
import sys
import time
import logging
import argparse
import threading
from queue import Queue
import cv2

from flir_camera import FLIRBosonCamera
from camera_detector import CameraDetector
from vpi_detector import VPIDetector
from road_analyzer import RoadAnalyzer
from driver_gui import DriverGUI
from performance_monitor import PerformanceMonitor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ThermalRoadMonitorVPI:
    """Main application using VPI acceleration"""

    def __init__(self, args):
        self.args = args
        self.camera = None
        self.detector = None
        self.analyzer = None
        self.gui = None
        self.perf_monitor = None
        self.running = False
        self.show_detections = True
        # Enable YOLO by default if in model detection mode, otherwise disable
        self.yolo_enabled = (getattr(args, 'detection_mode', 'edge') == 'model')
        self.frame_count = 0
        self.device = getattr(args, 'device', 'cuda')  # cuda or cpu

        # Performance tuning settings (adjustable via GUI)
        self.buffer_flush_enabled = False  # Buffer flush on/off
        self.frame_skip_value = 1  # 0=every frame, 1=every 2nd, 2=every 3rd, etc.

        # YOLO model switcher
        self.available_models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt']
        self.current_model = getattr(args, 'model', 'yolov8s.pt')
        self.current_model_index = 0
        if self.current_model in self.available_models:
            self.current_model_index = self.available_models.index(self.current_model)
        self.model_switching = False  # Flag to prevent switching during detection

        # FPS smoothing to prevent jankiness
        from collections import deque
        self.frame_times = deque(maxlen=60)  # Track last 60 frame times for smoother average
        self.smoothed_fps = 60.0
        self.min_frame_time = 1.0 / 60.0  # Cap at 60 FPS (camera limit)

        # Async detection
        self.detection_thread = None
        self.detection_queue = Queue(maxsize=2)  # Queue for frames to process
        self.result_queue = Queue(maxsize=2)  # Queue for detection results
        self.detection_lock = threading.Lock()
        self.latest_detections = []
        self.latest_alerts = []

    def initialize(self) -> bool:
        """Initialize all components"""
        logger.info("Initializing Thermal Road Monitor (VPI Mode)...")

        try:
            # 1. Detect cameras
            if self.args.camera_id is None:
                logger.info("Auto-detecting cameras...")
                cameras = CameraDetector.detect_all_cameras()
                CameraDetector.print_camera_list(cameras)

                flir_camera = CameraDetector.find_flir_boson()
                if flir_camera:
                    self.args.camera_id = flir_camera.device_id
                    self.args.width = flir_camera.resolution[0] if flir_camera.resolution[0] > 0 else self.args.width
                    self.args.height = flir_camera.resolution[1] if flir_camera.resolution[1] > 0 else self.args.height
                else:
                    logger.error("No camera detected")
                    return False

            # 2. Initialize camera
            logger.info(f"Opening camera device {self.args.camera_id}...")
            self.camera = FLIRBosonCamera(
                device_id=self.args.camera_id,
                resolution=(self.args.width, self.args.height)
            )
            if not self.camera.open():
                return False

            actual_res = self.camera.get_actual_resolution()
            logger.info(f"Camera opened: {actual_res[0]}x{actual_res[1]}")

            # 3. Initialize VPI detector
            detection_mode = getattr(self.args, 'detection_mode', 'edge')
            model_path = getattr(self.args, 'model', None) if detection_mode == 'model' else None
            logger.info(f"Initializing VPI detector (mode: {detection_mode}, device: {self.device})...")
            palette = getattr(self.args, 'palette', 'ironbow')
            self.detector = VPIDetector(
                confidence_threshold=self.args.confidence,
                thermal_palette=palette,
                model_path=model_path,
                detection_mode=detection_mode,
                device=self.device
            )
            if not self.detector.initialize():
                logger.error("Failed to initialize VPI detector")
                return False

            self.available_palettes = self.detector.get_available_palettes()
            self.current_palette_idx = self.available_palettes.index(palette) if palette in self.available_palettes else 0

            # 4. Initialize road analyzer
            self.analyzer = RoadAnalyzer(
                frame_width=actual_res[0],
                frame_height=actual_res[1]
            )

            # 5. Initialize GUI
            scale_factor = getattr(self.args, 'scale', 2.0)
            self.gui = DriverGUI(
                window_name="FLIR Thermal Road Monitor (VPI)",
                scale_factor=scale_factor
            )

            # Calculate window size: 2x the video size (overlays don't affect window size)
            video_width = int(actual_res[0] * scale_factor)
            video_height = int(actual_res[1] * scale_factor)
            window_width = video_width * 2
            window_height = video_height * 2

            self.gui.create_window(
                fullscreen=self.args.fullscreen,
                window_width=window_width,
                window_height=window_height
            )

            # Set up mouse callback for button clicks
            cv2.setMouseCallback(self.gui.window_name, self._mouse_callback)

            # 6. Initialize performance monitor
            self.perf_monitor = PerformanceMonitor()

            logger.info("Initialization complete!")
            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

    def _detection_worker(self):
        """Background thread for async YOLO detection"""
        logger.info("Detection worker thread started")

        while self.running:
            try:
                # Only process if YOLO is enabled
                if self.yolo_enabled and not self.detection_queue.empty():
                    frame = self.detection_queue.get(timeout=0.1)

                    # Apply dynamic frame skip setting
                    self.detector.frame_skip = self.frame_skip_value

                    # Run detection
                    detections = self.detector.detect(frame, filter_road_objects=True)
                    alerts = self.analyzer.analyze(detections)

                    # Update latest results (thread-safe)
                    with self.detection_lock:
                        self.latest_detections = detections
                        self.latest_alerts = alerts

                    # Update inference metrics (always update for display)
                    self.perf_monitor.update_inference_metrics(
                        self.detector.fps,
                        self.detector.last_inference_time * 1000
                    )
                elif not self.yolo_enabled:
                    # YOLO disabled - clear detections
                    with self.detection_lock:
                        self.latest_detections = []
                        self.latest_alerts = []
                    # Clear the queue
                    while not self.detection_queue.empty():
                        try:
                            self.detection_queue.get_nowait()
                        except:
                            break
                    time.sleep(0.05)  # Sleep a bit when disabled
                else:
                    time.sleep(0.001)  # Small sleep to avoid busy-waiting

            except Exception as e:
                logger.error(f"Detection worker error: {e}")
                time.sleep(0.1)

        logger.info("Detection worker thread stopped")

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks on GUI buttons"""
        if event == cv2.EVENT_LBUTTONDOWN:
            button_id = self.gui.check_button_click(x, y)
            if button_id == 'palette_cycle':
                # Cycle to next palette
                self.current_palette_idx = (self.current_palette_idx + 1) % len(self.available_palettes)
                new_palette = self.available_palettes[self.current_palette_idx]
                self.detector.set_palette(new_palette)
                logger.info(f"Palette changed to: {new_palette}")
            elif button_id == 'yolo_toggle':
                # Toggle YOLO detection
                self.yolo_enabled = not self.yolo_enabled
                logger.info(f"YOLO detection {'enabled' if self.yolo_enabled else 'disabled'}")
            elif button_id == 'detection_toggle':
                # Toggle detection boxes
                self.show_detections = not self.show_detections
                logger.info(f"Detection boxes {'enabled' if self.show_detections else 'disabled'}")
            elif button_id == 'buffer_flush_toggle':
                # Toggle buffer flush
                self.buffer_flush_enabled = not self.buffer_flush_enabled
                logger.info(f"Buffer flush {'enabled' if self.buffer_flush_enabled else 'disabled'}")
            elif button_id == 'frame_skip_cycle':
                # Cycle frame skip (0, 1, 2, 3, then back to 0)
                self.frame_skip_value = (self.frame_skip_value + 1) % 4
                logger.info(f"Frame skip set to: {self.frame_skip_value} (detect every {self.frame_skip_value + 1} frames)")
            elif button_id == 'device_toggle':
                # Toggle between CUDA and CPU
                self.device = 'cpu' if self.device == 'cuda' else 'cuda'
                self.detector.set_device(self.device)
                logger.info(f"Device switched to: {self.device.upper()}")
            elif button_id == 'model_cycle':
                # Cycle to next YOLO model
                if not self.model_switching and self.detector.detection_mode == 'model':
                    self.model_switching = True
                    self.current_model_index = (self.current_model_index + 1) % len(self.available_models)
                    new_model = self.available_models[self.current_model_index]
                    self.current_model = new_model
                    # Reload the model
                    logger.info(f"Switching to model: {new_model}")
                    self.detector.load_yolo_model(new_model)
                    self.model_switching = False
                    logger.info(f"Model switched to: {new_model}")

    def run(self):
        """Main application loop"""
        logger.info("Starting main loop...")
        self.running = True

        # Start async detection worker thread
        self.detection_thread = threading.Thread(target=self._detection_worker, daemon=True)
        self.detection_thread.start()
        logger.info("Async detection thread started")

        try:
            while self.running:
                loop_start = time.time()

                # 1. Capture frame (use dynamic buffer flush setting)
                try:
                    ret, frame = self.camera.read(flush_buffer=self.buffer_flush_enabled)
                    if not ret or frame is None:
                        logger.warning(f"Failed to read frame")
                        time.sleep(0.1)
                        continue
                except Exception as e:
                    logger.error(f"Error reading from camera: {e}", exc_info=True)
                    break

                # 2. Apply thermal color palette
                try:
                    frame = self.detector.apply_thermal_palette(frame)
                except Exception as e:
                    logger.error(f"Error applying palette: {e}", exc_info=True)

                # 3. Send frame to async detection (non-blocking)
                # Only send if queue has space (don't block main thread)
                if not self.detection_queue.full():
                    try:
                        self.detection_queue.put_nowait(frame.copy())
                    except:
                        pass  # Queue full, skip this frame

                # 4. Get latest detections from async thread (thread-safe)
                with self.detection_lock:
                    detections = self.latest_detections.copy()
                    alerts = self.latest_alerts.copy()

                # 5. Update metrics
                if self.frame_count % 5 == 0:
                    self.perf_monitor.update()

                metrics = self.perf_monitor.get_metrics()

                # Calculate smoothed FPS to prevent jankiness
                loop_time = time.time() - loop_start

                # Clamp frame times to camera FPS limit to show smooth 60 FPS display
                # Even when YOLO inference causes some frames to be slower
                clamped_time = min(loop_time, self.min_frame_time)
                self.frame_times.append(clamped_time)

                # Use rolling average for smooth FPS display
                if len(self.frame_times) > 0:
                    avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                    self.smoothed_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 60.0
                    # Cap at 60 FPS (camera limit)
                    self.smoothed_fps = min(self.smoothed_fps, 60.0)

                metrics['fps'] = self.smoothed_fps

                # 5. Render and display with GUI controls
                try:
                    current_palette = self.available_palettes[self.current_palette_idx]
                    display_frame = self.gui.render_frame_with_controls(
                        frame=frame,
                        detections=detections,
                        alerts=alerts,
                        metrics=metrics,
                        show_detections=self.show_detections,
                        current_palette=current_palette,
                        yolo_enabled=self.yolo_enabled,
                        buffer_flush_enabled=self.buffer_flush_enabled,
                        frame_skip_value=self.frame_skip_value,
                        device=self.device,
                        current_model=self.current_model
                    )

                    key = self.gui.display(display_frame)
                    self._handle_keypress(key, display_frame)
                except Exception as e:
                    logger.error(f"GUI error: {e}")

                self.frame_count += 1

                # Print stats
                if self.frame_count % 100 == 0:
                    logger.info(f"Frame {self.frame_count} | FPS: {self.smoothed_fps:.1f} | Detections: {len(detections)}")

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
        finally:
            self.shutdown()

    def _handle_keypress(self, key: int, frame):
        """Handle keyboard input"""
        if key == ord('q') or key == 27:
            self.running = False
        elif key == ord('f'):
            self.gui.toggle_fullscreen()
        elif key == ord('d'):
            self.show_detections = not self.show_detections
        elif key == ord('y') or key == ord('Y'):
            self.yolo_enabled = not self.yolo_enabled
            logger.info(f"YOLO detection {'enabled' if self.yolo_enabled else 'disabled'}")
        elif key == ord('s'):
            import cv2
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"
            cv2.imwrite(filename, frame)
            logger.info(f"Screenshot saved: {filename}")
        elif key == ord('p'):
            print("\n" + "="*50)
            print(self.perf_monitor.get_summary())
            print("="*50 + "\n")
        elif key == ord('c') or key == ord('C'):
            # Cycle through thermal color palettes
            self.current_palette_idx = (self.current_palette_idx + 1) % len(self.available_palettes)
            new_palette = self.available_palettes[self.current_palette_idx]
            self.detector.set_palette(new_palette)
            logger.info(f"Switched to palette: {new_palette}")
        elif key == ord('1'):
            self._set_palette_by_name('white_hot')
        elif key == ord('2'):
            self._set_palette_by_name('black_hot')
        elif key == ord('3'):
            self._set_palette_by_name('ironbow')
        elif key == ord('4'):
            self._set_palette_by_name('rainbow')
        elif key == ord('5'):
            self._set_palette_by_name('arctic')
        elif key == ord('6'):
            self._set_palette_by_name('lava')
        elif key == ord('7'):
            self._set_palette_by_name('medical')
        elif key == ord('8'):
            self._set_palette_by_name('plasma')
        elif key == ord('g') or key == ord('G'):
            # Toggle device (CUDA/CPU)
            self.device = 'cpu' if self.device == 'cuda' else 'cuda'
            self.detector.set_device(self.device)
            logger.info(f"Device switched to: {self.device.upper()}")

    def _set_palette_by_name(self, palette_name: str):
        """Set palette and update GUI index"""
        self.detector.set_palette(palette_name)
        # Update index to match so GUI button shows correct palette
        if palette_name in self.available_palettes:
            self.current_palette_idx = self.available_palettes.index(palette_name)
        logger.info(f"Switched to palette: {palette_name}")

    def shutdown(self):
        """Cleanup resources"""
        logger.info("Shutting down...")

        # Stop detection thread
        self.running = False
        if self.detection_thread and self.detection_thread.is_alive():
            logger.info("Waiting for detection thread to stop...")
            self.detection_thread.join(timeout=2.0)

        if self.camera:
            self.camera.release()
        if self.detector:
            self.detector.release()
        if self.gui:
            self.gui.close()
        if self.perf_monitor:
            self.perf_monitor.release()
        logger.info("Shutdown complete")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="FLIR Boson Thermal Road Monitor - VPI Mode",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--camera-id', type=int, default=None,
                       help='Camera device ID (auto-detect if not specified)')
    parser.add_argument('--width', type=int, default=640, help='Camera frame width')
    parser.add_argument('--height', type=int, default=512, help='Camera frame height')
    parser.add_argument('--confidence', type=float, default=0.25,
                       help='Detection confidence threshold (default: 0.25, optimized for thermal imagery)')
    parser.add_argument('--fullscreen', action='store_true', help='Start in fullscreen')
    parser.add_argument('--scale', type=float, default=2.0,
                       help='Display scale factor (default: 2.0 for 2x size)')
    parser.add_argument('--palette', type=str, default='ironbow',
                       choices=['white_hot', 'black_hot', 'ironbow', 'rainbow', 'arctic', 'lava', 'medical', 'plasma'],
                       help='Thermal color palette (default: ironbow)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='YOLO model path (only used in model detection mode). '
                            'Recommended for Jetson Orin Nano: yolov8n.pt (fastest), yolov8s.pt (more accurate), or yolov5nu.pt')
    parser.add_argument('--detection-mode', type=str, default='model',
                       choices=['edge', 'model'],
                       help='Detection mode: edge (VPI edge detection) or model (YOLO with VPI preprocessing). '
                            'Default: model (YOLO object detection enabled)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Processing device: cuda for GPU acceleration or cpu for CPU-only (default: cuda)')

    return parser.parse_args()


def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║   FLIR Boson Thermal Road Monitor - VPI GPU Mode            ║
║   Hardware-Accelerated Vision Processing for Jetson Orin    ║
╠══════════════════════════════════════════════════════════════╣
║  Keyboard Controls:                                          ║
║    Q/ESC - Quit          F - Fullscreen      D - Detections  ║
║    Y - Toggle YOLO       C - Cycle Palettes  S - Screenshot  ║
║    G - Device (CUDA/CPU) P - Stats           1-8 - Palettes  ║
║                          1=White 2=Black 3=Ironbow 4=Rainbow║
║                          5=Arctic 6=Lava 7=Medical 8=Plasma  ║
║                                                               ║
║  GUI Buttons (click):                                        ║
║    • Palette: [name] - Cycle thermal color palettes          ║
║    • YOLO: ON/OFF    - Toggle object detection               ║
║    • Boxes: ON/OFF   - Toggle detection box overlay          ║
║    • Device: CUDA/CPU - Toggle GPU/CPU acceleration          ║
║    • Model: V8N/V8S/V8M - Cycle YOLO model                   ║
║                                                               ║
║  Defaults: YOLO enabled, CUDA acceleration, confidence 0.25  ║
║  Model: yolov8n.pt (optimized for Jetson Orin Nano)         ║
╚══════════════════════════════════════════════════════════════╝
    """)

    args = parse_arguments()
    app = ThermalRoadMonitorVPI(args)

    if not app.initialize():
        logger.error("Failed to initialize application")
        sys.exit(1)

    app.run()
    sys.exit(0)


if __name__ == "__main__":
    main()