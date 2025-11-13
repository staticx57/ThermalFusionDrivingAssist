#!/usr/bin/env python3
"""
FLIR Boson Thermal Road Monitor
Real-time object detection and driver alert system
Optimized for NVIDIA Jetson Orin with GPU acceleration

Controls:
  Q or ESC: Quit application
  F: Toggle fullscreen
  D: Toggle detection visualization
  S: Save screenshot
  P: Print performance summary
"""
import sys
import time
import logging
import argparse
from pathlib import Path

from flir_camera import FLIRBosonCamera
from camera_detector import CameraDetector, CameraInfo
from object_detector import ObjectDetector
from vpi_detector import VPIDetector
from road_analyzer import RoadAnalyzer
from driver_gui import DriverGUI
from performance_monitor import PerformanceMonitor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ThermalRoadMonitor:
    """Main application class"""

    def __init__(self, args):
        """Initialize application"""
        self.args = args
        self.camera = None
        self.detector = None
        self.analyzer = None
        self.gui = None
        self.perf_monitor = None
        self.running = False
        self.show_detections = True
        self.frame_count = 0

    def initialize(self) -> bool:
        """Initialize all components"""
        logger.info("Initializing Thermal Road Monitor...")

        try:
            # 1. Detect cameras
            if self.args.camera_id is None:
                logger.info("Auto-detecting cameras...")
                cameras = CameraDetector.detect_all_cameras()
                CameraDetector.print_camera_list(cameras)

                # Find FLIR Boson
                flir_camera = CameraDetector.find_flir_boson()
                if flir_camera:
                    self.args.camera_id = flir_camera.device_id
                    self.args.width = flir_camera.resolution[0] if flir_camera.resolution[0] > 0 else self.args.width
                    self.args.height = flir_camera.resolution[1] if flir_camera.resolution[1] > 0 else self.args.height
                    logger.info(f"Selected FLIR Boson on device {self.args.camera_id}")
                else:
                    logger.error("No camera detected. Please connect a camera.")
                    return False

            # 2. Initialize camera
            logger.info(f"Opening camera device {self.args.camera_id}...")
            self.camera = FLIRBosonCamera(
                device_id=self.args.camera_id,
                resolution=(self.args.width, self.args.height)
            )
            if not self.camera.open():
                logger.error("Failed to open camera")
                return False

            actual_res = self.camera.get_actual_resolution()
            logger.info(f"Camera opened: {actual_res[0]}x{actual_res[1]}")

            # 2. Initialize object detector with GPU acceleration
            logger.info("Initializing GPU-accelerated object detector...")

            # Try PyTorch detector first
            use_vpi = False
            self.detector = ObjectDetector(
                model_path=self.args.model,
                confidence_threshold=self.args.confidence,
                use_tensorrt=self.args.use_tensorrt,
                device=str(self.args.gpu_id)
            )

            if not self.detector.initialize():
                logger.warning("PyTorch detector failed, trying VPI accelerated detector...")
                # Fallback to VPI detector
                self.detector = VPIDetector(confidence_threshold=self.args.confidence)
                if self.detector.initialize():
                    logger.info("Using VPI GPU-accelerated detector")
                    use_vpi = True
                else:
                    logger.error("Both PyTorch and VPI detectors failed")
                    return False

            # 3. Initialize road analyzer
            logger.info("Initializing road analyzer...")
            self.analyzer = RoadAnalyzer(
                frame_width=actual_res[0],
                frame_height=actual_res[1]
            )

            # 4. Initialize GUI
            logger.info("Initializing driver GUI...")
            self.gui = DriverGUI(window_name="FLIR Thermal Road Monitor")
            self.gui.create_window(fullscreen=self.args.fullscreen)

            # 5. Initialize performance monitor
            logger.info("Initializing performance monitor...")
            self.perf_monitor = PerformanceMonitor()

            logger.info("Initialization complete!")
            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

    def run(self):
        """Main application loop"""
        logger.info("Starting main loop...")
        self.running = True

        # Performance optimization variables
        skip_detection_frames = 0  # Skip detection every N frames for speed
        last_detections = []
        last_alerts = []
        frame_skip_counter = 0

        try:
            while self.running:
                loop_start = time.time()

                # 1. Capture frame from thermal camera
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    logger.warning("Failed to read frame from camera")
                    time.sleep(0.1)
                    continue

                # 2. Detect objects (skip frames for performance if needed)
                if frame_skip_counter == 0:
                    try:
                        detections = self.detector.detect(frame, filter_road_objects=True)
                        last_detections = detections

                        # 3. Analyze detections and generate alerts
                        alerts = self.analyzer.analyze(detections)
                        last_alerts = alerts
                    except Exception as e:
                        logger.error(f"Detection/Analysis error: {e}")
                        detections = last_detections
                        alerts = last_alerts
                else:
                    # Reuse previous detections for smoother display
                    detections = last_detections
                    alerts = last_alerts

                frame_skip_counter = (frame_skip_counter + 1) % (skip_detection_frames + 1)

                # 4. Update performance metrics (less frequently)
                if self.frame_count % 5 == 0:  # Update every 5 frames
                    self.perf_monitor.update()

                self.perf_monitor.update_inference_metrics(
                    self.detector.fps,
                    self.detector.last_inference_time * 1000
                )
                metrics = self.perf_monitor.get_metrics()

                # Calculate actual FPS from loop time
                loop_time = time.time() - loop_start
                actual_fps = 1.0 / loop_time if loop_time > 0 else 0
                metrics['fps'] = actual_fps

                # 5. Render GUI
                try:
                    display_frame = self.gui.render_frame(
                        frame=frame,
                        detections=detections,
                        alerts=alerts,
                        metrics=metrics,
                        show_detections=self.show_detections
                    )

                    # 6. Display and handle input (non-blocking)
                    key = self.gui.display(display_frame)
                    self._handle_keypress(key, display_frame)
                except Exception as e:
                    logger.error(f"GUI error: {e}")

                # Frame timing
                self.frame_count += 1

                # Optional: Print stats periodically
                if self.frame_count % 100 == 0:
                    stats = self.analyzer.get_statistics()
                    logger.info(f"Frame {self.frame_count} | "
                              f"FPS: {actual_fps:.1f} | "
                              f"Detections: {len(detections)} | "
                              f"Alerts: {stats['active_alerts']}")

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
        finally:
            self.shutdown()

    def _handle_keypress(self, key: int, frame):
        """Handle keyboard input"""
        if key == ord('q') or key == 27:  # Q or ESC
            logger.info("Quit requested")
            self.running = False

        elif key == ord('f'):  # F - Toggle fullscreen
            self.gui.toggle_fullscreen()
            logger.info("Toggled fullscreen")

        elif key == ord('d'):  # D - Toggle detection visualization
            self.show_detections = not self.show_detections
            logger.info(f"Detection visualization: {self.show_detections}")

        elif key == ord('s'):  # S - Save screenshot
            self._save_screenshot(frame)

        elif key == ord('p'):  # P - Print performance summary
            print("\n" + "="*50)
            print(self.perf_monitor.get_summary())
            stats = self.analyzer.get_statistics()
            print(f"\nDetection Statistics:")
            print(f"  Current detections: {stats['current_detections']}")
            print(f"  Active alerts: {stats['active_alerts']}")
            print(f"  Total frames: {self.frame_count}")
            print("="*50 + "\n")

    def _save_screenshot(self, frame):
        """Save screenshot with timestamp"""
        import cv2
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        cv2.imwrite(filename, frame)
        logger.info(f"Screenshot saved: {filename}")

    def shutdown(self):
        """Cleanup resources"""
        logger.info("Shutting down...")

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
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="FLIR Boson Thermal Road Monitor - Real-time object detection for drivers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Camera settings
    parser.add_argument('--camera-id', type=int, default=None,
                       help='Camera device ID (e.g., 0 for /dev/video0). Auto-detect if not specified.')
    parser.add_argument('--width', type=int, default=640,
                       help='Camera frame width')
    parser.add_argument('--height', type=int, default=512,
                       help='Camera frame height')

    # Detection settings
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='YOLO model path (n=nano, s=small, m=medium)')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Detection confidence threshold (0.0-1.0)')
    parser.add_argument('--use-tensorrt', action='store_true', default=True,
                       help='Use TensorRT optimization (recommended for Jetson)')
    parser.add_argument('--no-tensorrt', action='store_false', dest='use_tensorrt',
                       help='Disable TensorRT optimization')
    parser.add_argument('--gpu-id', type=int, default=0,
                       help='GPU device ID')

    # Display settings
    parser.add_argument('--fullscreen', action='store_true',
                       help='Start in fullscreen mode')

    return parser.parse_args()


def main():
    """Main entry point"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║        FLIR Boson Thermal Road Monitor                       ║
║        GPU-Accelerated Object Detection for Jetson Orin      ║
╚══════════════════════════════════════════════════════════════╝
    """)

    args = parse_arguments()

    # Create and run application
    app = ThermalRoadMonitor(args)

    if not app.initialize():
        logger.error("Failed to initialize application")
        sys.exit(1)

    app.run()
    sys.exit(0)


if __name__ == "__main__":
    main()