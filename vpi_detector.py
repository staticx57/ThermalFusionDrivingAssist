"""
VPI-Accelerated Object Detection for Jetson
Uses NVIDIA Vision Programming Interface for GPU-accelerated preprocessing
Combined with lightweight CPU inference for real-time performance
"""
import cv2
import numpy as np
import time
from typing import List, Tuple, Optional
import logging

try:
    import vpi
    VPI_AVAILABLE = True
except ImportError:
    VPI_AVAILABLE = False
    logging.warning("VPI not available")

from object_detector import Detection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VPIDetector:
    """
    Hardware-accelerated detector using VPI + lightweight model
    Optimized for Jetson Orin when PyTorch CUDA is not working
    """

    def __init__(self, confidence_threshold: float = 0.5, thermal_palette: str = "white_hot",
                 model_path: Optional[str] = None, detection_mode: str = "edge", device: str = "cuda"):
        """
        Initialize VPI detector

        Args:
            confidence_threshold: Minimum confidence for detections
            thermal_palette: Color palette for thermal display
                           Options: white_hot, black_hot, ironbow, rainbow, arctic, lava, medical
            model_path: Path to YOLO model (used only in 'model' mode)
            detection_mode: 'edge' for edge detection or 'model' for YOLO detection
            device: 'cuda' for GPU acceleration or 'cpu' for CPU-only processing
        """
        self.confidence_threshold = confidence_threshold
        self.is_initialized = False
        self.fps = 0
        self.last_inference_time = 0
        self.thermal_palette = thermal_palette
        self.detection_mode = detection_mode
        self.model_path = model_path
        self.model = None
        self.device = device

        # VPI backend - use CUDA for GPU, CPU for CPU-only mode
        self.vpi_backend = vpi.Backend.CUDA if device == 'cuda' else vpi.Backend.CPU

        # Simple motion/heat detection instead of full object detection
        self.use_simple_detection = (detection_mode == "edge")

        # Performance optimization for CPU-based YOLO
        self.frame_skip = 1  # Run detection every 2nd frame (0, 2, 4, ...) for smoother video
        self.frame_count = 0
        self.last_detections = []
        self.yolo_input_size = 416  # Balanced size for thermal imaging (faster than 640, more accurate than 320)

        # Motion detection for road safety
        self.prev_frame = None
        self.motion_threshold = 20  # Pixel difference threshold (lowered for better sensitivity)
        self.min_motion_area = 400  # Minimum area in pixels (lowered to detect smaller objects)
        self.max_motion_area_ratio = 0.6  # Max 60% of frame (increased to allow more camera pan)

        # Temporal filtering - track motion persistence
        self.motion_history = {}  # Track motion detections across frames
        self.motion_persistence_frames = 2  # Motion must persist for 2 frames (faster response)

        # Thermal color palettes (lookup tables)
        self.color_palettes = self._create_color_palettes()

    def _create_color_palettes(self) -> dict:
        """Create thermal color palette lookup tables"""
        palettes = {}

        # White Hot (default - already provided by camera)
        white_hot = np.arange(256, dtype=np.uint8)
        palettes['white_hot'] = cv2.applyColorMap(white_hot.reshape(1, -1), cv2.COLORMAP_BONE)

        # Black Hot (inverted)
        black_hot = np.arange(255, -1, -1, dtype=np.uint8)
        palettes['black_hot'] = cv2.applyColorMap(black_hot.reshape(1, -1), cv2.COLORMAP_BONE)

        # Ironbow (classic thermal imaging palette)
        palettes['ironbow'] = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(1, -1), cv2.COLORMAP_HOT)

        # Rainbow
        palettes['rainbow'] = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(1, -1), cv2.COLORMAP_JET)

        # Arctic (blue-white)
        palettes['arctic'] = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(1, -1), cv2.COLORMAP_WINTER)

        # Lava (red-yellow)
        palettes['lava'] = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(1, -1), cv2.COLORMAP_INFERNO)

        # Medical (green-based)
        palettes['medical'] = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(1, -1), cv2.COLORMAP_VIRIDIS)

        # Plasma (purple-yellow)
        palettes['plasma'] = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(1, -1), cv2.COLORMAP_PLASMA)

        return palettes

    def set_palette(self, palette_name: str):
        """Change thermal color palette"""
        if palette_name in self.color_palettes:
            self.thermal_palette = palette_name
            logger.info(f"Switched to {palette_name} palette")
        else:
            logger.warning(f"Unknown palette: {palette_name}")

    def get_available_palettes(self) -> list:
        """Get list of available color palettes"""
        return list(self.color_palettes.keys())

    def set_device(self, device: str):
        """
        Change processing device (cuda/cpu) dynamically

        Args:
            device: 'cuda' for GPU acceleration or 'cpu' for CPU-only
        """
        if device not in ['cuda', 'cpu']:
            logger.warning(f"Invalid device: {device}. Must be 'cuda' or 'cpu'")
            return

        if device == self.device:
            logger.info(f"Device already set to {device}")
            return

        old_device = self.device
        self.device = device

        # Update VPI backend
        if VPI_AVAILABLE:
            self.vpi_backend = vpi.Backend.CUDA if device == 'cuda' else vpi.Backend.CPU

        # If model is loaded, move it to new device
        if self.model is not None:
            try:
                import torch
                yolo_device = 'cuda' if device == 'cuda' else 'cpu'
                # YOLO models handle device switching via the device parameter in inference
                # No need to reload the model, just update the device for next inference
                logger.info(f"Device switched from {old_device} to {device}")
                logger.info(f"VPI backend: {self.vpi_backend}, YOLO device: {yolo_device}")
            except Exception as e:
                logger.error(f"Error switching device: {e}")
                self.device = old_device  # Revert on error
        else:
            logger.info(f"Device switched from {old_device} to {device}")
            logger.info(f"VPI backend: {self.vpi_backend}")

    def load_yolo_model(self, model_path: str):
        """
        Load or reload a YOLO model

        Args:
            model_path: Path to YOLO model file (.pt)
        """
        if self.detection_mode != 'model':
            logger.warning("Cannot load YOLO model in edge detection mode")
            return

        try:
            from ultralytics import YOLO
            logger.info(f"Loading YOLO model: {model_path}")
            self.model = YOLO(model_path)
            logger.info(f"YOLO model loaded successfully: {model_path}")
            # Clear cached detections when model changes
            self.last_detections = []
        except Exception as e:
            logger.error(f"Failed to load YOLO model {model_path}: {e}")
            # Keep the old model if loading fails

    def _get_colormap_id(self, palette_name: str) -> int:
        """Map palette name to OpenCV colormap ID"""
        colormap_mapping = {
            'white_hot': cv2.COLORMAP_BONE,
            'black_hot': cv2.COLORMAP_BONE,  # Will invert separately
            'ironbow': cv2.COLORMAP_HOT,
            'rainbow': cv2.COLORMAP_JET,
            'arctic': cv2.COLORMAP_WINTER,
            'lava': cv2.COLORMAP_INFERNO,
            'medical': cv2.COLORMAP_VIRIDIS,
            'plasma': cv2.COLORMAP_PLASMA
        }
        return colormap_mapping.get(palette_name, cv2.COLORMAP_HOT)

    def apply_thermal_palette(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply thermal color palette to grayscale frame (CPU-optimized for low latency)

        Args:
            frame: Grayscale or BGR thermal frame

        Returns:
            Colorized thermal frame
        """
        try:
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            # Apply palette directly on CPU (faster than VPI conversion overhead)
            # cv2.applyColorMap is highly optimized and fast enough
            if self.thermal_palette in self.color_palettes:
                # Invert for black_hot
                if self.thermal_palette == 'black_hot':
                    gray = 255 - gray
                return cv2.applyColorMap(gray, self._get_colormap_id(self.thermal_palette))
            else:
                return cv2.applyColorMap(gray, cv2.COLORMAP_HOT)

        except Exception as e:
            logger.error(f"Error applying palette: {e}")
            # Return original frame on error
            if len(frame.shape) == 2:
                return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            return frame

    def initialize(self) -> bool:
        """Initialize VPI detector"""
        try:
            if not VPI_AVAILABLE:
                logger.error("VPI not available")
                return False

            logger.info(f"VPI version: {vpi.__version__}")
            logger.info(f"Initializing VPI-accelerated detector (mode: {self.detection_mode})...")

            # Initialize YOLO model if in model mode
            if self.detection_mode == "model":
                if not self.model_path:
                    logger.error("Model path required for model detection mode")
                    return False

                try:
                    from ultralytics import YOLO
                    import torch
                    logger.info(f"Loading YOLO model: {self.model_path}")
                    logger.info(f"CUDA available: {torch.cuda.is_available()}")
                    if torch.cuda.is_available():
                        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

                    self.model = YOLO(self.model_path)
                    logger.info("YOLO model loaded successfully")
                except ImportError:
                    logger.error("ultralytics package not installed. Run: pip install ultralytics")
                    return False
                except Exception as e:
                    logger.error(f"Failed to load YOLO model: {e}")
                    return False

            # Try different backends based on device selection
            if self.device == 'cuda':
                # GPU acceleration - try CUDA, PVA, VIC, then CPU fallback
                backends_to_try = [
                    (vpi.Backend.CUDA, "CUDA"),
                    (vpi.Backend.PVA, "PVA"),
                    (vpi.Backend.VIC, "VIC"),
                    (vpi.Backend.CPU, "CPU")
                ]
            else:
                # CPU-only mode
                backends_to_try = [
                    (vpi.Backend.CPU, "CPU")
                ]

            test_img = np.zeros((100, 100, 3), dtype=np.uint8)

            for backend, name in backends_to_try:
                try:
                    logger.info(f"Trying VPI backend: {name}")
                    with backend:
                        test_vpi = vpi.asimage(test_img, vpi.Format.BGR8)
                        test_gray = test_vpi.convert(vpi.Format.U8)

                    logger.info(f"VPI {name} backend initialized successfully!")
                    self.vpi_backend = backend
                    self.is_initialized = True
                    return True
                except Exception as e:
                    logger.warning(f"VPI {name} backend failed: {e}")
                    continue

            logger.error("All VPI backends failed")
            return False

        except Exception as e:
            logger.error(f"Failed to initialize VPI detector: {e}")
            return False

    def _detect_with_model(self, frame: np.ndarray, filter_road_objects: bool = True) -> List[Detection]:
        """
        Detect objects using YOLO model with VPI-accelerated preprocessing

        Args:
            frame: Input image (BGR format)
            filter_road_objects: Filter to only road-relevant objects

        Returns:
            List of Detection objects
        """
        if not self.model:
            logger.error("Model not loaded")
            return []

        # Frame skipping for performance - only run detection every Nth frame
        self.frame_count += 1
        if self.frame_count % (self.frame_skip + 1) != 0:
            # Return cached detections from last inference
            return self.last_detections

        start_time = time.time()
        detections = []

        try:
            # Run YOLO inference with selected device
            # imgsz=416 balances speed and accuracy for thermal imagery
            # conf threshold lowered for thermal detection (YOLO trained on RGB)
            yolo_device = 'cuda' if self.device == 'cuda' else 'cpu'
            results = self.model(frame, verbose=False, device=yolo_device,
                               imgsz=self.yolo_input_size, conf=self.confidence_threshold)[0]

            # Road-relevant classes (COCO dataset)
            road_classes = {
                0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
                5: 'bus', 7: 'truck', 9: 'traffic light', 11: 'stop sign'
            }

            # Debug: log all detections before filtering
            all_detections = len(results.boxes) if results.boxes is not None else 0

            for box in results.boxes:
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = results.names.get(class_id, f'class_{class_id}')

                # Log detections for debugging
                if conf >= self.confidence_threshold * 0.5:  # Log at half threshold
                    logger.debug(f"Detection: {class_name} conf={conf:.2f} threshold={self.confidence_threshold}")

                if conf < self.confidence_threshold:
                    continue

                # Filter road objects if requested
                if filter_road_objects and class_id not in road_classes:
                    continue

                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                class_name = road_classes.get(class_id, class_name)

                detection = Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    class_id=class_id,
                    class_name=class_name
                )
                detections.append(detection)

            if all_detections > 0 and len(detections) == 0:
                logger.debug(f"Found {all_detections} detections but none passed filters")

            self.last_inference_time = time.time() - start_time
            self.fps = 1.0 / self.last_inference_time if self.last_inference_time > 0 else 0

            # Log inference performance periodically
            import random
            if random.random() < 0.02:  # Log ~2% of frames
                logger.info(f"YOLO inference: {self.last_inference_time*1000:.1f}ms ({self.fps:.1f} FPS), {len(detections)} detections")

            # Cache detections for frame skipping
            self.last_detections = detections
            return detections

        except Exception as e:
            logger.error(f"Model detection error: {e}")
            # Update timing metrics even on error so GUI shows inference is running
            self.last_inference_time = time.time() - start_time
            self.fps = 1.0 / self.last_inference_time if self.last_inference_time > 0 else 0
            # Return cached detections on error
            return self.last_detections

    def _detect_with_edges(self, frame: np.ndarray, filter_road_objects: bool = True) -> List[Detection]:
        """
        Detect objects using VPI edge detection and contours

        Args:
            frame: Input image (BGR format)
            filter_road_objects: Not used in edge mode

        Returns:
            List of Detection objects
        """
        start_time = time.time()
        detections = []

        try:
            # Use VPI for hardware-accelerated image processing
            with self.vpi_backend:
                # Convert to VPI image
                vpi_img = vpi.asimage(frame, vpi.Format.BGR8)

                # Convert to grayscale using hardware acceleration
                gray = vpi_img.convert(vpi.Format.U8)

                # Apply Gaussian blur for noise reduction
                blurred = gray.gaussian_filter(7, sigma=1.5, border=vpi.Border.CLAMP)

                # Edge detection (hardware-accelerated)
                edges = blurred.canny(thresh_weak=50, thresh_strong=150)

                # Convert back to numpy for contour detection
                edges_np = edges.cpu()

            # Find contours (CPU - fast enough for this operation)
            contours, _ = cv2.findContours(edges_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter and create detections based on contour size and shape
            frame_h, frame_w = frame.shape[:2]
            min_area = (frame_w * frame_h) * 0.001  # 0.1% of frame
            max_area = (frame_w * frame_h) * 0.3    # 30% of frame

            for contour in contours:
                area = cv2.contourArea(contour)

                if area < min_area or area > max_area:
                    continue

                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)

                # Calculate confidence based on contour properties
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue

                circularity = 4 * np.pi * area / (perimeter * perimeter)
                aspect_ratio = float(w) / h if h > 0 else 0

                # Confidence heuristic (better for rectangular objects)
                confidence = min(1.0, (1.0 - circularity) * 0.5 + min(aspect_ratio, 1/aspect_ratio) * 0.5)

                if confidence < self.confidence_threshold:
                    continue

                # Classify based on size and aspect ratio
                if aspect_ratio > 1.5 or aspect_ratio < 0.5:
                    class_name = "vehicle" if w > 100 else "bicycle"
                elif aspect_ratio > 0.7 and aspect_ratio < 1.3:
                    class_name = "person" if h > w * 1.2 else "object"
                else:
                    class_name = "object"

                detection = Detection(
                    bbox=(x, y, x + w, y + h),
                    confidence=confidence,
                    class_id=0,
                    class_name=class_name
                )
                detections.append(detection)

            # Calculate FPS
            self.last_inference_time = time.time() - start_time
            self.fps = 1.0 / self.last_inference_time if self.last_inference_time > 0 else 0

            return detections

        except Exception as e:
            logger.error(f"Edge detection error: {e}")
            return []

    def _detect_motion(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect motion using temporal filtering and size constraints
        Detects objects across entire frame (including roadside deer)
        Filters out camera motion by rejecting large uniform motion and requiring persistence

        Args:
            frame: Input image (BGR format)

        Returns:
            List of Detection objects for persistent motion areas
        """
        detections = []

        try:
            frame_h, frame_w = frame.shape[:2]
            frame_area = frame_h * frame_w

            # Convert to grayscale for motion detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur to reduce noise
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            # Initialize previous frame if needed
            if self.prev_frame is None:
                self.prev_frame = gray
                return []

            # Compute absolute difference between current and previous frame
            frame_diff = cv2.absdiff(self.prev_frame, gray)

            # Threshold to get binary image (higher threshold reduces false positives)
            _, thresh = cv2.threshold(frame_diff, self.motion_threshold, 255, cv2.THRESH_BINARY)

            # Dilate to fill in holes
            thresh = cv2.dilate(thresh, None, iterations=2)

            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Calculate total motion area to detect widespread camera motion
            total_motion_area = sum(cv2.contourArea(c) for c in contours)
            motion_ratio = total_motion_area / frame_area

            # If too much of the frame is in motion, it's likely camera movement - skip
            if motion_ratio > self.max_motion_area_ratio:
                logger.debug(f"Skipping frame - widespread motion detected ({motion_ratio*100:.1f}% of frame)")
                self.prev_frame = gray
                return []

            # Process individual motion contours
            current_motion_regions = []

            for contour in contours:
                area = cv2.contourArea(contour)

                # Filter by size
                if area < self.min_motion_area:
                    continue

                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)

                # Calculate center point for tracking
                center_x = x + w // 2
                center_y = y + h // 2
                region_key = (center_x // 50, center_y // 50)  # Grid-based tracking

                # Track this motion region
                current_motion_regions.append(region_key)

                # Check if this region has been detected before
                if region_key not in self.motion_history:
                    self.motion_history[region_key] = {'count': 1, 'bbox': (x, y, w, h)}
                else:
                    self.motion_history[region_key]['count'] += 1
                    self.motion_history[region_key]['bbox'] = (x, y, w, h)

                # Only report motion that has persisted for multiple frames
                if self.motion_history[region_key]['count'] >= self.motion_persistence_frames:
                    # Calculate confidence based on persistence and area
                    persistence_bonus = min(0.3, self.motion_history[region_key]['count'] * 0.05)
                    size_confidence = min(0.5, area / (frame_area * 0.1))
                    confidence = min(0.95, 0.4 + size_confidence + persistence_bonus)

                    detection = Detection(
                        bbox=(x, y, x + w, y + h),
                        confidence=confidence,
                        class_id=999,  # Special ID for motion
                        class_name="motion"
                    )
                    detections.append(detection)

            # Clean up old motion history (remove regions not seen in this frame)
            old_regions = set(self.motion_history.keys()) - set(current_motion_regions)
            for region in old_regions:
                del self.motion_history[region]

            # Update previous frame
            self.prev_frame = gray

            return detections

        except Exception as e:
            logger.error(f"Motion detection error: {e}")
            return []

    def detect(self, frame: np.ndarray, filter_road_objects: bool = True) -> List[Detection]:
        """
        Detect objects using selected detection mode + motion detection

        Args:
            frame: Input image (BGR format)
            filter_road_objects: Filter to road-relevant objects (model mode only)

        Returns:
            List of Detection objects (including motion detections)
        """
        if not self.is_initialized:
            return []

        # Get primary detections (YOLO or edge-based)
        if self.detection_mode == "model":
            detections = self._detect_with_model(frame, filter_road_objects)
        else:
            detections = self._detect_with_edges(frame, filter_road_objects)

        # Add motion detections for road safety (deer, animals, etc.)
        motion_detections = self._detect_motion(frame)

        # Combine detections, filtering out motion that overlaps with YOLO detections
        combined = list(detections)  # Start with YOLO/edge detections

        for motion_det in motion_detections:
            # Check if motion overlaps with existing detection
            overlaps = False
            for det in detections:
                if self._boxes_overlap(motion_det.bbox, det.bbox, threshold=0.3):
                    overlaps = True
                    break

            # Only add motion detection if it doesn't overlap with YOLO detection
            if not overlaps:
                combined.append(motion_det)

        return combined

    def _boxes_overlap(self, box1: tuple, box2: tuple, threshold: float = 0.3) -> bool:
        """
        Check if two bounding boxes overlap

        Args:
            box1: (x1, y1, x2, y2)
            box2: (x1, y1, x2, y2)
            threshold: IoU threshold for overlap

        Returns:
            True if boxes overlap significantly
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return False  # No overlap

        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        iou = intersection / union if union > 0 else 0
        return iou >= threshold

    def get_performance_metrics(self) -> dict:
        """Get current performance metrics"""
        return {
            'fps': self.fps,
            'inference_time_ms': self.last_inference_time * 1000,
            'using_vpi': True,
            'backend': 'VPI-CUDA'
        }

    def release(self):
        """Release resources"""
        self.is_initialized = False
        logger.info("VPI detector released")