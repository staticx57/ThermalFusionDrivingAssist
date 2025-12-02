"""
Dual Model Detector - Thermal & RGB YOLO Models
Manages separate YOLO models for thermal and RGB camera feeds
Uses thermal.pt (FLIR-trained) for thermal, yolov8n.pt for RGB
"""
import logging
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class DualModelDetector:
    """
    Dual YOLO model manager for thermal and RGB cameras
    
    Uses specialized models for each camera type:
    - thermal.pt: YOLO trained on FLIR thermal dataset
    - yolov8n.pt: Standard YOLO for RGB camera
    """
    
    def __init__(self, thermal_model_path: str = "thermal.pt", 
                 rgb_model_path: str = "yolov8n.pt",
                 device: str = "cpu"):
        """
        Initialize dual model detector
        
        Args:
            thermal_model_path: Path to thermal YOLO model
            rgb_model_path: Path to RGB YOLO model  
            device: 'cuda' or 'cpu'
        """
        self.thermal_model_path = thermal_model_path
        self.rgb_model_path = rgb_model_path
        self.device = device
        
        self.thermal_model = None
        self.rgb_model = None
        self.current_model_name = "None"
        
        logger.info(f"DualModelDetector initialized:")
        logger.info(f"  Thermal model: {thermal_model_path}")
        logger.info(f"  RGB model: {rgb_model_path}")
        logger.info(f"  Device: {device}")
        
    def load_models(self):
        """Load both YOLO models"""
        try:
            from ultralytics import YOLO
            
            # Load thermal model
            logger.info(f"Loading thermal model: {self.thermal_model_path}")
            self.thermal_model = YOLO(self.thermal_model_path)
            self.thermal_model.to(self.device)
            logger.info(f"✓ Thermal model loaded successfully")
            
            # Load RGB model
            logger.info(f"Loading RGB model: {self.rgb_model_path}")
            self.rgb_model = YOLO(self.rgb_model_path)
            self.rgb_model.to(self.device)
            logger.info(f"✓ RGB model loaded successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False
    
    def detect(self, frame: np.ndarray, frame_source: str = "thermal", 
               confidence_threshold: float = 0.5) -> List:
        """
        Run detection on frame using appropriate model
        
        Args:
            frame: Input image (BGR format)
            frame_source: 'thermal', 'rgb', or 'fusion'
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            List of Detection objects
        """
        # Select model based on frame source
        if frame_source in ["thermal", "fusion"]:
            # Use thermal model for thermal and fusion frames
            model = self.thermal_model
            self.current_model_name = self.thermal_model_path
        else:
            # Use RGB model for RGB frames
            model = self.rgb_model
            self.current_model_name = self.rgb_model_path
            
        if model is None:
            logger.warning(f"Model not loaded for {frame_source}")
            return []
            
        try:
            # Run YOLO inference
            results = model(frame, conf=confidence_threshold, verbose=False)
            
            # Convert to Detection objects
            from object_detector import Detection
            detections = []
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    label = model.names[cls]
                    
                    detection = Detection(
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        confidence=conf,
                        class_name=label
                    )
                    detections.append(detection)
                    
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []
    
    def get_current_model_name(self) -> str:
        """Get the name of the currently active model"""
        return self.current_model_name
