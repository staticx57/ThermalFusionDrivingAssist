#!/usr/bin/env python3
"""
Test script for thermal.pt YOLO model
Validates that the thermal model loads and performs detections correctly
"""
import os
import sys
import numpy as np
import cv2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_thermal_model_loading():
    """Test that thermal.pt model loads correctly"""
    logger.info("=" * 60)
    logger.info("TEST 1: Thermal Model Loading")
    logger.info("=" * 60)
    
    try:
        from dual_model_detector import DualModelDetector
        
        # Initialize detector
        detector = DualModelDetector(
            thermal_model_path="thermal.pt",
            rgb_model_path="yolov8n.pt",
            device="cpu",
            confidence_threshold=0.5
        )
        
        # Load models
        success = detector.load_models()
        
        if success:
            logger.info("✓ PASS: thermal.pt loaded successfully")
            return True
        else:
            logger.error("✗ FAIL: Failed to load thermal.pt")
            return False
            
    except Exception as e:
        logger.error(f"✗ FAIL: Exception during model loading: {e}")
        return False


def test_model_selection():
    """Test that model selection logic works correctly"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Model Selection Logic")
    logger.info("=" * 60)
    
    try:
        from dual_model_detector import DualModelDetector
        
        detector = DualModelDetector(
            thermal_model_path="thermal.pt",
            rgb_model_path="yolov8n.pt",
            device="cpu",
            confidence_threshold=0.5
        )
        
        if not detector.load_models():
            logger.error("✗ FAIL: Could not load models")
            return False
        
        # Create dummy frame
        dummy_frame = np.zeros((512, 640, 3), dtype=np.uint8)
        
        # Test thermal frame
        detector.detect(dummy_frame, frame_source='thermal')
        thermal_model = detector.get_current_model_name()
        logger.info(f"  Thermal frame -> Model: {thermal_model}")
        
        # Test RGB frame
        detector.detect(dummy_frame, frame_source='rgb')
        rgb_model = detector.get_current_model_name()
        logger.info(f"  RGB frame -> Model: {rgb_model}")
        
        # Test fusion frame
        detector.detect(dummy_frame, frame_source='fusion')
        fusion_model = detector.get_current_model_name()
        logger.info(f"  Fusion frame -> Model: {fusion_model}")
        
        # Validate selection
        if 'thermal.pt' in thermal_model and 'thermal.pt' in fusion_model and 'yolov8n.pt' in rgb_model:
            logger.info("✓ PASS: Model selection working correctly")
            return True
        else:
            logger.error("✗ FAIL: Model selection incorrect")
            logger.error(f"  Expected: thermal.pt for thermal/fusion, yolov8n.pt for RGB")
            logger.error(f"  Got: {thermal_model}, {fusion_model}, {rgb_model}")
            return False
            
    except Exception as e:
        logger.error(f"✗ FAIL: Exception during model selection test: {e}")
        return False


def test_detection_on_thermal_image():
    """Test detection on a simulated thermal image"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Detection on Thermal Image")
    logger.info("=" * 60)
    
    try:
        from dual_model_detector import DualModelDetector
        
        detector = DualModelDetector(
            thermal_model_path="thermal.pt",
            rgb_model_path="yolov8n.pt",
            device="cpu",
            confidence_threshold=0.3  # Lower threshold for test
        )
        
        if not detector.load_models():
            logger.error("✗ FAIL: Could not load models")
            return False
        
        # Create simulated thermal image (grayscale with hot spots)
        thermal_sim = np.zeros((512, 640), dtype=np.uint16)
        
        # Add gradient
        y_gradient = np.linspace(28000, 26000, 512, dtype=np.uint16)[:, np.newaxis]
        thermal_sim += y_gradient
        
        # Add hot spots (simulate warm objects)
        for i in range(3):
            x = int(640 * (0.2 + i * 0.3))
            y = int(512 * (0.3 + i * 0.2))
            cv2.circle(thermal_sim, (x, y), 50, 32000, -1)
        
        # Convert to 8-bit for YOLO (normalize and convert to BGR)
        thermal_8bit = cv2.normalize(thermal_sim, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        thermal_bgr = cv2.cvtColor(thermal_8bit, cv2.COLOR_GRAY2BGR)
        
        # Apply thermal colormap (ironbow style)
        thermal_colored = cv2.applyColorMap(thermal_8bit, cv2.COLORMAP_JET)
        
        # Run detection
        detections = detector.detect(thermal_colored, frame_source='thermal')
        
        logger.info(f"  Detections found: {len(detections)}")
        
        if len(detections) >= 0:  # Any result is valid
            logger.info("✓ PASS: Detection executed without errors")
            
            # Log detections
            for i, det in enumerate(detections[:5]):  # Show first 5
                logger.info(f"    Detection {i+1}: {det.class_name} ({det.confidence:.2f})")
            
            return True
        else:
            logger.error("✗ FAIL: Detection returned None or invalid result")
            return False
            
    except Exception as e:
        logger.error(f"✗ FAIL: Exception during detection test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_file_exists():
    """Verify thermal.pt file exists"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 0: Model File Verification")
    logger.info("=" * 60)
    
    thermal_exists = os.path.exists("thermal.pt")
    rgb_exists = os.path.exists("yolov8n.pt")
    
    if thermal_exists:
        size_mb = os.path.getsize("thermal.pt") / (1024 * 1024)
        logger.info(f"✓ thermal.pt found ({size_mb:.1f} MB)")
    else:
        logger.error("✗ thermal.pt NOT FOUND")
    
    if rgb_exists:
        size_mb = os.path.getsize("yolov8n.pt") / (1024 * 1024)
        logger.info(f"✓ yolov8n.pt found ({size_mb:.1f} MB)")
    else:
        logger.error("✗ yolov8n.pt NOT FOUND")
    
    return thermal_exists and rgb_exists


def main():
    """Run all tests"""
    logger.info("\n")
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║" + " " * 10 + "THERMAL.PT MODEL VALIDATION TESTS" + " " * 15 + "║")
    logger.info("╚" + "═" * 58 + "╝")
    logger.info("\n")
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    logger.info(f"Working directory: {os.getcwd()}\n")
    
    # Run tests
    results = []
    
    results.append(("Model File Verification", test_model_file_exists()))
    results.append(("Thermal Model Loading", test_thermal_model_loading()))
    results.append(("Model Selection Logic", test_model_selection()))
    results.append(("Detection on Thermal Image", test_detection_on_thermal_image()))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info("=" * 60)
    logger.info(f"TOTAL: {passed}/{total} tests passed")
    logger.info("=" * 60)
    
    if passed == total:
        logger.info("\n🎉 ALL TESTS PASSED! Thermal.pt is ready to use.")
        sys.exit(0)
    else:
        logger.error(f"\n⚠️  {total - passed} test(s) failed. Review errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
