#!/usr/bin/env python3
"""
Test script to verify YOLO detection setup
"""

import sys
import torch
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_dependencies():
    """Test if all dependencies are installed"""
    logger.info("Testing dependencies...")
    
    try:
        import cv2
        logger.info(f"✓ OpenCV: {cv2.__version__}")
    except ImportError:
        logger.error("✗ OpenCV not installed")
        return False
    
    try:
        from ultralytics import YOLO
        logger.info("✓ Ultralytics YOLO")
    except ImportError:
        logger.error("✗ Ultralytics not installed")
        return False
    
    try:
        import numpy as np
        logger.info(f"✓ NumPy: {np.__version__}")
    except ImportError:
        logger.error("✗ NumPy not installed")
        return False
    
    return True


def test_gpu():
    """Test GPU availability"""
    logger.info("Testing GPU...")
    
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"GPU count: {gpu_count}")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        return True
    else:
        logger.warning("CUDA not available - will use CPU")
        return False


def test_files():
    """Test required files"""
    logger.info("Testing project files...")
    
    required_files = [
        "yolo_detection.py",
        "run_quick.py",
        "run_batch.py", 
        "requirements.txt",
        "assets/models/yolo11l.pt"
    ]
    
    base_path = Path(__file__).parent
    all_found = True
    
    for file_path in required_files:
        full_path = base_path / file_path
        if full_path.exists():
            logger.info(f"✓ {file_path}")
        else:
            logger.error(f"✗ {file_path} - Missing!")
            all_found = False
    
    return all_found


def test_detector():
    """Test detector import and creation"""
    logger.info("Testing detector...")
    
    try:
        from yolo_detection import YOLODetector, VideoProcessor
        logger.info("✓ Successfully imported YOLODetector and VideoProcessor")
        
        # Test detector creation (without actually loading model)
        logger.info("✓ Detector import successful")
        return True
        
    except Exception as e:
        logger.error(f"✗ Detector test failed: {e}")
        return False


def main():
    """Run all tests"""
    logger.info("=" * 60)
    logger.info("YOLO DETECTION SETUP VERIFICATION")
    logger.info("=" * 60)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("GPU Support", test_gpu),
        ("Project Files", test_files),
        ("Detector Import", test_detector)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{test_name}:")
        logger.info("-" * 40)
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nPassed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        logger.info("🎉 All tests passed! Project is ready to use.")
        logger.info("\nQuick start commands:")
        logger.info("  python yolo_detection.py                              # Camera detection")
        logger.info("  python yolo_detection.py --video assets/videos/test.mp4  # Video detection")
        logger.info("  python run_quick.py                                   # Quick test")
        logger.info("  python run_batch.py input_folder output_folder       # Batch processing")
    else:
        logger.error("❌ Some tests failed. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
