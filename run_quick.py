#!/usr/bin/env python3
"""
Quick run script for testing YOLO detection
"""

import sys
from pathlib import Path
import logging

# Import the main detector
from yolo_detection import YOLODetector, VideoProcessor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Quick test with default video"""
    # Default paths
    model_path = Path("assets/models/yolo11l.pt")
    video_path = Path("assets/videos/test.mp4")
    
    # Check if files exist
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        logger.info("Please ensure yolo11l.pt is in the assets/models/ directory")
        sys.exit(1)
    
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        logger.info("Please ensure test.mp4 is in the assets/videos/ directory")
        sys.exit(1)
    
    logger.info(f"Running YOLO detection on: {video_path}")
    
    # Create detector
    detector = YOLODetector(
        model_path=str(model_path),
        device=None,  # Auto-detect
        gpu_id=0,
        confidence=0.5,
        iou_threshold=0.45
    )
    
    # Create video processor
    video_processor = VideoProcessor(
        detector=detector,
        output_dir="output"
    )
    
    try:
        # Run video detection
        video_processor.process_video(
            video_path=str(video_path),
            display_fps=True,
            save_output=False
        )
    except KeyboardInterrupt:
        logger.info("Detection interrupted by user")
    except Exception as e:
        logger.error(f"Error during detection: {e}")
        sys.exit(1)
    finally:
        detector.cleanup()
        logger.info("Quick run completed")


if __name__ == "__main__":
    main()
