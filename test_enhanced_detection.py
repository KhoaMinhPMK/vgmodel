#!/usr/bin/env python3
"""
Test script for enhanced people detection with corner points and hand/foot detection
"""

import argparse
import sys
from pathlib import Path
from simple_people_detection import SimplePeopleDetector, process_video
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Enhanced People Detection with Corner Points and Hand/Foot Detection"
    )
    
    # Input arguments
    parser.add_argument("--video", type=str, default="assets/videos/test.mp4",
                       help="Path to input video file")
    
    # Model arguments
    parser.add_argument("--model", type=str, default="assets/models/yolo11l.pt",
                       help="Path to YOLO model file")
    
    # Detection arguments
    parser.add_argument("--confidence", type=float, default=0.5,
                       help="Detection confidence threshold (default: 0.5)")
    
    # Output arguments
    parser.add_argument("--output", type=str, default="output/enhanced_detection.mp4",
                       help="Output video path")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save output video")
    
    args = parser.parse_args()
    
    # Validate input video
    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        sys.exit(1)
    
    # Validate model
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        sys.exit(1)
    
    # Setup output path
    output_path = None if args.no_save else Path(args.output)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*50)
    logger.info("Enhanced People Detection Test")
    logger.info("="*50)
    logger.info(f"Video: {video_path}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Confidence: {args.confidence}")
    logger.info(f"Output: {output_path if output_path else 'Not saving'}")
    logger.info("="*50)
    
    # Create detector with custom parameters
    detector = SimplePeopleDetector(
        model_path=str(model_path),
        confidence=args.confidence
    )
    
    # Process video
    try:
        # Process video with custom detector
        process_video(video_path, output_path, detector)
        
    except KeyboardInterrupt:
        logger.info("Detection interrupted by user")
    except Exception as e:
        logger.error(f"Error during detection: {e}")
        sys.exit(1)
    
    logger.info("Enhanced detection completed successfully!")

if __name__ == "__main__":
    main()
