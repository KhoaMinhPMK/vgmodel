#!/usr/bin/env python3
"""
Batch processing script for multiple images
"""

import sys
import argparse
import json
import logging
from pathlib import Path
import cv2

# Import the main detector
from yolo_detection import YOLODetector

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_images(input_dir, output_dir, detector):
    """Process all images in a directory"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in input_path.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        logger.error(f"No image files found in {input_dir}")
        return
    
    results = []
    
    for image_file in image_files:
        logger.info(f"Processing: {image_file.name}")
        
        # Read image
        image = cv2.imread(str(image_file))
        if image is None:
            logger.error(f"Could not read image: {image_file}")
            continue
        
        # Detect objects
        annotated_frame, detection_results = detector.detect_objects(image)
        detection_info = detector.get_detection_info(detection_results)
        
        # Save annotated image
        output_file = output_path / f"detected_{image_file.name}"
        cv2.imwrite(str(output_file), annotated_frame)
        
        # Save detection info
        results.append({
            "filename": image_file.name,
            "detections": detection_info
        })
        
        logger.info(f"Saved: {output_file}")
    
    # Save results summary
    results_file = output_path / "detection_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Batch processing completed. Results saved to {results_file}")


def main():
    """Main function for batch processing"""
    parser = argparse.ArgumentParser(description="Batch YOLO Object Detection")
    
    parser.add_argument("input_dir", help="Input directory containing images")
    parser.add_argument("output_dir", help="Output directory for results")
    parser.add_argument("--model", default="assets/models/yolo11l.pt",
                       help="YOLO model path")
    parser.add_argument("--device", default=None, choices=['cuda', 'cpu'],
                       help="Device to use")
    parser.add_argument("--gpu", type=int, default=0,
                       help="GPU ID to use")
    parser.add_argument("--confidence", type=float, default=0.5,
                       help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45,
                       help="IoU threshold")
    
    args = parser.parse_args()
    
    # Validate paths
    input_path = Path(args.input_dir)
    if not input_path.exists():
        logger.error(f"Input directory not found: {input_path}")
        sys.exit(1)
    
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        sys.exit(1)
    
    # Create detector
    detector = YOLODetector(
        model_path=str(model_path),
        device=args.device,
        gpu_id=args.gpu,
        confidence=args.confidence,
        iou_threshold=args.iou
    )
    
    try:
        # Process images
        process_images(args.input_dir, args.output_dir, detector)
    except Exception as e:
        logger.error(f"Error during batch processing: {e}")
        sys.exit(1)
    finally:
        detector.cleanup()


if __name__ == "__main__":
    main()
