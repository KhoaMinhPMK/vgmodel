#!/usr/bin/env python3
"""
YOLO People Detection - Only detect people and show bounding boxes
"""

import cv2
import torch
import time
import argparse
import sys
import logging
from pathlib import Path
from typing import Optional, Tuple, Union, Any
from ultralytics import YOLO  # type: ignore

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# COCO class names for reference
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

class PeopleDetector:
    """YOLO People Detection - Only detect people"""
    
    def __init__(self, model_path: str = "assets/models/yolo11l.pt", device: Optional[str] = None, 
                 confidence: float = 0.5, iou_threshold: float = 0.45) -> None:
        """
        Initialize People detector
        
        Args:
            model_path: Path to YOLO model file
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
            confidence: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.model_path = model_path
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        
        # Setup device
        self.device = self._setup_device(device)
        
        # Load model
        self._load_model()
        
    def _setup_device(self, device: Optional[str]) -> str:
        """Setup device for inference"""
        if device is not None:
            return device
        
        if torch.cuda.is_available():
            logger.info(f"CUDA available! Using GPU: {torch.cuda.get_device_name(0)}")
            return 'cuda'
        else:
            logger.info("CUDA not available, using CPU")
            return 'cpu'
    
    def _load_model(self) -> None:
        """Load YOLO model"""
        try:
            logger.info(f"Loading YOLO model: {self.model_path}")
            self.model = YOLO(self.model_path)
            
            if self.device == 'cuda':
                self.model.to('cuda')
                logger.info("Model moved to GPU")
            
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            sys.exit(1)
    
    def detect_people(self, frame: Any) -> Tuple[Any, int]:
        """
        Detect only people in the frame and draw bounding boxes
        
        Args:
            frame: Input frame
            
        Returns:
            annotated_frame: Frame with only people bounding boxes
            people_count: Number of people detected
        """
        try:
            # Run inference
            results = self.model(frame, conf=self.confidence, iou=self.iou_threshold) # type: ignore
            
            # Create a copy of the frame
            annotated_frame = frame.copy()
            
            # Process detections
            people_count = 0
            if results[0].boxes is not None: # type: ignore
                for box in results[0].boxes: # type: ignore
                    # Get class ID
                    class_id = int(box.cls.cpu().numpy()[0]) # type: ignore
                    
                    # Only process if it's a person (class 0)
                    if class_id == 0:  # Person class
                        people_count += 1
                        
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0].astype(int) # type: ignore
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # type: ignore # Type cast for clarity
                        
                        # Get confidence
                        confidence = float(box.conf.cpu().numpy()[0]) # type: ignore
                        
                        # Draw bounding box (green color)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Optional: Draw confidence score
                        label = f"Person {confidence:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                        cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), (0, 255, 0), -1)
                        cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            return annotated_frame, people_count
            
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return frame, 0


class VideoProcessor:
    """Video processing for people detection"""
    
    def __init__(self, detector: 'PeopleDetector', output_dir: str = "output") -> None:
        self.detector = detector
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps_calculator = FPSCalculator()
    
    def process_video(self, video_path: Union[str, Path], display_fps: bool = True, save_output: bool = False) -> None:
        """Process video file with people detection"""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            logger.error(f"Could not open video file {video_path}")
            return
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Video loaded: {video_path}")
        logger.info(f"Total frames: {total_frames}")
        logger.info(f"Video FPS: {video_fps}")
        
        # Setup output video writer if needed
        out_writer = None
        if save_output:
            output_path = self.output_dir / f"people_detection_{Path(video_path).stem}.mp4"
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
            out_writer = cv2.VideoWriter(str(output_path), fourcc, video_fps,  # type: ignore
                                       (frame_width, frame_height))
        
        frame_count = 0
        total_people_detected = 0
        
        logger.info("Starting people detection...")
        logger.info("Press 'q' to quit, 's' to save frame")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.info("End of video reached")
                    break
                
                frame_count += 1
                
                # Process frame
                processed_frame, people_count = self._process_frame(frame, display_fps, 
                                                                  frame_count, total_frames)
                total_people_detected += people_count
                
                # Save frame to output video
                if out_writer:
                    out_writer.write(processed_frame)
                
                # Display frame
                cv2.imshow('People Detection', processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self._save_frame(processed_frame)
        
        finally:
            cap.release()
            if out_writer:
                out_writer.release()
            cv2.destroyAllWindows()
            logger.info(f"People detection completed")
            logger.info(f"Total people detected: {total_people_detected}")
    
    def _process_frame(self, frame: Any, display_fps: bool = True, frame_count: Optional[int] = None, total_frames: Optional[int] = None) -> Tuple[Any, int]:
        """Process a single frame with people detection"""
        # Perform people detection
        start_time = time.time()
        annotated_frame, people_count = self.detector.detect_people(frame)
        inference_time = time.time() - start_time
        
        # Update FPS
        fps = self.fps_calculator.update()
        
        # Add overlays
        if display_fps:
            self._add_overlays(annotated_frame, fps, inference_time, 
                             frame_count, total_frames, people_count)
        
        return annotated_frame, people_count
    
    def _add_overlays(self, frame: Any, fps: float, inference_time: float, frame_count: Optional[int] = None, 
                     total_frames: Optional[int] = None, people_count: int = 0) -> None:
        """Add FPS and other overlays to frame"""
        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Inference time
        cv2.putText(frame, f"Inference: {inference_time*1000:.1f}ms", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # People count
        cv2.putText(frame, f"People: {people_count}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Frame progress
        if frame_count is not None and total_frames is not None:
            cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    def _save_frame(self, frame: Any) -> None:
        """Save current frame to file"""
        timestamp = int(time.time())
        filename = self.output_dir / f"people_detection_{timestamp}.jpg"
        cv2.imwrite(str(filename), frame)
        logger.info(f"Frame saved as {filename}")


class FPSCalculator:
    """Calculate FPS for video processing"""
    
    def __init__(self) -> None:
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
    
    def update(self) -> float:
        """Update FPS calculation"""
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.start_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.start_time)
            self.frame_count = 0
            self.start_time = current_time
        
        return self.fps


def main() -> None:
    """Main function"""
    parser = argparse.ArgumentParser(
        description="YOLO People Detection - Only detect people and show bounding boxes"
    )
    
    # Model arguments
    parser.add_argument("--model", default="assets/models/yolo11l.pt",
                       help="YOLO model path")
    
    # Device arguments
    parser.add_argument("--device", default=None, choices=['cuda', 'cpu'],
                       help="Device to use (default: auto-detect)")
    
    # Detection arguments
    parser.add_argument("--confidence", type=float, default=0.5,
                       help="Confidence threshold (default: 0.5)")
    parser.add_argument("--iou", type=float, default=0.45,
                       help="IoU threshold (default: 0.45)")
    
    # Input arguments
    parser.add_argument("--video", type=str, default="assets/videos/test.mp4",
                       help="Path to video file")
    
    # Display arguments
    parser.add_argument("--no-fps", action="store_true",
                       help="Don't display FPS on screen")
    parser.add_argument("--save-output", action="store_true",
                       help="Save detection output video")
    
    # Output arguments
    parser.add_argument("--output-dir", default="output",
                       help="Output directory (default: output)")
    
    args = parser.parse_args()
    
    # Validate model path
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        sys.exit(1)
    
    # Validate video path
    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        sys.exit(1)
    
    # Create detector
    logger.info("Initializing People detector...")
    detector = PeopleDetector(
        model_path=str(model_path),
        device=args.device,
        confidence=args.confidence,
        iou_threshold=args.iou
    )
    
    # Create video processor
    video_processor = VideoProcessor(
        detector=detector,
        output_dir=args.output_dir
    )
    
    try:
        # Run detection
        logger.info(f"Starting people detection on video: {args.video}")
        video_processor.process_video(
            video_path=args.video,
            display_fps=not args.no_fps,
            save_output=args.save_output
        )
    
    except KeyboardInterrupt:
        logger.info("Detection interrupted by user")
    except Exception as e:
        logger.error(f"Error during detection: {e}")
        sys.exit(1)
    finally:
        logger.info("Application shutdown complete")


if __name__ == "__main__":
    main()
