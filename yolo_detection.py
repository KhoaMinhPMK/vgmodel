#!/usr/bin/env python3
"""
YOLO Real-time Detection - Professional Version
Main application with modular structure
"""

import cv2
import torch
import time
import argparse
import sys
import logging
from pathlib import Path
from ultralytics import YOLO
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DeviceManager:
    """Manages GPU/CPU device selection"""
    
    def __init__(self, device=None, gpu_id=0):
        self.device = device
        self.gpu_id = gpu_id
        self._selected_device = self._setup_device()
    
    def _setup_device(self):
        """Setup and configure device"""
        if self.device is not None:
            if self.device == 'cuda' and torch.cuda.is_available():
                self._setup_gpu()
            return self.device
        
        # Auto-detection
        if torch.cuda.is_available():
            self._setup_gpu()
            return 'cuda'
        else:
            logger.info("CUDA not available, using CPU")
            return 'cpu'
    
    def _setup_gpu(self):
        """Setup GPU configuration"""
        gpu_count = torch.cuda.device_count()
        logger.info(f"CUDA available! Found {gpu_count} GPU(s):")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Select GPU
        if self.gpu_id < gpu_count:
            torch.cuda.set_device(self.gpu_id)
            logger.info(f"Using GPU {self.gpu_id}: {torch.cuda.get_device_name(self.gpu_id)}")
        else:
            logger.warning(f"GPU {self.gpu_id} not available, using GPU 0")
            torch.cuda.set_device(0)
            logger.info(f"Using GPU 0: {torch.cuda.get_device_name(0)}")
    
    def get_device(self):
        """Get selected device"""
        return self._selected_device


class YOLODetector:
    """YOLO Object Detection class with GPU support"""
    
    def __init__(self, model_path="assets/models/yolo11l.pt", device=None, gpu_id=0, 
                 confidence=0.5, iou_threshold=0.45):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO model file
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
            gpu_id: GPU ID to use when device is 'cuda'
            confidence: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.model_path = model_path
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        
        # Initialize device manager
        self.device_manager = DeviceManager(device, gpu_id)
        self.device = self.device_manager.get_device()
        
        # Load model
        self._load_model()
        
    def _load_model(self):
        """Load YOLO model"""
        try:
            logger.info(f"Loading YOLO model: {self.model_path}")
            self.model = YOLO(self.model_path)
            
            # Move model to GPU if available
            if self.device == 'cuda':
                self.model.to('cuda')
                logger.info("Model moved to GPU")
            
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            sys.exit(1)
    
    def detect_objects(self, frame):
        """
        Perform object detection on a single frame
        
        Args:
            frame: Input frame
            
        Returns:
            annotated_frame: Frame with detection annotations
            results: Detection results
        """
        try:
            # Run inference
            results = self.model(frame, conf=self.confidence, iou=self.iou_threshold)
            
            # Annotate frame with detections
            annotated_frame = results[0].plot()
            
            return annotated_frame, results
            
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return frame, None
    
    def get_detection_info(self, results):
        """Extract detection information from results"""
        if results is None or results[0].boxes is None:
            return {"count": 0, "objects": []}
        
        detections = results[0].boxes
        objects = []
        
        for box in detections:
            if hasattr(box, 'cls') and hasattr(box, 'conf'):
                obj_info = {
                    "class": int(box.cls.cpu().numpy()[0]) if len(box.cls) > 0 else 0,
                    "confidence": float(box.conf.cpu().numpy()[0]) if len(box.conf) > 0 else 0.0,
                    "bbox": box.xyxy.cpu().numpy()[0].tolist() if len(box.xyxy) > 0 else []
                }
                objects.append(obj_info)
        
        return {
            "count": len(objects),
            "objects": objects
        }
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'model'):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Detector cleanup completed")


class VideoProcessor:
    """Video processing for YOLO detection"""
    
    def __init__(self, detector, output_dir="output"):
        self.detector = detector
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps_calculator = FPSCalculator()
    
    def process_camera(self, camera_index=0, display_fps=True, 
                      frame_width=640, frame_height=480):
        """Process camera feed with real-time detection"""
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            logger.error(f"Could not open camera {camera_index}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        logger.info("Starting camera detection...")
        logger.info("Press 'q' to quit, 's' to save frame")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Could not read frame from camera")
                    break
                
                # Process frame
                processed_frame = self._process_frame(frame, display_fps)
                
                # Display frame
                cv2.imshow('YOLO Camera Detection', processed_frame)
                
                # Handle key presses
                if self._handle_key_press(processed_frame):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Camera detection stopped")
    
    def process_video(self, video_path, display_fps=True, save_output=False):
        """Process video file with detection"""
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
            output_path = self.output_dir / f"output_{Path(video_path).stem}.mp4"
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(str(output_path), fourcc, video_fps, 
                                       (frame_width, frame_height))
        
        frame_count = 0
        paused = False
        
        logger.info("Starting video detection...")
        logger.info("Press 'q' to quit, 's' to save frame, 'space' to pause/resume")
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        logger.info("End of video reached")
                        break
                    frame_count += 1
                
                # Process frame
                processed_frame = self._process_frame(frame, display_fps, frame_count, total_frames)
                
                # Add pause indicator
                if paused:
                    cv2.putText(processed_frame, "PAUSED", (10, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Save frame to output video
                if out_writer and not paused:
                    out_writer.write(processed_frame)
                
                # Display frame
                cv2.imshow('YOLO Video Detection', processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self._save_frame(processed_frame)
                elif key == ord(' '):
                    paused = not paused
                    logger.info("Video paused" if paused else "Video resumed")
        
        finally:
            cap.release()
            if out_writer:
                out_writer.release()
            cv2.destroyAllWindows()
            logger.info("Video detection completed")
    
    def _process_frame(self, frame, display_fps=True, frame_count=None, total_frames=None):
        """Process a single frame with detection"""
        # Perform detection
        start_time = time.time()
        annotated_frame, results = self.detector.detect_objects(frame)
        inference_time = time.time() - start_time
        
        # Update FPS
        fps = self.fps_calculator.update()
        
        # Add overlays
        if display_fps:
            self._add_overlays(annotated_frame, fps, inference_time, 
                             frame_count, total_frames)
        
        return annotated_frame
    
    def _add_overlays(self, frame, fps, inference_time, frame_count=None, total_frames=None):
        """Add FPS and other overlays to frame"""
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Inference: {inference_time*1000:.1f}ms", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if frame_count is not None and total_frames is not None:
            cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    def _handle_key_press(self, frame):
        """Handle key press events"""
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return True
        elif key == ord('s'):
            self._save_frame(frame)
        return False
    
    def _save_frame(self, frame):
        """Save current frame to file"""
        timestamp = int(time.time())
        filename = self.output_dir / f"detection_frame_{timestamp}.jpg"
        cv2.imwrite(str(filename), frame)
        logger.info(f"Frame saved as {filename}")


class FPSCalculator:
    """Calculate FPS for video processing"""
    
    def __init__(self):
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
    
    def update(self):
        """Update FPS calculation"""
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.start_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.start_time)
            self.frame_count = 0
            self.start_time = current_time
        
        return self.fps


def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(
        description="YOLO Real-time Object Detection - Professional Version"
    )
    
    # Model arguments
    parser.add_argument("--model", default="assets/models/yolo11l.pt",
                       help="YOLO model path")
    
    # Device arguments
    parser.add_argument("--device", default=None, choices=['cuda', 'cpu'],
                       help="Device to use (default: auto-detect)")
    parser.add_argument("--gpu", type=int, default=0,
                       help="GPU ID to use (default: 0)")
    
    # Detection arguments
    parser.add_argument("--confidence", type=float, default=0.5,
                       help="Confidence threshold (default: 0.5)")
    parser.add_argument("--iou", type=float, default=0.45,
                       help="IoU threshold (default: 0.45)")
    
    # Input arguments
    parser.add_argument("--camera", type=int, default=0,
                       help="Camera index (default: 0)")
    parser.add_argument("--video", type=str, default=None,
                       help="Path to video file (if not provided, uses camera)")
    
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
    
    # Validate video path if provided
    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            sys.exit(1)
    
    # Create detector
    logger.info("Initializing YOLO detector...")
    detector = YOLODetector(
        model_path=str(model_path),
        device=args.device,
        gpu_id=args.gpu,
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
        if args.video:
            logger.info(f"Starting video detection: {args.video}")
            video_processor.process_video(
                video_path=args.video,
                display_fps=not args.no_fps,
                save_output=args.save_output
            )
        else:
            logger.info("Starting camera detection...")
            video_processor.process_camera(
                camera_index=args.camera,
                display_fps=not args.no_fps,
                frame_width=640,
                frame_height=480
            )
    
    except KeyboardInterrupt:
        logger.info("Detection interrupted by user")
    except Exception as e:
        logger.error(f"Error during detection: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        detector.cleanup()
        logger.info("Application shutdown complete")


if __name__ == "__main__":
    main()
