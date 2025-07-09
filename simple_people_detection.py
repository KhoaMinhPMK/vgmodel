#!/usr/bin/env python3
"""
YOLO People Detection - Enhanced with corner points and hand/foot detection
"""

# Type: ignore comments are used for YOLO and OpenCV libraries
# which don't have complete type stubs available

import cv2
import torch
import time
import logging
from pathlib import Path
from ultralytics import YOLO  # type: ignore
import numpy as np
from typing import Tuple, Optional, Union, List # type: ignore
"""
YOLO People Detection - Enhanced with corner points and hand/foot detection
"""

import cv2 # type: ignore
import torch # type: ignore
import time # type: ignore
import logging # type: ignore
from pathlib import Path
from ultralytics import YOLO # type: ignore
import numpy as np
from typing import Tuple, Optional, Union, List # type: ignore

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimplePeopleDetector:
    """Simple People Detection - Enhanced with corner points and hand/foot detection"""
    
    def __init__(self, model_path: str = "assets/models/yolo11l.pt", confidence: float = 0.5) -> None:
        self.model_path = model_path
        self.confidence = confidence
        
        # Setup device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        # Load models
        self._load_model()
        self._load_pose_model()
    
    def _load_model(self) -> None:
        """Load YOLO model"""
        try:
            logger.info(f"Loading YOLO model: {self.model_path}")
            self.model = YOLO(self.model_path)
            
            if self.device == 'cuda':
                self.model.to('cuda')
            
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _load_pose_model(self) -> None:
        """Load YOLO pose model for keypoint detection"""
        try:
            # Try to load pose model, fallback to regular model if not available
            pose_model_path = "yolo11n-pose.pt"  # Lighter pose model
            logger.info(f"Loading YOLO pose model: {pose_model_path}")
            self.pose_model = YOLO(pose_model_path)
            
            if self.device == 'cuda':
                self.pose_model.to('cuda')
            
            logger.info("Pose model loaded successfully!")
            self.has_pose_model = True
            
        except Exception as e:
            logger.warning(f"Could not load pose model: {e}. Hand/foot detection will be estimated.")
            self.has_pose_model = False
    
    def _draw_corner_points(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> None:  # type: ignore
        """Draw corner points A, B, C, D on bounding box"""
        # Define corner points
        corners = {
            'A': (x1, y1),      # Top-left
            'B': (x2, y1),      # Top-right  
            'C': (x2, y2),      # Bottom-right
            'D': (x1, y2)       # Bottom-left
        }
        
        # Draw corner points
        for label, (x, y) in corners.items():
            # Draw circle for corner point
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Red filled circle  # type: ignore
            
            # Draw label
            cv2.putText(frame, label, (x - 10, y - 10),   # type: ignore
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    def _draw_roi_area(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> None:  # type: ignore
        """Draw ROI (Region of Interest) area with semi-transparent overlay"""
        # Create overlay
        overlay = frame.copy()  # type: ignore
        
        # Draw filled rectangle for ROI
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 0), -1)  # Yellow fill  # type: ignore
        
        # Blend with original frame (30% opacity)
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)  # type: ignore
        
        # Draw ROI border
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)  # Yellow border  # type: ignore
        
        # Add ROI label
        cv2.putText(frame, "ROI", (x1, y1 - 10),   # type: ignore
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    def _estimate_hand_foot_boxes(self, x1: int, y1: int, x2: int, y2: int) -> List[Tuple[Tuple[int, int, int, int], str]]:  # type: ignore
        """Estimate hand and foot bounding boxes based on person bbox"""
        boxes = []
        
        width = x2 - x1
        height = y2 - y1
        
        # Estimate hand positions (upper 40% of body, sides)
        hand_size = int(width * 0.15)  # Hand size relative to body width
        
        # Left hand (estimated position)
        left_hand_x = x1 - hand_size // 2
        left_hand_y = y1 + int(height * 0.2)
        boxes.append(((left_hand_x, left_hand_y, left_hand_x + hand_size, left_hand_y + hand_size), "Left Hand"))
        
        # Right hand (estimated position)  
        right_hand_x = x2 - hand_size // 2
        right_hand_y = y1 + int(height * 0.2)
        boxes.append(((right_hand_x, right_hand_y, right_hand_x + hand_size, right_hand_y + hand_size), "Right Hand"))
        
        # Estimate foot positions (bottom 20% of body)
        foot_width = int(width * 0.2)
        foot_height = int(height * 0.15)
        
        # Left foot
        left_foot_x = x1 + int(width * 0.2)
        left_foot_y = y2 - foot_height
        boxes.append(((left_foot_x, left_foot_y, left_foot_x + foot_width, left_foot_y + foot_height), "Left Foot"))
        
        # Right foot
        right_foot_x = x2 - int(width * 0.4)
        right_foot_y = y2 - foot_height  
        boxes.append(((right_foot_x, right_foot_y, right_foot_x + foot_width, right_foot_y + foot_height), "Right Foot"))
        
        return boxes
    
    def _draw_hand_foot_boxes(self, frame: np.ndarray, hand_foot_boxes: List[Tuple[Tuple[int, int, int, int], str]]) -> None:  # type: ignore
        """Draw hand and foot bounding boxes"""
        colors = {
            "Left Hand": (255, 0, 0),   # Blue
            "Right Hand": (255, 0, 0),  # Blue  
            "Left Foot": (0, 165, 255), # Orange
            "Right Foot": (0, 165, 255) # Orange
        }
        
        for (x1, y1, x2, y2), label in hand_foot_boxes:
            color = colors.get(label, (128, 128, 128))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # type: ignore
            
            # Draw label
            cv2.putText(frame, label, (x1, y1 - 5),   # type: ignore
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def detect_people(self, frame: np.ndarray) -> Tuple[np.ndarray, int]: # type: ignore
        """
        Detect people and draw enhanced bounding boxes with corner points and hand/foot detection
        
        Args:
            frame: Input frame
            
        Returns:
            annotated_frame: Frame with enhanced people detection
            people_count: Number of people detected
        """
        try:
            # Run inference
            results = self.model(frame, conf=self.confidence) # type: ignore
            
            # Create a copy of the frame
            annotated_frame = frame.copy() # type: ignore
            
            # Process detections
            people_count = 0
            if results[0].boxes is not None: # type: ignore
                for box in results[0].boxes: # type: ignore
                    # Get class ID - use type: ignore for YOLO attributes
                    class_id = int(box.cls.cpu().numpy()[0])  # type: ignore
                    
                    # Only process if it's a person (class 0)
                    if class_id == 0:  # Person class
                        people_count += 1
                        
                        # Get bounding box coordinates - use type: ignore for YOLO attributes
                        coords = box.xyxy.cpu().numpy()[0].astype(int)  # type: ignore
                        x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                        
                        # Draw ROI area first (so it's behind other elements)
                        self._draw_roi_area(annotated_frame, x1, y1, x2, y2)
                        
                        # Draw main bounding box (green)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # type: ignore
                        
                        # Draw corner points A, B, C, D
                        self._draw_corner_points(annotated_frame, x1, y1, x2, y2)
                        
                        # Estimate and draw hand/foot boxes
                        hand_foot_boxes = self._estimate_hand_foot_boxes(x1, y1, x2, y2)
                        self._draw_hand_foot_boxes(annotated_frame, hand_foot_boxes)
                        
                        # Get and display confidence
                        confidence = float(box.conf.cpu().numpy()[0])  # type: ignore
                        cv2.putText(annotated_frame, f"Person {confidence:.2f}", (x1, y1 - 25),  # type: ignore
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            return annotated_frame, people_count # type: ignore
            
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return frame, 0 # type: ignore

def process_video(video_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None, 
                 detector: Optional[SimplePeopleDetector] = None) -> None:
    """Process video file with people detection"""
    # Initialize detector if not provided
    if detector is None:
        detector = SimplePeopleDetector()
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        logger.error(f"Could not open video file {video_path}")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Video loaded: {video_path}")
    logger.info(f"Total frames: {total_frames}")
    logger.info(f"Video FPS: {video_fps}")
    
    # Setup output video writer if needed
    out_writer: Optional[cv2.VideoWriter] = None
    if output_path:
        # Fix for cv2.VideoWriter_fourcc type issue
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')  # type: ignore
        out_writer = cv2.VideoWriter(str(output_path), fourcc, video_fps, 
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
            
            # Detect people
            annotated_frame, people_count = detector.detect_people(frame) # type: ignore
            total_people_detected += people_count
            
            # Add frame counter
            cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}",  # type: ignore
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add people count
            cv2.putText(annotated_frame, f"People: {people_count}",  # type: ignore
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Save frame to output video
            if out_writer:
                out_writer.write(annotated_frame) # type: ignore
            
            # Display frame
            cv2.imshow('People Detection - Enhanced with Corner Points & Hand/Foot', annotated_frame) # type: ignore
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = int(time.time())
                filename = f"people_detection_{timestamp}.jpg"
                cv2.imwrite(filename, annotated_frame) # type: ignore
                logger.info(f"Frame saved as {filename}")
    
    finally:
        cap.release()
        if out_writer:
            out_writer.release()
        cv2.destroyAllWindows()
        logger.info(f"People detection completed")
        logger.info(f"Total people detected: {total_people_detected}")

def main() -> None:
    """Main function"""
    # Video path
    video_path = Path("assets/videos/test.mp4")
    
    # Output path (optional)
    output_path = Path("output/people_detection_enhanced.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if video exists
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        return
    
    # Process video
    process_video(video_path, output_path)

if __name__ == "__main__":
    main()
