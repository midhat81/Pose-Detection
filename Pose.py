import cv2
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from collections import defaultdict, deque
from dataclasses import dataclass
import logging
import argparse
from ultralytics import YOLO
import torch.serialization
from ultralytics.nn.tasks import DetectionModel
from torch.nn.modules.container import Sequential


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Detection:
    """Single detection with confidence and class info"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    features: Optional[np.ndarray] = None

@dataclass
class Track:
    """Track object for maintaining identity across frames"""
    track_id: int
    global_id: str
    detections: List[Detection]
    features_history: deque
    last_seen: int
    confidence_history: deque
    bbox_history: deque
    
    def __post_init__(self):
        if not hasattr(self, 'features_history'):
            self.features_history = deque(maxlen=30)
        if not hasattr(self, 'confidence_history'):
            self.confidence_history = deque(maxlen=10)
        if not hasattr(self, 'bbox_history'):
            self.bbox_history = deque(maxlen=10)

class PersonDetector:
    """Enhanced person detection using YOLOv8/v11"""
    
    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.5):
        try:
            from ultralytics import YOLO
            import torch.serialization
            from ultralytics.nn.tasks import DetectionModel
            from torch.nn.modules.container import Sequential

            # Allow YOLO model classes to be deserialized safely
            torch.serialization.add_safe_globals([DetectionModel])
            torch.serialization.add_safe_globals([Sequential])

            self.model = YOLO(model_path)
            self.conf_threshold = conf_threshold
            logger.info(f"Loaded YOLO model: {model_path}")
        except ImportError:
            raise ImportError("Please install ultralytics: pip install ultralytics")

    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect persons in frame"""
        results = self.model(frame, conf=self.conf_threshold, classes=[0])  # Class 0 = person
        detections = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    # Filter out very small detections
                    if (x2 - x1) * (y2 - y1) > 1000:  # Min area threshold
                        detections.append(Detection(
                            bbox=(int(x1), int(y1), int(x2), int(y2)),
                            confidence=float(conf),
                            class_id=0
                        ))
        
        return detections

class FeatureExtractor:
    """Extract features for person re-identification"""
    
    def __init__(self, model_type: str = "insightface"):
        self.model_type = model_type
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the feature extraction model"""
        if self.model_type == "insightface":
            try:
                import insightface
                self.model = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'])
                self.model.prepare(ctx_id=0, det_size=(640, 640))
                logger.info("Initialized InsightFace model")
            except ImportError:
                logger.warning("InsightFace not available, using basic features")
                self.model = None
        else:
            # Fallback to basic histogram features
            self.model = None
    
    def extract_features(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract features from person crop"""
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]
        
        if crop.size == 0:
            return np.zeros(512)
        
        if self.model is not None and self.model_type == "insightface":
            # Use face features if available
            faces = self.model.get(crop)
            if faces:
                return faces[0].embedding
        
        # Fallback: Use color histogram + HOG features
        return self._extract_basic_features(crop)
    
    def _extract_basic_features(self, crop: np.ndarray) -> np.ndarray:
        """Extract basic visual features as fallback"""
        # Resize crop for consistent feature extraction
        crop_resized = cv2.resize(crop, (64, 128))
        
        # Color histogram
        hist_b = cv2.calcHist([crop_resized], [0], None, [32], [0, 256])
        hist_g = cv2.calcHist([crop_resized], [1], None, [32], [0, 256])
        hist_r = cv2.calcHist([crop_resized], [2], None, [32], [0, 256])
        
        # HOG features
        gray = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)
        hog = cv2.HOGDescriptor()
        hog_features = hog.compute(gray)
        
        # Combine features
        features = np.concatenate([
            hist_b.flatten(),
            hist_g.flatten(), 
            hist_r.flatten(),
            hog_features.flatten()[:100]  # Truncate HOG features
        ])
        
        return features / (np.linalg.norm(features) + 1e-6)

class MultiPersonTracker:
    """Enhanced multi-person tracking with identity management"""
    
    def __init__(self, max_disappeared: int = 30, max_distance: float = 0.3):
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1
        self.next_global_id = 1
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.frame_count = 0
        
        # Feature-based matching
        self.feature_extractor = FeatureExtractor()
        
    def update(self, detections: List[Detection], frame: np.ndarray) -> Dict[int, Track]:
        """Update tracks with new detections"""
        self.frame_count += 1
        
        # Extract features for all detections
        for detection in detections:
            detection.features = self.feature_extractor.extract_features(frame, detection.bbox)
        
        if not self.tracks:
            # Initialize tracks for first frame
            for detection in detections:
                self._create_new_track(detection)
        else:
            # Match detections to existing tracks
            self._match_detections_to_tracks(detections)
        
        # Remove old tracks
        self._remove_stale_tracks()
        
        return self.tracks
    
    def _match_detections_to_tracks(self, detections: List[Detection]):
        """Match detections to existing tracks using Hungarian algorithm"""
        if not detections or not self.tracks:
            return
        
        # Calculate cost matrix
        track_ids = list(self.tracks.keys())
        cost_matrix = np.zeros((len(detections), len(track_ids)))
        
        for i, detection in enumerate(detections):
            for j, track_id in enumerate(track_ids):
                track = self.tracks[track_id]
                cost = self._calculate_matching_cost(detection, track)
                cost_matrix[i, j] = cost
        
        # Simple greedy matching (can be replaced with Hungarian algorithm)
        matches, unmatched_detections, unmatched_tracks = self._greedy_match(
            cost_matrix, detections, track_ids
        )
        
        # Update matched tracks
        for det_idx, track_id in matches:
            self._update_track(self.tracks[track_id], detections[det_idx])
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            self._create_new_track(detections[det_idx])
        
        # Mark unmatched tracks as disappeared
        for track_id in unmatched_tracks:
            self.tracks[track_id].last_seen = self.frame_count
    
    def _calculate_matching_cost(self, detection: Detection, track: Track) -> float:
        """Calculate matching cost between detection and track"""
        # IoU-based cost
        if track.bbox_history:
            last_bbox = track.bbox_history[-1]
            iou = self._calculate_iou(detection.bbox, last_bbox)
            iou_cost = 1.0 - iou
        else:
            iou_cost = 1.0
        
        # Feature-based cost
        if detection.features is not None and track.features_history:
            avg_features = np.mean([f for f in track.features_history], axis=0)
            feature_similarity = np.dot(detection.features, avg_features)
            feature_cost = 1.0 - max(0, feature_similarity)
        else:
            feature_cost = 0.5
        
        # Combined cost
        total_cost = 0.4 * iou_cost + 0.6 * feature_cost
        
        return total_cost
    
    def _greedy_match(self, cost_matrix: np.ndarray, detections: List[Detection], 
                     track_ids: List[int]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Simple greedy matching algorithm"""
        matches = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = track_ids.copy()
        
        # Sort by cost and match greedily
        costs = []
        for i in range(cost_matrix.shape[0]):
            for j in range(cost_matrix.shape[1]):
                if cost_matrix[i, j] < self.max_distance:
                    costs.append((cost_matrix[i, j], i, j))
        
        costs.sort()
        
        for cost, det_idx, track_idx in costs:
            if det_idx in unmatched_detections and track_ids[track_idx] in unmatched_tracks:
                matches.append((det_idx, track_ids[track_idx]))
                unmatched_detections.remove(det_idx)
                unmatched_tracks.remove(track_ids[track_idx])
        
        return matches, unmatched_detections, unmatched_tracks
    
    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], 
                      bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate IoU between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _create_new_track(self, detection: Detection):
        """Create a new track for unmatched detection"""
        track_id = self.next_track_id
        global_id = f"person_{self.next_global_id}"
        
        track = Track(
            track_id=track_id,
            global_id=global_id,
            detections=[detection],
            features_history=deque(maxlen=30),
            last_seen=self.frame_count,
            confidence_history=deque(maxlen=10),
            bbox_history=deque(maxlen=10)
        )
        
        # Add initial data
        if detection.features is not None:
            track.features_history.append(detection.features)
        track.confidence_history.append(detection.confidence)
        track.bbox_history.append(detection.bbox)
        
        self.tracks[track_id] = track
        self.next_track_id += 1
        self.next_global_id += 1
        
        logger.info(f"Created new track: {global_id} (ID: {track_id})")
    
    def _update_track(self, track: Track, detection: Detection):
        """Update existing track with new detection"""
        track.detections.append(detection)
        track.last_seen = self.frame_count
        
        if detection.features is not None:
            track.features_history.append(detection.features)
        track.confidence_history.append(detection.confidence)
        track.bbox_history.append(detection.bbox)
    
    def _remove_stale_tracks(self):
        """Remove tracks that haven't been seen for too long"""
        stale_tracks = []
        for track_id, track in self.tracks.items():
            if self.frame_count - track.last_seen > self.max_disappeared:
                stale_tracks.append(track_id)
        
        for track_id in stale_tracks:
            logger.info(f"Removing stale track: {self.tracks[track_id].global_id}")
            del self.tracks[track_id]

class VideoProcessor:
    """Main video processing pipeline"""
    
    def __init__(self, yolo_model: str = "yolov8n.pt", conf_threshold: float = 0.5):
        self.detector = PersonDetector(yolo_model, conf_threshold)
        self.tracker = MultiPersonTracker()
        self.results = []
    
    def process_video(self, video_path: str, output_path: str = None, 
                     save_annotated: bool = False) -> List[Dict]:
        """Process entire video and return tracking results"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Processing video: {video_path}")
        logger.info(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup video writer for annotated output
        out_writer = None
        if save_annotated and output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            annotated_path = output_path.replace('.json', '_annotated.mp4')
            out_writer = cv2.VideoWriter(annotated_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        self.results = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect persons
            detections = self.detector.detect(frame)
            
            # Update tracker
            tracks = self.tracker.update(detections, frame)
            
            # Store results
            frame_results = []
            for track in tracks.values():
                if track.bbox_history:  # Only include tracks with recent detections
                    bbox = track.bbox_history[-1]
                    confidence = track.confidence_history[-1] if track.confidence_history else 0.0
                    
                    frame_results.append({
                        "frame": frame_idx,
                        "global_id": track.global_id,
                        "track_id": track.track_id,
                        "bbox": list(bbox),
                        "confidence": float(confidence)
                    })
            
            self.results.extend(frame_results)
            
            # Draw annotations if requested
            if save_annotated and out_writer:
                annotated_frame = self._draw_annotations(frame, tracks)
                out_writer.write(annotated_frame)
            
            frame_idx += 1
            if frame_idx % 100 == 0:
                logger.info(f"Processed {frame_idx}/{total_frames} frames")
        
        cap.release()
        if out_writer:
            out_writer.release()
        
        # Save results to JSON
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(self.results, f, indent=2)
            logger.info(f"Results saved to: {output_path}")
        
        return self.results
    
    def _draw_annotations(self, frame: np.ndarray, tracks: Dict[int, Track]) -> np.ndarray:
        """Draw bounding boxes and labels on frame"""
        annotated = frame.copy()
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        for i, track in enumerate(tracks.values()):
            if track.bbox_history:
                bbox = track.bbox_history[-1]
                x1, y1, x2, y2 = bbox
                color = colors[i % len(colors)]
                
                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{track.global_id} (ID: {track.track_id})"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(annotated, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description="Multi-Person Video Tracking Pipeline")
    parser.add_argument("input_video", help="Path to input video file")
    parser.add_argument("--output", "-o", help="Output JSON file path", 
                       default="tracking_results.json")
    parser.add_argument("--model", "-m", help="YOLO model path", default="yolov8n.pt")
    parser.add_argument("--conf", "-c", type=float, help="Confidence threshold", default=0.5)
    parser.add_argument("--annotated", "-a", action="store_true", 
                       help="Save annotated video")
    
    args = parser.parse_args()
    
    # Process video
    processor = VideoProcessor(args.model, args.conf)
    results = processor.process_video(args.input_video, args.output, args.annotated)
    
    # Print summary
    unique_persons = set(r["global_id"] for r in results)
    total_detections = len(results)
    
    print(f"\n=== Processing Complete ===")
    print(f"Total detections: {total_detections}")
    print(f"Unique persons detected: {len(unique_persons)}")
    print(f"Persons: {', '.join(sorted(unique_persons))}")
    print(f"Results saved to: {args.output}")
    
    if args.annotated:
        annotated_path = args.output.replace('.json', '_annotated.mp4')
        print(f"Annotated video saved to: {annotated_path}")

if __name__ == "__main__":
    main()
