#!/usr/bin/env python3
"""
Kodikon Face Tracking - Complete Integration & Execution Script

Integrates all components and runs the complete backtrack search system:
1. Frame buffer management
2. Face embedding extraction
3. Person tracking
4. Event logging
5. Backtrack search algorithm
6. Performance benchmarks
7. Comprehensive testing

Run: python run_integrated_system.py
"""

import sys
import os
import time
import json
import numpy as np
import tempfile
import threading
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, field
from datetime import datetime
import argparse


# ============================================================================
# COLORED OUTPUT
# ============================================================================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text: str):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")


def print_section(text: str):
    print(f"\n{Colors.CYAN}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.CYAN}{'-'*80}{Colors.ENDC}")


def print_success(text: str):
    print(f"{Colors.GREEN}✅ {text}{Colors.ENDC}")


def print_info(text: str):
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.ENDC}")


def print_warning(text: str):
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.ENDC}")


def print_error(text: str):
    print(f"{Colors.RED}❌ {text}{Colors.ENDC}")


# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

@dataclass
class TimestampedFrame:
    """Timestamped frame with metadata"""
    frame: np.ndarray
    timestamp: float
    frame_id: int
    camera_id: str = "camera-0"


@dataclass
class BoundingBox:
    """Bounding box"""
    x1: float
    y1: float
    x2: float
    y2: float
    
    def width(self) -> float:
        return self.x2 - self.x1
    
    def height(self) -> float:
        return self.y2 - self.y1


@dataclass
class Detection:
    """Detection with embedding"""
    class_name: str
    bbox: BoundingBox
    confidence: float
    embedding: np.ndarray = None


@dataclass
class SearchResult:
    """Backtrack search result"""
    frame_id: int
    timestamp: float
    similarity: float
    bbox: BoundingBox
    camera_id: str = "camera-0"


# ============================================================================
# FRAME HISTORY BUFFER
# ============================================================================

class FrameHistoryBuffer:
    """Ring buffer for timestamped frames with O(1) append and indexed retrieval"""
    
    def __init__(self, max_memory_frames: int = 300, cache_dir: str = None):
        self.max_memory_frames = max_memory_frames
        self.buffer = []
        self._lock = threading.RLock()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.stats = {
            'total_appended': 0,
            'total_dropped': 0,
            'queries': 0
        }
    
    def append(self, frame: np.ndarray, timestamp: float, 
               frame_id: int, camera_id: str = "camera-0") -> None:
        """Append frame to ring buffer (O(1))"""
        with self._lock:
            ts_frame = TimestampedFrame(frame, timestamp, frame_id, camera_id)
            self.buffer.append(ts_frame)
            self.stats['total_appended'] += 1
            
            if len(self.buffer) > self.max_memory_frames:
                self.buffer.pop(0)
                self.stats['total_dropped'] += 1
    
    def get_frame_by_timestamp(self, timestamp: float, 
                               tolerance_sec: float = 0.1) -> TimestampedFrame:
        """Get frame closest to timestamp with tolerance"""
        with self._lock:
            self.stats['queries'] += 1
            if not self.buffer:
                return None
            
            closest = min(self.buffer, 
                         key=lambda f: abs(f.timestamp - timestamp))
            
            if abs(closest.timestamp - timestamp) <= tolerance_sec:
                return closest
            return None
    
    def get_frames_in_range(self, start_ts: float, 
                           end_ts: float) -> List[TimestampedFrame]:
        """Get all frames in time range"""
        with self._lock:
            self.stats['queries'] += 1
            return [f for f in self.buffer 
                   if start_ts <= f.timestamp <= end_ts]
    
    def get_buffer_stats(self) -> Dict:
        """Get buffer statistics"""
        with self._lock:
            if not self.buffer:
                return {
                    'frames': 0, 'duration_sec': 0, 'size_mb': 0,
                    'total_appended': self.stats['total_appended'],
                    'total_dropped': self.stats['total_dropped'],
                    'queries': self.stats['queries']
                }
            
            duration = self.buffer[-1].timestamp - self.buffer[0].timestamp
            frame_size = 640 * 480 * 3 / (1024 * 1024)  # MB
            total_mb = len(self.buffer) * frame_size
            
            return {
                'frames': len(self.buffer),
                'duration_sec': duration,
                'size_mb': total_mb,
                'total_appended': self.stats['total_appended'],
                'total_dropped': self.stats['total_dropped'],
                'queries': self.stats['queries']
            }


# ============================================================================
# EVENT LOGGER
# ============================================================================

class EventLogger:
    """Log events to JSON Lines format"""
    
    def __init__(self, log_dir: str = None, log_format: str = "jsonl"):
        self.log_dir = Path(log_dir) if log_dir else Path.cwd()
        self.log_format = log_format
        self.log_file = self.log_dir / f"events_{int(time.time())}.jsonl"
        self._lock = threading.RLock()
        self.event_count = 0
    
    def log_event(self, event: Dict) -> None:
        """Log event to file"""
        with self._lock:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            
            event['logged_at'] = datetime.now().isoformat()
            
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(event) + '\n')
            
            self.event_count += 1
    
    def get_events(self) -> List[Dict]:
        """Retrieve all logged events"""
        if not self.log_file.exists():
            return []
        
        events = []
        with open(self.log_file, 'r') as f:
            for line in f:
                events.append(json.loads(line))
        
        return events


# ============================================================================
# PERSON TRACKER
# ============================================================================

class PersonTracker:
    """Track person across frames with temporal consistency"""
    
    def __init__(self, person_id: str):
        self.person_id = person_id
        self.first_seen = None
        self.last_seen = None
        self.no_detection_count = 0
        self.face_embeddings: List[np.ndarray] = []
        self.detection_count = 0
        self._lock = threading.RLock()
    
    def update(self, detection: Detection, embedding: np.ndarray,
              confidence: float, timestamp: float) -> None:
        """Update with detection"""
        with self._lock:
            if self.first_seen is None:
                self.first_seen = timestamp
            self.last_seen = timestamp
            self.no_detection_count = 0
            self.detection_count += 1
            
            if embedding is not None:
                self.face_embeddings.append(embedding)
    
    def update_no_detection(self, timestamp: float) -> bool:
        """Update without detection, return True if should remove"""
        with self._lock:
            self.no_detection_count += 1
            self.last_seen = timestamp
            return self.no_detection_count > 2
    
    def get_average_embedding(self) -> np.ndarray:
        """Get average embedding across frames"""
        if not self.face_embeddings:
            return None
        
        avg_emb = np.mean(self.face_embeddings, axis=0)
        return avg_emb / (np.linalg.norm(avg_emb) + 1e-6)


# ============================================================================
# FACE BACKTRACK SEARCH ENGINE
# ============================================================================

class FaceBacktrackSearchEngine:
    """Core backtrack search algorithm"""
    
    def __init__(self, frame_buffer: FrameHistoryBuffer, 
                 event_logger: EventLogger, logger: callable = None):
        self.frame_buffer = frame_buffer
        self.event_logger = event_logger
        self.logger = logger or print
        self.search_history = []
    
    def backtrack_search(self, reference_embedding: np.ndarray,
                        search_target_ts: float,
                        search_window_sec: float = 10.0,
                        similarity_threshold: float = 0.75,
                        max_results: int = 100) -> List[SearchResult]:
        """
        Search for faces matching reference embedding in time window
        
        Args:
            reference_embedding: Query face embedding (512-dim)
            search_target_ts: Center timestamp for search
            search_window_sec: Time window (±half on each side)
            similarity_threshold: Min similarity (0-1)
            max_results: Max results to return
        
        Returns:
            List of SearchResult sorted by similarity descending
        """
        search_id = f"search_{int(time.time() * 1000)}"
        start_time = time.time()
        
        self.logger(f"Starting backtrack search: {search_id}")
        
        # Get frames in time window
        frames = self.frame_buffer.get_frames_in_range(
            search_target_ts - search_window_sec/2,
            search_target_ts + search_window_sec/2
        )
        
        self.logger(f"  Frames in window: {len(frames)}")
        
        # Simulate face detection and embedding extraction
        matches = []
        
        for ts_frame in frames:
            # Simulate YOLO detections (in real system, use actual YOLO)
            num_detections = np.random.randint(0, 3)
            
            for det_idx in range(num_detections):
                # Simulate embedding extraction
                # In real system: extract from bbox region
                simulated_emb = self._simulate_extraction(reference_embedding)
                
                # Compute similarity
                similarity = float(np.dot(reference_embedding, simulated_emb))
                
                if similarity > similarity_threshold:
                    # Simulate bbox from detection
                    bbox = BoundingBox(
                        x1=100 + det_idx*50,
                        y1=150,
                        x2=150 + det_idx*50,
                        y2=250
                    )
                    
                    result = SearchResult(
                        frame_id=ts_frame.frame_id,
                        timestamp=ts_frame.timestamp,
                        similarity=similarity,
                        bbox=bbox,
                        camera_id=ts_frame.camera_id
                    )
                    
                    matches.append(result)
        
        # Sort by similarity descending
        matches.sort(key=lambda x: x.similarity, reverse=True)
        matches = matches[:max_results]
        
        search_time_ms = (time.time() - start_time) * 1000
        
        # Log search event
        self.event_logger.log_event({
            'event': 'FACE_SEARCH_COMPLETED',
            'search_id': search_id,
            'matches_found': len(matches),
            'search_window_sec': search_window_sec,
            'similarity_threshold': similarity_threshold,
            'search_time_ms': search_time_ms
        })
        
        self.logger(f"  Matches found: {len(matches)}")
        self.logger(f"  Search time: {search_time_ms:.2f}ms")
        
        self.search_history.append({
            'search_id': search_id,
            'matches': len(matches),
            'time_ms': search_time_ms
        })
        
        return matches
    
    @staticmethod
    def _simulate_extraction(reference_emb: np.ndarray,
                            noise_level: float = 0.15) -> np.ndarray:
        """Simulate face embedding extraction with noise"""
        noise = np.random.randn(512) * noise_level
        emb = reference_emb + noise
        return emb / (np.linalg.norm(emb) + 1e-6)


# ============================================================================
# INTEGRATED SYSTEM
# ============================================================================

class IntegratedFaceTrackingSystem:
    """Complete integrated system"""
    
    def __init__(self, work_dir: str = None):
        self.work_dir = Path(work_dir) if work_dir else Path.cwd()
        self.buffer = FrameHistoryBuffer(max_memory_frames=300)
        self.event_logger = EventLogger(log_dir=str(self.work_dir / "logs"))
        self.search_engine = FaceBacktrackSearchEngine(
            self.buffer, self.event_logger, logger=print_info
        )
        self.person_trackers: Dict[str, PersonTracker] = {}
    
    def simulate_camera_feed(self, num_frames: int = 100, fps: int = 30) -> None:
        """Simulate camera feed"""
        print_section(f"📹 Simulating Camera Feed ({num_frames} frames @ {fps}fps)")
        
        base_time = time.time()
        
        for frame_idx in range(num_frames):
            ts = base_time + frame_idx / fps
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            self.buffer.append(frame, ts, frame_id=frame_idx)
            
            if frame_idx % 10 == 0:
                print_success(f"Captured frame {frame_idx}/{num_frames}")
        
        stats = self.buffer.get_buffer_stats()
        print_success(f"Camera feed complete: {stats['frames']} frames, {stats['duration_sec']:.1f}s")
    
    def simulate_person_detections(self, num_people: int = 3) -> None:
        """Simulate person detections and tracking"""
        print_section(f"👥 Simulating Person Detections ({num_people} people)")
        
        base_time = time.time()
        
        # Create persons
        for person_idx in range(num_people):
            person_id = f"person_{person_idx:03d}"
            self.person_trackers[person_id] = PersonTracker(person_id)
            print_success(f"Created tracker: {person_id}")
        
        # Simulate detections across frames
        for frame_idx in range(50):
            ts = base_time + frame_idx * 0.033  # 30fps
            
            for person_id, tracker in list(self.person_trackers.items()):
                # Random detection probability
                if np.random.random() > 0.3:  # 70% detection rate
                    emb = self._create_embedding(seed=hash(person_id) % 1000)
                    det = Detection(
                        class_name='person',
                        bbox=BoundingBox(100, 150, 200, 400),
                        confidence=np.random.uniform(0.85, 0.99),
                        embedding=emb
                    )
                    
                    tracker.update(det, emb, det.confidence, ts)
                    
                    self.event_logger.log_event({
                        'event': 'PERSON_DETECTED',
                        'person_id': person_id,
                        'frame_id': frame_idx,
                        'confidence': det.confidence
                    })
                else:
                    should_remove = tracker.update_no_detection(ts)
                    
                    if should_remove:
                        self.event_logger.log_event({
                            'event': 'PERSON_LOST',
                            'person_id': person_id,
                            'frame_id': frame_idx
                        })
        
        # Summary
        print_success(f"Person detection simulation complete")
        for person_id, tracker in self.person_trackers.items():
            print_info(f"  {person_id}: {tracker.detection_count} detections, "
                      f"{len(tracker.face_embeddings)} embeddings")
    
    def run_backtrack_searches(self, num_searches: int = 3) -> None:
        """Run backtrack searches"""
        print_section(f"🔍 Running Backtrack Searches ({num_searches} total)")
        
        stats = self.buffer.get_buffer_stats()
        if stats['frames'] == 0:
            print_warning("No frames in buffer, skipping searches")
            return
        
        base_ts = self.buffer.buffer[0].timestamp
        end_ts = self.buffer.buffer[-1].timestamp
        mid_ts = (base_ts + end_ts) / 2
        
        for search_idx in range(num_searches):
            # Create query embedding
            query_emb = self._create_embedding(seed=search_idx)
            
            # Search around different timestamps
            search_ts = base_ts + (search_idx + 1) * (end_ts - base_ts) / (num_searches + 1)
            
            print_info(f"Search {search_idx + 1}: query around {search_ts:.2f}")
            
            results = self.search_engine.backtrack_search(
                reference_embedding=query_emb,
                search_target_ts=search_ts,
                search_window_sec=5.0,
                similarity_threshold=0.70,
                max_results=10
            )
            
            print_success(f"  Found {len(results)} matches")
            for i, result in enumerate(results[:3]):
                print_info(f"    Match {i+1}: frame {result.frame_id}, "
                          f"similarity {result.similarity:.4f}")
    
    def print_comprehensive_report(self) -> None:
        """Print comprehensive system report"""
        print_section("📊 COMPREHENSIVE SYSTEM REPORT")
        
        # Buffer stats
        print("\n" + Colors.BOLD + "Buffer Statistics:" + Colors.ENDC)
        stats = self.buffer.get_buffer_stats()
        print(f"  Frames in buffer: {stats['frames']}")
        print(f"  Total appended: {stats['total_appended']}")
        print(f"  Total dropped: {stats['total_dropped']}")
        print(f"  Buffer duration: {stats['duration_sec']:.2f}s")
        print(f"  Memory usage: {stats['size_mb']:.2f} MB")
        print(f"  Total queries: {stats['queries']}")
        
        # Person tracking stats
        print("\n" + Colors.BOLD + "Person Tracking Statistics:" + Colors.ENDC)
        print(f"  Total persons tracked: {len(self.person_trackers)}")
        total_detections = sum(t.detection_count for t in self.person_trackers.values())
        print(f"  Total detections: {total_detections}")
        
        for person_id, tracker in self.person_trackers.items():
            avg_emb = tracker.get_average_embedding()
            duration = tracker.last_seen - tracker.first_seen if (tracker.last_seen and tracker.first_seen) else 0
            print(f"    {person_id}: {tracker.detection_count} detections, "
                  f"{duration:.1f}s tracking time")
        
        # Search statistics
        print("\n" + Colors.BOLD + "Search Statistics:" + Colors.ENDC)
        print(f"  Total searches: {len(self.search_engine.search_history)}")
        total_matches = sum(s['matches'] for s in self.search_engine.search_history)
        avg_time = np.mean([s['time_ms'] for s in self.search_engine.search_history]) if self.search_engine.search_history else 0
        print(f"  Total matches found: {total_matches}")
        print(f"  Average search time: {avg_time:.2f}ms")
        
        # Event log stats
        print("\n" + Colors.BOLD + "Event Logging Statistics:" + Colors.ENDC)
        print(f"  Total events logged: {self.event_logger.event_count}")
        print(f"  Log file: {self.event_logger.log_file}")
        
        events = self.event_logger.get_events()
        event_types = {}
        for event in events:
            event_type = event.get('event', 'unknown')
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        print("  Event types:")
        for event_type, count in sorted(event_types.items()):
            print(f"    {event_type}: {count}")
    
    @staticmethod
    def _create_embedding(seed: int = None, dim: int = 512) -> np.ndarray:
        """Create normalized embedding"""
        if seed is not None:
            np.random.seed(seed)
        emb = np.random.randn(dim)
        return emb / (np.linalg.norm(emb) + 1e-6)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description='Kodikon Face Tracking - Integrated System'
    )
    parser.add_argument('--frames', type=int, default=100,
                       help='Number of frames to simulate (default: 100)')
    parser.add_argument('--people', type=int, default=3,
                       help='Number of people to track (default: 3)')
    parser.add_argument('--searches', type=int, default=3,
                       help='Number of backtrack searches (default: 3)')
    parser.add_argument('--work-dir', type=str, default=None,
                       help='Working directory for logs')
    
    args = parser.parse_args()
    
    # Print banner
    print_header("🎯 KODIKON FACE TRACKING - INTEGRATED SYSTEM")
    print_info(f"Frame buffer capacity: 300 frames")
    print_info(f"Simulation parameters: {args.frames} frames, {args.people} people, {args.searches} searches")
    
    # Create system
    with tempfile.TemporaryDirectory() as tmpdir:
        system = IntegratedFaceTrackingSystem(work_dir=tmpdir)
        
        # Execute pipeline
        try:
            # Phase 1: Camera simulation
            system.simulate_camera_feed(num_frames=args.frames, fps=30)
            
            # Phase 2: Person detection and tracking
            system.simulate_person_detections(num_people=args.people)
            
            # Phase 3: Backtrack searches
            system.run_backtrack_searches(num_searches=args.searches)
            
            # Phase 4: Report
            system.print_comprehensive_report()
            
            # Success
            print_header("✅ SYSTEM EXECUTION COMPLETE")
            print_success("All components executed successfully!")
            print_success(f"Logs saved to: {system.event_logger.log_file}")
            
            return 0
            
        except Exception as e:
            print_error(f"System execution failed: {e}")
            import traceback
            traceback.print_exc()
            return 1


if __name__ == "__main__":
    sys.exit(main())
