"""
Face Tracking Backtrack Search - Standalone Test Suite

Comprehensive testing without requiring integrated_runtime/vision module imports.
Tests core algorithms using mock data and utilities.
"""

import numpy as np
import time
import json
from pathlib import Path
from datetime import datetime
import tempfile
import threading
from typing import List, Dict, Tuple
from dataclasses import dataclass

# ============================================================================
# MOCK DATA STRUCTURES
# ============================================================================

@dataclass
class TimestampedFrame:
    """Mock timestamped frame"""
    frame: np.ndarray
    timestamp: float
    frame_id: int
    camera_id: str = "camera-0"


@dataclass
class BoundingBox:
    """Mock bounding box"""
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
    """Mock detection"""
    class_name: str
    bbox: BoundingBox
    confidence: float
    embedding: np.ndarray = None


# ============================================================================
# FRAME HISTORY BUFFER IMPLEMENTATION
# ============================================================================

class FrameHistoryBuffer:
    """Ring buffer for timestamped frame storage"""
    
    def __init__(self, max_memory_frames: int = 300, cache_dir: str = None):
        self.max_memory_frames = max_memory_frames
        self.buffer = []
        self._lock = threading.RLock()
        self.cache_dir = Path(cache_dir) if cache_dir else None
    
    def append(self, frame: np.ndarray, timestamp: float, 
               frame_id: int, camera_id: str = "camera-0"):
        """Append frame to buffer"""
        with self._lock:
            ts_frame = TimestampedFrame(frame, timestamp, frame_id, camera_id)
            self.buffer.append(ts_frame)
            
            # Ring buffer: remove oldest if exceeds max
            if len(self.buffer) > self.max_memory_frames:
                self.buffer.pop(0)
    
    def get_frame_by_timestamp(self, timestamp: float, 
                               tolerance_sec: float = 0.1) -> TimestampedFrame:
        """Get frame closest to timestamp"""
        with self._lock:
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
            return [f for f in self.buffer 
                   if start_ts <= f.timestamp <= end_ts]
    
    def get_buffer_stats(self) -> Dict:
        """Get buffer statistics"""
        with self._lock:
            if not self.buffer:
                return {'frames': 0, 'duration_sec': 0, 'size_mb': 0}
            
            duration = self.buffer[-1].timestamp - self.buffer[0].timestamp
            
            # Estimate memory (frame is typically 640x480x3 bytes)
            frame_size = 640 * 480 * 3 / (1024 * 1024)  # MB
            total_mb = len(self.buffer) * frame_size
            
            return {
                'frames': len(self.buffer),
                'duration_sec': duration,
                'size_mb': total_mb
            }


# ============================================================================
# EVENT LOGGER IMPLEMENTATION
# ============================================================================

class EventLogger:
    """Log events to JSON Lines format"""
    
    def __init__(self, log_dir: str = None, log_format: str = "jsonl"):
        self.log_dir = Path(log_dir) if log_dir else Path.cwd()
        self.log_format = log_format
        self.log_file = self.log_dir / f"events_{int(time.time())}.jsonl"
        self._lock = threading.RLock()
    
    def log_event(self, event: Dict):
        """Log event to file"""
        with self._lock:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(event) + '\n')


# ============================================================================
# PERSON TRACKER IMPLEMENTATION
# ============================================================================

class PersonTracker:
    """Track person across frames"""
    
    def __init__(self, person_id: str):
        self.person_id = person_id
        self.first_seen = None
        self.last_seen = None
        self.no_detection_count = 0
        self.face_embeddings = []
        self._lock = threading.RLock()
    
    def update(self, detection: Detection, embedding: np.ndarray,
              confidence: float, timestamp: float):
        """Update with detection"""
        with self._lock:
            if self.first_seen is None:
                self.first_seen = timestamp
            self.last_seen = timestamp
            self.no_detection_count = 0
            
            if embedding is not None:
                self.face_embeddings.append(embedding)
    
    def update_no_detection(self, timestamp: float) -> bool:
        """Update without detection, return True if should remove"""
        with self._lock:
            self.no_detection_count += 1
            self.last_seen = timestamp
            
            # Remove if not detected for 2 consecutive frames
            return self.no_detection_count > 2


# ============================================================================
# TEST FIXTURES
# ============================================================================

def create_dummy_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Create dummy frame"""
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)


def create_dummy_embedding(seed: int = None, dim: int = 512) -> np.ndarray:
    """Create normalized embedding"""
    if seed is not None:
        np.random.seed(seed)
    emb = np.random.randn(dim)
    emb = emb / (np.linalg.norm(emb) + 1e-6)
    return emb


def create_detection(x1: int = 50, y1: int = 50, 
                    x2: int = 150, y2: int = 200) -> Detection:
    """Create mock detection"""
    bbox = BoundingBox(float(x1), float(y1), float(x2), float(y2))
    return Detection(
        class_name='person',
        bbox=bbox,
        confidence=0.95,
        embedding=create_dummy_embedding()
    )


# ============================================================================
# TEST 1-3: FRAME BUFFER
# ============================================================================

def test_frame_buffer_append_retrieve():
    """Test 1: Append and retrieve frames"""
    print("\n" + "="*80)
    print("TEST 1: Frame Buffer - Append & Retrieve")
    print("="*80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        buffer = FrameHistoryBuffer(max_memory_frames=10, cache_dir=tmpdir)
        
        base_time = time.time()
        for i in range(5):
            frame = create_dummy_frame()
            ts = base_time + i * 0.1
            buffer.append(frame, ts, frame_id=i)
        
        retrieved = buffer.get_frame_by_timestamp(base_time + 0.2, tolerance_sec=0.15)
        
        assert retrieved is not None, "Failed to retrieve frame"
        assert retrieved.frame_id == 2, f"Expected frame_id 2, got {retrieved.frame_id}"
        
        stats = buffer.get_buffer_stats()
        assert stats['frames'] == 5, f"Expected 5 frames, got {stats['frames']}"
        
        print(f"✅ Appended: {len([1,2,3,4,5])} frames")
        print(f"✅ Retrieved frame_id: {retrieved.frame_id}")
        print(f"✅ Buffer stats: {stats['frames']} frames")
        print(f"✅ TEST 1 PASSED")


def test_frame_buffer_range_query():
    """Test 2: Range queries"""
    print("\n" + "="*80)
    print("TEST 2: Frame Buffer - Range Query")
    print("="*80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        buffer = FrameHistoryBuffer(max_memory_frames=20, cache_dir=tmpdir)
        
        base_time = time.time()
        for i in range(10):
            frame = create_dummy_frame()
            ts = base_time + i * 0.05
            buffer.append(frame, ts, frame_id=i)
        
        start_ts = base_time + 0.1
        end_ts = base_time + 0.25
        frames = buffer.get_frames_in_range(start_ts, end_ts)
        
        assert len(frames) > 0, "Range query returned no frames"
        assert all(start_ts <= f.timestamp <= end_ts for f in frames)
        
        print(f"✅ Range query: [{start_ts:.3f}, {end_ts:.3f}]")
        print(f"✅ Retrieved {len(frames)} frames from 10 total")
        print(f"✅ TEST 2 PASSED")


def test_frame_buffer_ring_behavior():
    """Test 3: Ring buffer overflow"""
    print("\n" + "="*80)
    print("TEST 3: Frame Buffer - Ring Behavior")
    print("="*80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        buffer = FrameHistoryBuffer(max_memory_frames=5, cache_dir=tmpdir)
        
        base_time = time.time()
        for i in range(8):
            frame = create_dummy_frame()
            ts = base_time + i * 0.1
            buffer.append(frame, ts, frame_id=i)
        
        stats = buffer.get_buffer_stats()
        assert stats['frames'] == 5, f"Expected 5 frames (ring max), got {stats['frames']}"
        
        oldest_frame = buffer.buffer[0]
        assert oldest_frame.frame_id >= 3, f"Expected oldest >= 3, got {oldest_frame.frame_id}"
        
        print(f"✅ Appended 8 frames to 5-frame buffer")
        print(f"✅ Ring buffer kept last {stats['frames']} frames")
        print(f"✅ Oldest frame_id: {oldest_frame.frame_id}")
        print(f"✅ TEST 3 PASSED")


# ============================================================================
# TEST 4-6: SIMILARITY & TRACKING
# ============================================================================

def test_cosine_similarity():
    """Test 4: Cosine similarity"""
    print("\n" + "="*80)
    print("TEST 4: Cosine Similarity Matching")
    print("="*80)
    
    np.random.seed(42)
    ref_emb = create_dummy_embedding(seed=42)
    similar_emb = create_dummy_embedding(seed=42)
    
    np.random.seed(99)
    different_emb = np.random.randn(512)
    different_emb = different_emb / (np.linalg.norm(different_emb) + 1e-6)
    
    sim_same = float(np.dot(ref_emb, similar_emb))
    sim_diff = float(np.dot(ref_emb, different_emb))
    
    print(f"✅ Similar embedding similarity: {sim_same:.4f} (should be ~1.0)")
    print(f"✅ Different embedding similarity: {sim_diff:.4f} (should be ~0.0)")
    
    assert sim_same > 0.9, f"Expected high similarity, got {sim_same}"
    assert sim_diff < 0.1, f"Expected low similarity, got {sim_diff}"
    
    print(f"✅ TEST 4 PASSED")


def test_person_tracker():
    """Test 5: Person tracker"""
    print("\n" + "="*80)
    print("TEST 5: Person Tracker State Machine")
    print("="*80)
    
    tracker = PersonTracker("person_001")
    
    det = create_detection()
    emb = create_dummy_embedding()
    ts = time.time()
    
    tracker.update(det, emb, 0.9, ts)
    
    assert tracker.last_seen == ts
    assert tracker.no_detection_count == 0
    assert len(tracker.face_embeddings) == 1
    
    print(f"✅ Tracker initialized: {tracker.person_id}")
    print(f"✅ Updated with detection")
    print(f"✅ Face embeddings stored: {len(tracker.face_embeddings)}")
    
    new_ts = ts + 1.0
    result = tracker.update_no_detection(new_ts)
    
    print(f"✅ After no-detection: should_remove={result}")
    print(f"✅ TEST 5 PASSED")


def test_event_logging():
    """Test 6: Event logging"""
    print("\n" + "="*80)
    print("TEST 6: Event Logger")
    print("="*80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = EventLogger(log_dir=tmpdir)
        
        events = [
            {'event': 'PERSON_IN', 'person_id': 'p_001', 'timestamp': time.time()},
            {'event': 'PERSON_OUT', 'person_id': 'p_001', 'timestamp': time.time() + 10},
            {'event': 'FACE_MATCHED', 'search_id': 's_001', 'similarity': 0.89},
        ]
        
        for event in events:
            logger.log_event(event)
        
        log_files = list(Path(tmpdir).glob("events_*.jsonl"))
        assert len(log_files) > 0, "No log files created"
        
        log_file = log_files[0]
        logged_events = []
        
        with open(log_file, 'r') as f:
            for line in f:
                logged_events.append(json.loads(line))
        
        assert len(logged_events) == 3, f"Expected 3 events, got {len(logged_events)}"
        
        print(f"✅ Logged {len(logged_events)} events")
        print(f"✅ Event types: {[e['event'] for e in logged_events]}")
        print(f"✅ TEST 6 PASSED")


# ============================================================================
# TEST 7-8: BACKTRACK & PERFORMANCE
# ============================================================================

def test_backtrack_search_simulation():
    """Test 7: Backtrack search algorithm"""
    print("\n" + "="*80)
    print("TEST 7: Simulated Backtrack Search")
    print("="*80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        buffer = FrameHistoryBuffer(max_memory_frames=20, cache_dir=tmpdir)
        
        base_time = time.time()
        
        # Add frames with face embeddings
        for frame_idx in range(10):
            frame = create_dummy_frame()
            ts = base_time + frame_idx * 0.1
            buffer.append(frame, ts, frame_id=frame_idx)
        
        # Search: find frames near middle
        target_ts = base_time + 0.5
        search_window = 0.3
        similarity_threshold = 0.7
        
        reference_emb = create_dummy_embedding(seed=42)
        
        matches = []
        frames_in_range = buffer.get_frames_in_range(
            target_ts - search_window/2,
            target_ts + search_window/2
        )
        
        for ts_frame in frames_in_range:
            # Simulate detections
            emb = create_dummy_embedding(seed=42)
            similarity = float(np.dot(reference_emb, emb))
            
            if similarity > similarity_threshold:
                matches.append({
                    'frame_id': ts_frame.frame_id,
                    'timestamp': ts_frame.timestamp,
                    'similarity': similarity
                })
        
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        print(f"✅ Populated buffer with 10 frames")
        print(f"✅ Search window: ±{search_window/2:.1f}s")
        print(f"✅ Found {len(matches)} matching frames")
        
        assert len(matches) > 0, "No matches found"
        print(f"✅ TEST 7 PASSED")


def test_performance_benchmark():
    """Test 8: Performance benchmarks"""
    print("\n" + "="*80)
    print("TEST 8: Performance Benchmark (300 frames)")
    print("="*80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        buffer = FrameHistoryBuffer(max_memory_frames=300, cache_dir=tmpdir)
        
        base_time = time.time()
        
        print("📊 Populating buffer with 300 frames...")
        start_populate = time.time()
        
        for i in range(300):
            frame = create_dummy_frame()
            ts = base_time + i / 30.0  # 30fps
            buffer.append(frame, ts, frame_id=i)
        
        populate_time_ms = (time.time() - start_populate) * 1000
        
        stats = buffer.get_buffer_stats()
        print(f"✅ Populated {stats['frames']} frames in {populate_time_ms:.2f}ms")
        print(f"✅ Buffer duration: {stats['duration_sec']:.2f}s")
        
        # Benchmark retrieve
        print("\n📊 Benchmarking frame retrieval...")
        
        target_ts = base_time + 5.0
        start_retrieve = time.time()
        
        for _ in range(100):
            frame = buffer.get_frame_by_timestamp(target_ts, tolerance_sec=0.1)
        
        retrieve_time_us = (time.time() - start_retrieve) * 1000000 / 100
        
        print(f"✅ Average retrieve time: {retrieve_time_us:.2f}µs")
        
        # Benchmark range query
        print("\n📊 Benchmarking range queries...")
        
        start_range = time.time()
        
        for _ in range(100):
            frames = buffer.get_frames_in_range(base_time + 2.0, base_time + 8.0)
        
        range_time_ms = (time.time() - start_range) * 1000 / 100
        
        print(f"✅ Average range query: {range_time_ms:.3f}ms")
        print(f"✅ TEST 8 PASSED")


# ============================================================================
# TEST 9: CONCURRENCY
# ============================================================================

def test_concurrent_access():
    """Test 9: Thread safety"""
    print("\n" + "="*80)
    print("TEST 9: Concurrent Access (Thread Safety)")
    print("="*80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        buffer = FrameHistoryBuffer(max_memory_frames=100, cache_dir=tmpdir)
        
        errors = []
        base_time = time.time()
        
        def writer_thread():
            try:
                for i in range(50):
                    frame = create_dummy_frame()
                    ts = base_time + i * 0.01
                    buffer.append(frame, ts, frame_id=i)
            except Exception as e:
                errors.append(f"Writer: {e}")
        
        def reader_thread():
            try:
                for i in range(50):
                    frame = buffer.get_frame_by_timestamp(base_time + i * 0.01)
            except Exception as e:
                errors.append(f"Reader: {e}")
        
        threads = [
            threading.Thread(target=writer_thread),
            threading.Thread(target=writer_thread),
            threading.Thread(target=reader_thread),
            threading.Thread(target=reader_thread),
        ]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        stats = buffer.get_buffer_stats()
        
        print(f"✅ Ran 2 writer + 2 reader threads")
        print(f"✅ Final buffer: {stats['frames']} frames")
        
        if errors:
            print(f"❌ Errors: {errors}")
            assert False, "Concurrent access failed"
        
        print(f"✅ No race conditions detected")
        print(f"✅ TEST 9 PASSED")


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all tests"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "  FACE TRACKING BACKTRACK SEARCH - TEST SUITE (STANDALONE)".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    
    tests = [
        ("Frame Buffer", [
            test_frame_buffer_append_retrieve,
            test_frame_buffer_range_query,
            test_frame_buffer_ring_behavior,
        ]),
        ("Similarity & Tracking", [
            test_cosine_similarity,
            test_person_tracker,
            test_event_logging,
        ]),
        ("Search & Performance", [
            test_backtrack_search_simulation,
            test_performance_benchmark,
        ]),
        ("Concurrency", [
            test_concurrent_access,
        ]),
    ]
    
    passed = 0
    failed = 0
    
    for category, test_list in tests:
        print(f"\n\n{'='*80}")
        print(f"CATEGORY: {category}")
        print(f"{'='*80}")
        
        for test_func in test_list:
            try:
                test_func()
                passed += 1
            except Exception as e:
                failed += 1
                print(f"\n❌ TEST FAILED: {test_func.__name__}")
                print(f"   Error: {e}")
                import traceback
                traceback.print_exc()
    
    # Summary
    print("\n\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print(f"║  TEST RESULTS: {passed} PASSED, {failed} FAILED".ljust(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝\n")
    
    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
