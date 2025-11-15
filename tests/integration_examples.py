"""
Backtrack Search - Integration Examples & Usage Patterns

Demonstrates how to use backtrack search in real scenarios:
1. Live search during streaming
2. Offline search from buffer
3. Mesh-based distributed search
4. Performance optimization strategies
"""

import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import tempfile

# Import components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Use standalone implementations
from tests.test_backtrack_search_standalone import (
    FrameHistoryBuffer, EventLogger, PersonTracker
)
from tests.mock_data_generator import (
    SyntheticEmbeddingGenerator,
    SyntheticYOLODetectionGenerator,
    MockMeshMessage,
    TestScenarioGenerator
)


# ============================================================================
# EXAMPLE 1: BASIC BACKTRACK SEARCH
# ============================================================================

def example_basic_backtrack_search():
    """
    Example: Search for a specific person by face embedding
    
    Use case: User asks "Find person with blue shirt from 2 minutes ago"
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Backtrack Search")
    print("="*80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize components
        buffer = FrameHistoryBuffer(max_memory_frames=300, cache_dir=tmpdir)
        logger = EventLogger(log_dir=tmpdir)
        
        # Simulate: 300 frames captured (10 seconds @ 30fps)
        print("📹 Simulating 10 seconds of video capture...")
        base_time = time.time()
        
        for frame_idx in range(300):
            ts = base_time + frame_idx / 30.0
            
            # Create dummy frame
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            buffer.append(frame, ts, frame_id=frame_idx, camera_id="camera-0")
            
            # Log occasional person detections
            if frame_idx % 15 == 0:
                logger.log_event({
                    'event': 'PERSON_DETECTED',
                    'timestamp': ts,
                    'frame_id': frame_idx,
                    'person_id': 'person-123'
                })
        
        print(f"✅ Captured 300 frames ({buffer.get_buffer_stats()['duration_sec']:.1f}s)")
        
        # Backtrack search: Find frames around t=5 seconds
        print("\n🔍 Searching for person from ~5 seconds ago...")
        search_target_ts = base_time + 5.0
        search_window = 2.0  # ±1 second
        
        search_start = time.time()
        frames = buffer.get_frames_in_range(
            search_target_ts - search_window/2,
            search_target_ts + search_window/2
        )
        search_time_ms = (time.time() - search_start) * 1000
        
        print(f"✅ Search completed in {search_time_ms:.2f}ms")
        print(f"✅ Found {len(frames)} frames in ±{search_window/2:.1f}s window")
        
        if frames:
            for frame in frames[:3]:
                print(f"   - Frame {frame.frame_id} @ {frame.timestamp:.3f}")


# ============================================================================
# EXAMPLE 2: FACE MATCHING WITH EMBEDDINGS
# ============================================================================

def example_face_matching_with_embeddings():
    """
    Example: Match a query face embedding against historical faces
    
    Use case: "Find all instances of person with this face"
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Face Matching with Embeddings")
    print("="*80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        buffer = FrameHistoryBuffer(max_memory_frames=100, cache_dir=tmpdir)
        
        # Simulate: 100 frames with face detections
        print("📹 Simulating video with face detections...")
        base_time = time.time()
        
        # Create reference person embedding
        reference_emb = SyntheticEmbeddingGenerator.create_embedding(seed=42)
        
        # Store embeddings per frame
        frame_embeddings = {}
        
        for frame_idx in range(100):
            ts = base_time + frame_idx / 30.0
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            buffer.append(frame, ts, frame_id=frame_idx, camera_id="camera-0")
            
            # Randomly add face embeddings (simulate YOLO detections)
            if frame_idx % 3 == 0:  # Face detected every ~3 frames
                # Create embeddings similar to reference (same person)
                embeddings = SyntheticEmbeddingGenerator.create_similar_embeddings(
                    base_seed=42, noise_level=0.05, count=1
                )
                frame_embeddings[frame_idx] = embeddings
        
        print(f"✅ Simulated {len(frame_embeddings)} frames with faces")
        
        # Search: Match reference embedding
        print("\n🔍 Matching query embedding against historical faces...")
        
        threshold = 0.75
        matches = []
        
        start_search = time.time()
        
        # Check all frames
        for frame_idx, face_embeddings in frame_embeddings.items():
            for face_emb in face_embeddings:
                similarity = float(np.dot(reference_emb, face_emb))
                
                if similarity > threshold:
                    frame = buffer.get_frame_by_timestamp(
                        base_time + frame_idx / 30.0, tolerance_sec=0.05
                    )
                    if frame:
                        matches.append({
                            'frame_id': frame_idx,
                            'timestamp': frame.timestamp,
                            'similarity': similarity
                        })
        
        search_time_ms = (time.time() - start_search) * 1000
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        print(f"✅ Search completed in {search_time_ms:.2f}ms")
        print(f"✅ Found {len(matches)} matching faces (threshold={threshold})")
        
        for i, match in enumerate(matches[:5]):
            print(f"   Match {i+1}: similarity={match['similarity']:.4f} @ frame {match['frame_id']}")


# ============================================================================
# EXAMPLE 3: PERSON TRACKING ACROSS FRAMES
# ============================================================================

def example_person_tracking():
    """
    Example: Track person across multiple frames
    
    Use case: Maintain consistent person ID across occlusions and gaps
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Person Tracking Across Frames")
    print("="*80)
    
    # Create tracker for a person
    tracker = PersonTracker("person_001")
    
    print(f"🧑 Tracking: {tracker.person_id}")
    
    base_time = time.time()
    
    # Simulate: Person visible in 20 frames with 5-frame gap
    print("\n📹 Frame sequence: [VISIBLE 10 frames] [GAP 5 frames] [VISIBLE 10 frames]")
    
    # Phase 1: Person visible (frames 0-9)
    print("\n📍 Phase 1: Person visible (frames 0-9)")
    for i in range(10):
        det = SyntheticYOLODetectionGenerator.create_person_detection(person_idx=0)
        emb = SyntheticEmbeddingGenerator.create_embedding(seed=42)
        ts = base_time + i * 0.033  # 30fps
        
        tracker.update(det, emb, confidence=0.95, timestamp=ts)
        print(f"   Frame {i}: Confidence=0.95, Embeddings stored={len(tracker.face_embeddings)}")
    
    # Phase 2: Person occluded (frames 10-14)
    print("\n📍 Phase 2: Person occluded (frames 10-14)")
    for i in range(10, 15):
        ts = base_time + i * 0.033
        result = tracker.update_no_detection(ts)
        print(f"   Frame {i}: No detection, should_remove={result}")
    
    # Phase 3: Person visible again (frames 15-24)
    print("\n📍 Phase 3: Person visible again (frames 15-24)")
    for i in range(15, 25):
        det = SyntheticYOLODetectionGenerator.create_person_detection(person_idx=0)
        emb = SyntheticEmbeddingGenerator.create_embedding(seed=42)
        ts = base_time + i * 0.033
        
        tracker.update(det, emb, confidence=0.92, timestamp=ts)
        print(f"   Frame {i}: Confidence=0.92, Embeddings stored={len(tracker.face_embeddings)}")
    
    # Summary
    print(f"\n✅ Tracking summary:")
    print(f"   Total embeddings collected: {len(tracker.face_embeddings)}")
    print(f"   Duration: {tracker.last_seen - tracker.first_seen:.2f}s")
    print(f"   Status: Tracked person survived occlusion and gap")


# ============================================================================
# EXAMPLE 4: DISTRIBUTED MESH SEARCH
# ============================================================================

def example_mesh_search():
    """
    Example: Broadcast search request to mesh nodes
    
    Use case: Search across multiple cameras (distributed system)
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Distributed Mesh Search")
    print("="*80)
    
    # Create query
    query_embedding = SyntheticEmbeddingGenerator.create_embedding(seed=42)
    search_id = f"search_{int(time.time() * 1000)}"
    
    print(f"🌐 Distributed search request:")
    print(f"   Search ID: {search_id}")
    print(f"   Query type: Face embedding match")
    print(f"   Similarity threshold: 0.75")
    
    # Create FACE_SEARCH_REQUEST message
    msg = MockMeshMessage(
        msg_type=MockMeshMessage.FACE_SEARCH_REQUEST,
        data={
            'search_id': search_id,
            'query_embedding': query_embedding.tolist()[:10],  # Truncate for display
            'similarity_threshold': 0.75,
            'search_window_sec': 30.0,
            'max_results': 100
        },
        sender_id='node-0'
    )
    
    print(f"📤 Message to broadcast to mesh:")
    print(f"   Type: {msg.msg_type}")
    print(f"   Sender: {msg.sender_id}")
    print(f"   Timestamp: {msg.timestamp:.3f}")
    
    # Simulate responses from 3 nodes
    print(f"\n📥 Simulated responses from mesh nodes:")
    
    for node_id in range(3):
        response = MockMeshMessage(
            msg_type=MockMeshMessage.FACE_SEARCH_RESULT,
            data={
                'search_id': search_id,
                'node_id': f'node-{node_id}',
                'matches_found': np.random.randint(2, 8),
                'search_time_ms': np.random.uniform(50, 200)
            },
            sender_id=f'node-{node_id}'
        )
        
        print(f"   Node-{node_id}: {response.data['matches_found']} matches, "
              f"{response.data['search_time_ms']:.1f}ms")
    
    print(f"\n✅ Mesh search completed across 3 nodes")


# ============================================================================
# EXAMPLE 5: PERFORMANCE OPTIMIZATION
# ============================================================================

def example_performance_optimization():
    """
    Example: Optimize backtrack search performance
    
    Use case: Reduce latency for real-time search
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Performance Optimization")
    print("="*80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Strategy 1: Smaller buffer for faster search
        print("\n📊 Strategy 1: Smaller buffer (recent frames only)")
        buffer_small = FrameHistoryBuffer(max_memory_frames=100, cache_dir=tmpdir)
        
        base_time = time.time()
        for i in range(100):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            buffer_small.append(frame, base_time + i * 0.033, frame_id=i)
        
        start = time.time()
        for _ in range(100):
            buffer_small.get_frame_by_timestamp(base_time + 1.5, tolerance_sec=0.1)
        time_small = (time.time() - start) * 1000 / 100
        
        print(f"   Average query time: {time_small:.3f}ms")
        print(f"   Use case: Low-latency search (last 3 seconds)")
        
        # Strategy 2: Larger buffer for comprehensive search
        print("\n📊 Strategy 2: Larger buffer (full history)")
        buffer_large = FrameHistoryBuffer(max_memory_frames=300, cache_dir=tmpdir)
        
        for i in range(300):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            buffer_large.append(frame, base_time + i * 0.033, frame_id=i)
        
        start = time.time()
        for _ in range(100):
            buffer_large.get_frame_by_timestamp(base_time + 5.0, tolerance_sec=0.5)
        time_large = (time.time() - start) * 1000 / 100
        
        print(f"   Average query time: {time_large:.3f}ms")
        print(f"   Use case: Comprehensive search (full 10 seconds)")
        
        # Strategy 3: Batch similarity computations
        print("\n📊 Strategy 3: Batch similarity computation")
        emb1 = SyntheticEmbeddingGenerator.create_embedding()
        
        # Single similarities
        start = time.time()
        for _ in range(1000):
            emb2 = SyntheticEmbeddingGenerator.create_embedding()
            sim = np.dot(emb1, emb2)
        time_single = (time.time() - start) * 1000
        
        # Batch similarities
        embs = [SyntheticEmbeddingGenerator.create_embedding() for _ in range(1000)]
        start = time.time()
        sims = np.dot(embs, emb1)  # Vectorized
        time_batch = (time.time() - start) * 1000
        
        print(f"   Single comparisons: {time_single:.2f}ms (1000 iterations)")
        print(f"   Batch comparison: {time_batch:.2f}ms (1000 iterations)")
        print(f"   Speedup: {time_single / time_batch:.1f}x")
        print(f"   Use case: Compare query against many embeddings at once")


# ============================================================================
# EXAMPLE 6: TEST SCENARIO WALKTHROUGH
# ============================================================================

def example_test_scenario_walkthrough():
    """
    Example: Complete backtrack search on realistic scenario
    
    Use case: Validate system end-to-end
    """
    print("\n" + "="*80)
    print("EXAMPLE 6: Complete Backtrack Search on Test Scenario")
    print("="*80)
    
    # Generate test scenario
    print("📊 Generating test scenario: 'Person with Baggage'")
    scenario = TestScenarioGenerator.scenario_person_with_baggage()
    
    print(f"   Frames: {scenario['frame_count']}")
    print(f"   Duration: {scenario['duration_sec']}s")
    print(f"   Objects: {scenario['detections']}")
    
    # Simulate: Add frames to buffer
    with tempfile.TemporaryDirectory() as tmpdir:
        buffer = FrameHistoryBuffer(max_memory_frames=100, cache_dir=tmpdir)
        logger = EventLogger(log_dir=tmpdir)
        
        print("\n📹 Processing frames...")
        base_time = time.time()
        
        for frame_data in scenario['frames']:
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            ts = base_time + frame_data['timestamp']
            
            buffer.append(frame, ts, frame_id=frame_data['frame_idx'])
            
            # Log detections
            for det_dict in frame_data['detections']:
                logger.log_event({
                    'event': 'DETECTION',
                    'timestamp': ts,
                    'frame_id': frame_data['frame_idx'],
                    'class_name': det_dict['class_name'],
                    'confidence': det_dict['confidence']
                })
        
        print(f"✅ Processed {scenario['frame_count']} frames")
        
        # Perform backtrack search
        print("\n🔍 Searching for baggage detections...")
        
        search_target = base_time + scenario['duration_sec'] / 2
        frames = buffer.get_frames_in_range(base_time, base_time + scenario['duration_sec'])
        
        print(f"✅ Found {len(frames)} frames in search window")
        print(f"✅ Buffer stats: {buffer.get_buffer_stats()}")


# ============================================================================
# MAIN RUNNER
# ============================================================================

def run_all_examples():
    """Run all examples"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "  BACKTRACK SEARCH - INTEGRATION EXAMPLES".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    
    examples = [
        example_basic_backtrack_search,
        example_face_matching_with_embeddings,
        example_person_tracking,
        example_mesh_search,
        example_performance_optimization,
        example_test_scenario_walkthrough,
    ]
    
    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n❌ Example failed: {example_func.__name__}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    run_all_examples()
