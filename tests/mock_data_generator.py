"""
Backtrack Search - Test Data Generator & Mock Utilities

Provides:
1. Synthetic YOLO detections (without running actual YOLO)
2. Synthetic embeddings for testing
3. Mock mesh messages
4. Test scenario generators
"""

import numpy as np
import time
from typing import List, Dict, Tuple
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class MockYOLODetection:
    """Mock YOLO detection for testing"""
    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int
    class_name: str
    confidence: float
    
    def to_dict(self):
        return {
            'x1': self.x1, 'y1': self.y1,
            'x2': self.x2, 'y2': self.y2,
            'class_id': self.class_id,
            'class_name': self.class_name,
            'confidence': self.confidence
        }


class SyntheticEmbeddingGenerator:
    """Generate synthetic embeddings for testing"""
    
    @staticmethod
    def create_embedding(seed: int = None, dim: int = 512) -> np.ndarray:
        """Create normalized embedding"""
        if seed is not None:
            np.random.seed(seed)
        emb = np.random.randn(dim)
        emb = emb / (np.linalg.norm(emb) + 1e-6)
        return emb
    
    @staticmethod
    def create_similar_embeddings(base_seed: int, 
                                   noise_level: float = 0.1,
                                   count: int = 3,
                                   dim: int = 512) -> List[np.ndarray]:
        """Create embeddings similar to base (same person)"""
        base_emb = SyntheticEmbeddingGenerator.create_embedding(base_seed, dim)
        embeddings = []
        
        for i in range(count):
            noise = np.random.randn(dim) * noise_level
            emb = base_emb + noise
            emb = emb / (np.linalg.norm(emb) + 1e-6)
            embeddings.append(emb)
        
        return embeddings
    
    @staticmethod
    def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings"""
        return float(np.dot(emb1, emb2))


class SyntheticYOLODetectionGenerator:
    """Generate synthetic YOLO detections"""
    
    @staticmethod
    def create_person_detection(person_idx: int = 0,
                               frame_width: int = 640,
                               frame_height: int = 480) -> MockYOLODetection:
        """Create detection for a person"""
        # Simulate person at random location
        np.random.seed(person_idx)
        
        person_width = frame_width * 0.3
        person_height = frame_height * 0.4
        
        x1 = np.random.uniform(0, frame_width - person_width)
        y1 = np.random.uniform(0, frame_height - person_height)
        x2 = x1 + person_width
        y2 = y1 + person_height
        
        return MockYOLODetection(
            x1=x1, y1=y1, x2=x2, y2=y2,
            class_id=0, class_name='person',
            confidence=np.random.uniform(0.85, 0.99)
        )
    
    @staticmethod
    def create_baggage_detection(baggage_idx: int = 0,
                                frame_width: int = 640,
                                frame_height: int = 480) -> MockYOLODetection:
        """Create detection for baggage"""
        np.random.seed(100 + baggage_idx)
        
        bag_width = frame_width * 0.15
        bag_height = frame_height * 0.25
        
        x1 = np.random.uniform(0, frame_width - bag_width)
        y1 = np.random.uniform(frame_height * 0.5, frame_height - bag_height)
        x2 = x1 + bag_width
        y2 = y1 + bag_height
        
        return MockYOLODetection(
            x1=x1, y1=y1, x2=x2, y2=y2,
            class_id=60, class_name='backpack',
            confidence=np.random.uniform(0.80, 0.95)
        )


class MockMeshMessage:
    """Mock mesh protocol message"""
    
    FACE_SEARCH_REQUEST = 'FACE_SEARCH_REQUEST'
    FACE_SEARCH_RESULT = 'FACE_SEARCH_RESULT'
    FACE_EMBEDDING_SYNC = 'FACE_EMBEDDING_SYNC'
    
    def __init__(self, msg_type: str, data: Dict = None, 
                 sender_id: str = 'node-0', seq_num: int = 1):
        self.msg_type = msg_type
        self.data = data or {}
        self.sender_id = sender_id
        self.seq_num = seq_num
        self.timestamp = time.time()
    
    def to_dict(self):
        return {
            'type': self.msg_type,
            'sender_id': self.sender_id,
            'seq_num': self.seq_num,
            'timestamp': self.timestamp,
            'data': self.data
        }
    
    def to_json(self):
        return json.dumps(self.to_dict())


class TestScenarioGenerator:
    """Generate complete test scenarios"""
    
    @staticmethod
    def scenario_single_person_walk() -> Dict:
        """Person walking through frame (single track)"""
        frames = []
        base_time = time.time()
        
        for frame_idx in range(30):  # 1 second @ 30fps
            frame_data = {
                'frame_idx': frame_idx,
                'timestamp': base_time + frame_idx / 30.0,
                'detections': []
            }
            
            # Person moves left to right across frame
            x_pos = frame_idx * (640 / 30.0)
            
            det = MockYOLODetection(
                x1=x_pos, y1=150, x2=x_pos+200, y2=450,
                class_id=0, class_name='person',
                confidence=0.95
            )
            
            frame_data['detections'].append(det.to_dict())
            frames.append(frame_data)
        
        return {
            'scenario': 'single_person_walk',
            'duration_sec': 1.0,
            'frame_count': 30,
            'frames': frames
        }
    
    @staticmethod
    def scenario_multiple_people() -> Dict:
        """Multiple people in frame"""
        frames = []
        base_time = time.time()
        
        for frame_idx in range(20):
            frame_data = {
                'frame_idx': frame_idx,
                'timestamp': base_time + frame_idx / 30.0,
                'detections': []
            }
            
            # 3 people at fixed positions
            positions = [
                (100, 150, 300, 450),
                (350, 200, 550, 500),
                (500, 100, 700, 400),
            ]
            
            for person_id, (x1, y1, x2, y2) in enumerate(positions):
                det = MockYOLODetection(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    class_id=0, class_name='person',
                    confidence=0.90 + person_id * 0.03
                )
                frame_data['detections'].append(det.to_dict())
            
            frames.append(frame_data)
        
        return {
            'scenario': 'multiple_people',
            'duration_sec': 0.67,
            'frame_count': 20,
            'person_count': 3,
            'frames': frames
        }
    
    @staticmethod
    def scenario_person_with_baggage() -> Dict:
        """Person detected with nearby baggage"""
        frames = []
        base_time = time.time()
        
        for frame_idx in range(15):
            frame_data = {
                'frame_idx': frame_idx,
                'timestamp': base_time + frame_idx / 30.0,
                'detections': []
            }
            
            # Person
            person_det = MockYOLODetection(
                x1=200, y1=100, x2=400, y2=400,
                class_id=0, class_name='person',
                confidence=0.95
            )
            frame_data['detections'].append(person_det.to_dict())
            
            # Baggage near person
            bag_det = MockYOLODetection(
                x1=300, y1=380, x2=420, y2=480,
                class_id=60, class_name='backpack',
                confidence=0.88
            )
            frame_data['detections'].append(bag_det.to_dict())
            
            frames.append(frame_data)
        
        return {
            'scenario': 'person_with_baggage',
            'duration_sec': 0.5,
            'frame_count': 15,
            'detections': {
                'persons': 1,
                'baggage': 1
            },
            'frames': frames
        }
    
    @staticmethod
    def scenario_search_challenge() -> Dict:
        """Challenging scenario: crowded, occlusions"""
        frames = []
        base_time = time.time()
        
        for frame_idx in range(60):  # 2 seconds
            frame_data = {
                'frame_idx': frame_idx,
                'timestamp': base_time + frame_idx / 30.0,
                'detections': []
            }
            
            # 5-7 people with occlusions
            num_people = 5 + (1 if frame_idx % 10 < 5 else 0)
            
            for person_id in range(num_people):
                x = 50 + person_id * 100 + np.random.randint(-20, 20)
                y = 100 + np.random.randint(-30, 30)
                
                # Lower confidence for occluded people
                base_conf = 0.92
                if frame_idx % 15 < 3:  # Occlusion in some frames
                    base_conf -= 0.1
                
                det = MockYOLODetection(
                    x1=x, y1=y, x2=x+150, y2=y+300,
                    class_id=0, class_name='person',
                    confidence=max(0.70, base_conf)
                )
                frame_data['detections'].append(det.to_dict())
            
            frames.append(frame_data)
        
        return {
            'scenario': 'search_challenge',
            'duration_sec': 2.0,
            'frame_count': 60,
            'challenges': ['multiple_people', 'occlusions', 'motion'],
            'frames': frames
        }


class BacktrackSearchValidator:
    """Validate backtrack search results"""
    
    @staticmethod
    def validate_matches(matches: List[Dict], 
                        expected_count: int = None,
                        min_similarity: float = 0.7) -> bool:
        """Validate search results"""
        # Check structure
        for match in matches:
            assert 'frame_id' in match, "Missing frame_id"
            assert 'timestamp' in match, "Missing timestamp"
            assert 'similarity' in match, "Missing similarity"
            assert 'bbox' in match, "Missing bbox"
            assert match['similarity'] >= min_similarity, \
                f"Similarity {match['similarity']} below threshold {min_similarity}"
        
        # Check ordering (should be sorted by similarity descending)
        similarities = [m['similarity'] for m in matches]
        assert similarities == sorted(similarities, reverse=True), \
            "Matches not sorted by similarity descending"
        
        # Check count if expected
        if expected_count is not None:
            assert len(matches) == expected_count, \
                f"Expected {expected_count} matches, got {len(matches)}"
        
        return True
    
    @staticmethod
    def validate_match_distribution(matches: List[Dict],
                                   max_time_window: float = 1.0) -> bool:
        """Validate that matches are in expected time window"""
        if not matches:
            return True
        
        timestamps = [m['timestamp'] for m in matches]
        time_span = max(timestamps) - min(timestamps)
        
        assert time_span <= max_time_window, \
            f"Match time span {time_span}s exceeds window {max_time_window}s"
        
        return True


# Example usage for manual testing
if __name__ == "__main__":
    print("=" * 80)
    print("MOCK DATA GENERATION EXAMPLES")
    print("=" * 80)
    
    # Example 1: Generate single person embedding
    print("\n1. Single Embedding:")
    emb = SyntheticEmbeddingGenerator.create_embedding(seed=42)
    print(f"   Embedding shape: {emb.shape}")
    print(f"   Norm: {np.linalg.norm(emb):.4f}")
    
    # Example 2: Generate similar embeddings
    print("\n2. Similar Embeddings (same person):")
    embeddings = SyntheticEmbeddingGenerator.create_similar_embeddings(
        base_seed=42, noise_level=0.1, count=3
    )
    for i, emb in enumerate(embeddings):
        sim = SyntheticEmbeddingGenerator.compute_similarity(emb, embeddings[0])
        print(f"   Emb {i} similarity to Emb 0: {sim:.4f}")
    
    # Example 3: Generate YOLO detections
    print("\n3. Person Detection:")
    det = SyntheticYOLODetectionGenerator.create_person_detection()
    print(f"   Detection: {det.to_dict()}")
    
    # Example 4: Mock mesh message
    print("\n4. Mock Mesh Message:")
    msg = MockMeshMessage(
        msg_type=MockMeshMessage.FACE_SEARCH_REQUEST,
        data={'search_id': 's_001', 'similarity_threshold': 0.75},
        sender_id='node-0'
    )
    print(f"   Message: {msg.to_json()}")
    
    # Example 5: Test scenario - single person
    print("\n5. Test Scenario - Single Person Walk:")
    scenario = TestScenarioGenerator.scenario_single_person_walk()
    print(f"   Scenario: {scenario['scenario']}")
    print(f"   Frames: {scenario['frame_count']}, Duration: {scenario['duration_sec']}s")
    print(f"   First frame detections: {len(scenario['frames'][0]['detections'])}")
    
    # Example 6: Test scenario - challenging
    print("\n6. Test Scenario - Search Challenge:")
    scenario = TestScenarioGenerator.scenario_search_challenge()
    print(f"   Scenario: {scenario['scenario']}")
    print(f"   Challenges: {scenario['challenges']}")
    print(f"   Frames: {scenario['frame_count']}")
    
    print("\n" + "=" * 80)
    print("Use these generators to create test data for backtrack search validation")
    print("=" * 80)
