"""
Vision Pipeline Test Suite
Comprehensive testing for person-bag linking, mismatch detection, and search
"""

import unittest
import numpy as np
import cv2
from datetime import datetime

from vision import (
    BaggageLinking,
    YOLODetectionEngine,
    EmbeddingExtractor,
    ColorDescriptor,
    PersonBagLinkingEngine,
    HashIDGenerator,
    MismatchDetector,
    DescriptionSearchEngine,
    ObjectClass,
    LinkingStatus,
    BoundingBox,
    ColorHistogram,
    Detection,
    PersonBagLink,
    BaggageProfile,
)


# ============================================================================
# BOUNDING BOX TESTS
# ============================================================================

class TestBoundingBox(unittest.TestCase):
    """Test bounding box geometry operations"""
    
    def setUp(self):
        self.bbox1 = BoundingBox(100, 100, 300, 400)
        self.bbox2 = BoundingBox(320, 350, 380, 450)
        self.bbox3 = BoundingBox(150, 150, 250, 350)
    
    def test_dimensions(self):
        self.assertEqual(self.bbox1.width(), 200)
        self.assertEqual(self.bbox1.height(), 300)
        self.assertEqual(self.bbox1.area(), 60000)
    
    def test_center(self):
        center = self.bbox1.center()
        self.assertEqual(center, (200, 250))
    
    def test_distance(self):
        dist = self.bbox1.distance_to(self.bbox2)
        self.assertGreater(dist, 0)
    
    def test_iou_overlap(self):
        iou = self.bbox1.iou(self.bbox3)
        self.assertGreater(iou, 0)
        self.assertLess(iou, 1)
    
    def test_iou_no_overlap(self):
        bbox_far = BoundingBox(1000, 1000, 1100, 1100)
        iou = self.bbox1.iou(bbox_far)
        self.assertEqual(iou, 0)
    
    def test_to_int_coords(self):
        coords = self.bbox1.to_int_coords()
        self.assertEqual(coords, (100, 100, 300, 400))


# ============================================================================
# COLOR HISTOGRAM TESTS
# ============================================================================

class TestColorHistogram(unittest.TestCase):
    """Test color histogram extraction"""
    
    def setUp(self):
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.bbox = BoundingBox(50, 50, 200, 200)
    
    def test_extract_histogram_empty(self):
        hist = ColorDescriptor.extract_histogram(self.frame, self.bbox)
        self.assertIsNotNone(hist)
        self.assertEqual(len(hist.h_hist), 180)
        self.assertEqual(len(hist.s_hist), 256)
        self.assertEqual(len(hist.v_hist), 256)
        self.assertEqual(len(hist.lab_hist), 256)
    
    def test_extract_histogram_colored(self):
        # Create colored region
        cv2.rectangle(self.frame, (50, 50), (200, 200), (0, 255, 0), -1)
        hist = ColorDescriptor.extract_histogram(self.frame, self.bbox)
        
        self.assertGreater(np.sum(hist.h_hist), 0)
        self.assertGreater(np.sum(hist.s_hist), 0)
        self.assertGreater(np.sum(hist.v_hist), 0)
    
    def test_histogram_similarity_identical(self):
        cv2.rectangle(self.frame, (50, 50), (200, 200), (0, 255, 0), -1)
        hist1 = ColorDescriptor.extract_histogram(self.frame, self.bbox)
        hist2 = ColorDescriptor.extract_histogram(self.frame, self.bbox)
        
        similarity = ColorDescriptor.histogram_distance(hist1, hist2)
        self.assertAlmostEqual(similarity, 1.0, places=2)
    
    def test_histogram_similarity_different(self):
        # Create two different colored regions
        self.frame[50:200, 50:200] = [0, 255, 0]
        hist1 = ColorDescriptor.extract_histogram(self.frame, self.bbox)
        
        self.frame[:] = 0  # Clear
        self.frame[50:200, 50:200] = [255, 0, 0]
        hist2 = ColorDescriptor.extract_histogram(self.frame, self.bbox)
        
        similarity = ColorDescriptor.histogram_distance(hist1, hist2)
        self.assertLess(similarity, 0.9)
    
    def test_histogram_to_dict(self):
        hist = ColorHistogram()
        d = hist.to_dict()
        
        self.assertIn('h_hist', d)
        self.assertIn('s_hist', d)
        self.assertIn('v_hist', d)
        self.assertIn('lab_hist', d)
    
    def test_histogram_from_dict(self):
        hist1 = ColorHistogram()
        d = hist1.to_dict()
        hist2 = ColorHistogram.from_dict(d)
        
        np.testing.assert_array_almost_equal(hist1.h_hist, hist2.h_hist)


# ============================================================================
# DETECTION TESTS
# ============================================================================

class TestDetection(unittest.TestCase):
    """Test detection objects"""
    
    def setUp(self):
        self.detection = Detection(
            class_name=ObjectClass.SUITCASE,
            bbox=BoundingBox(100, 100, 200, 300),
            confidence=0.95,
            embedding=np.random.randn(512),
            camera_id="CAM_001",
            frame_id=0
        )
    
    def test_detection_creation(self):
        self.assertEqual(self.detection.class_name, ObjectClass.SUITCASE)
        self.assertEqual(self.detection.confidence, 0.95)
        self.assertEqual(self.detection.camera_id, "CAM_001")
    
    def test_embedding_normalized(self):
        norm_emb = self.detection.get_embedding_normalized()
        norm = np.linalg.norm(norm_emb)
        self.assertAlmostEqual(norm, 1.0, places=5)


# ============================================================================
# EMBEDDING EXTRACTOR TESTS
# ============================================================================

class TestEmbeddingExtractor(unittest.TestCase):
    """Test embedding extraction"""
    
    def setUp(self):
        self.extractor = EmbeddingExtractor(embedding_dim=512)
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.bbox = BoundingBox(50, 50, 200, 200)
    
    def test_extract_embedding_dimension(self):
        embedding = self.extractor.extract(self.frame, self.bbox)
        self.assertEqual(len(embedding), 512)
    
    def test_extract_embedding_type(self):
        embedding = self.extractor.extract(self.frame, self.bbox)
        self.assertIsInstance(embedding, np.ndarray)
    
    def test_extract_embedding_empty_region(self):
        bbox_empty = BoundingBox(5, 5, 10, 10)
        embedding = self.extractor.extract(self.frame, bbox_empty)
        self.assertEqual(len(embedding), 512)


# ============================================================================
# PERSON-BAG LINKING TESTS
# ============================================================================

class TestPersonBagLinking(unittest.TestCase):
    """Test person-bag linking engine"""
    
    def setUp(self):
        self.engine = PersonBagLinkingEngine(
            spatial_threshold=150.0,
            feature_threshold=0.6
        )
        
        self.person = Detection(
            class_name=ObjectClass.PERSON,
            bbox=BoundingBox(100, 100, 200, 400),
            confidence=0.95,
            embedding=np.random.randn(512)
        )
        
        self.bag_close = Detection(
            class_name=ObjectClass.SUITCASE,
            bbox=BoundingBox(220, 350, 280, 450),
            confidence=0.85,
            embedding=self.person.embedding + np.random.randn(512) * 0.1
        )
        
        self.bag_far = Detection(
            class_name=ObjectClass.BACKPACK,
            bbox=BoundingBox(600, 100, 700, 250),
            confidence=0.80,
            embedding=np.random.randn(512)
        )
    
    def test_link_close_bags(self):
        link = self.engine.link_person_to_bags(self.person, [self.bag_close])
        
        self.assertIsNotNone(link)
        self.assertEqual(link.person_id, self.person.bbox.center().__str__())
    
    def test_link_far_bags(self):
        link = self.engine.link_person_to_bags(self.person, [self.bag_far])
        
        # Far bag shouldn't link due to spatial threshold
        # Link might still be created if score is high, but status might differ
        if link:
            self.assertGreater(link.spatial_distance, self.engine.spatial_threshold)
    
    def test_link_multiple_bags(self):
        bags = [self.bag_close, self.bag_far]
        link = self.engine.link_person_to_bags(self.person, bags)
        
        # Should link to closest bag
        if link:
            self.assertLess(link.spatial_distance, 150)
    
    def test_link_empty_bags(self):
        link = self.engine.link_person_to_bags(self.person, [])
        self.assertIsNone(link)


# ============================================================================
# HASH ID GENERATION TESTS
# ============================================================================

class TestHashIDGenerator(unittest.TestCase):
    """Test hash ID generation"""
    
    def setUp(self):
        self.detection1 = Detection(
            class_name=ObjectClass.SUITCASE,
            bbox=BoundingBox(100, 100, 200, 300),
            confidence=0.95,
            embedding=np.random.randn(512)
        )
        
        self.detection2 = Detection(
            class_name=ObjectClass.SUITCASE,
            bbox=BoundingBox(100, 100, 200, 300),
            confidence=0.95,
            embedding=np.random.randn(512)
        )
    
    def test_generate_hash_id_format(self):
        hash_id = HashIDGenerator.generate_hash_id(self.detection1)
        
        self.assertIsInstance(hash_id, str)
        self.assertEqual(len(hash_id), 16)
        self.assertTrue(all(c in '0123456789abcdef' for c in hash_id))
    
    def test_generate_hash_id_deterministic(self):
        # Same embedding should give same hash
        detection = Detection(
            class_name=ObjectClass.SUITCASE,
            bbox=BoundingBox(100, 100, 200, 300),
            confidence=0.95,
            embedding=np.ones(512)
        )
        
        hash1 = HashIDGenerator.generate_hash_id(detection)
        hash2 = HashIDGenerator.generate_hash_id(detection)
        
        self.assertEqual(hash1, hash2)
    
    def test_generate_bag_id_format(self):
        bag_id = HashIDGenerator.generate_bag_id("CAM_001", 10, 2)
        
        self.assertIn("BAG_", bag_id)
        self.assertIn("CAM_001", bag_id)


# ============================================================================
# BAGGAGE PROFILE TESTS
# ============================================================================

class TestBaggageProfile(unittest.TestCase):
    """Test baggage profile"""
    
    def setUp(self):
        self.profile = BaggageProfile(
            bag_id="BAG_001",
            hash_id="HASH_001",
            class_name=ObjectClass.SUITCASE,
            color_histogram=ColorHistogram(),
            embedding=np.random.randn(512),
            description="Red suitcase with wheels"
        )
    
    def test_profile_creation(self):
        self.assertEqual(self.profile.bag_id, "BAG_001")
        self.assertEqual(self.profile.hash_id, "HASH_001")
        self.assertEqual(self.profile.class_name, ObjectClass.SUITCASE)
    
    def test_profile_to_dict(self):
        d = self.profile.to_dict()
        
        self.assertIn('bag_id', d)
        self.assertIn('hash_id', d)
        self.assertIn('class_name', d)
        self.assertIn('description', d)


# ============================================================================
# MISMATCH DETECTOR TESTS
# ============================================================================

class TestMismatchDetector(unittest.TestCase):
    """Test mismatch detection"""
    
    def setUp(self):
        self.detector = MismatchDetector()
        
        self.link = PersonBagLink(
            person_id="PERSON_001",
            bag_id="BAG_001",
            confidence=0.95
        )
    
    def test_register_link(self):
        self.detector.register_link("CAM_001", self.link)
        
        self.assertIn("CAM_001", self.detector.person_bag_registry)
        self.assertIn("PERSON_001", self.detector.person_bag_registry["CAM_001"])


# ============================================================================
# DESCRIPTION SEARCH TESTS
# ============================================================================

class TestDescriptionSearchEngine(unittest.TestCase):
    """Test baggage search engine"""
    
    def setUp(self):
        self.search_engine = DescriptionSearchEngine()
        
        self.profiles = [
            BaggageProfile(
                bag_id="BAG_001",
                hash_id="HASH_001",
                class_name=ObjectClass.SUITCASE,
                color_histogram=ColorHistogram(),
                embedding=np.random.randn(512),
                description="Red hard shell suitcase with wheels"
            ),
            BaggageProfile(
                bag_id="BAG_002",
                hash_id="HASH_002",
                class_name=ObjectClass.BACKPACK,
                color_histogram=ColorHistogram(),
                embedding=np.random.randn(512),
                description="Blue fabric backpack with laptop compartment"
            ),
            BaggageProfile(
                bag_id="BAG_003",
                hash_id="HASH_003",
                class_name=ObjectClass.HANDBAG,
                color_histogram=ColorHistogram(),
                embedding=np.random.randn(512),
                description="Black leather handbag"
            ),
        ]
        
        for profile in self.profiles:
            self.search_engine.add_baggage(profile)
    
    def test_search_by_description(self):
        results = self.search_engine.search_by_description("suitcase")
        
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].bag_id, "BAG_001")
    
    def test_search_by_description_multiple_matches(self):
        results = self.search_engine.search_by_description("with", top_k=10)
        
        # Multiple bags have "with" in description
        self.assertGreater(len(results), 0)
    
    def test_search_by_embedding(self):
        query = self.profiles[0].embedding
        results = self.search_engine.search_by_embedding(query, top_k=3)
        
        self.assertEqual(len(results), 3)
        self.assertGreater(results[0][1], 0)  # Similarity score
    
    def test_search_by_color(self):
        query_hist = ColorHistogram()
        results = self.search_engine.search_by_color(query_hist, top_k=3)
        
        self.assertEqual(len(results), 3)
    
    def test_search_empty_database(self):
        empty_search = DescriptionSearchEngine()
        results = empty_search.search_by_description("test")
        
        self.assertEqual(len(results), 0)


# ============================================================================
# YOLO DETECTION ENGINE TESTS
# ============================================================================

class TestYOLODetectionEngine(unittest.TestCase):
    """Test YOLO detection engine"""
    
    def setUp(self):
        self.engine = YOLODetectionEngine(
            model_name="yolov8n",
            confidence_threshold=0.5
        )
        self.frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    def test_detect_empty_frame(self):
        detections = self.engine.detect(self.frame)
        
        self.assertIsInstance(detections, list)
    
    def test_detect_with_camera_id(self):
        detections = self.engine.detect(self.frame, camera_id="TEST_CAM", frame_id=1)
        
        # All detections should have camera_id set
        for det in detections:
            self.assertEqual(det.camera_id, "TEST_CAM")
            self.assertEqual(det.frame_id, 1)


# ============================================================================
# COMPLETE PIPELINE TESTS
# ============================================================================

class TestBaggageLinkingPipeline(unittest.TestCase):
    """Test complete baggage linking pipeline"""
    
    def setUp(self):
        self.pipeline = BaggageLinking()
        self.frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    def test_process_frame_empty(self):
        result = self.pipeline.process_frame(self.frame, camera_id="CAM_001")
        
        self.assertIn('frame_id', result)
        self.assertIn('camera_id', result)
        self.assertIn('detections', result)
        self.assertIn('persons', result)
        self.assertIn('bags', result)
        self.assertIn('links', result)
        self.assertIn('processing_time_ms', result)
    
    def test_get_statistics(self):
        stats = self.pipeline.get_statistics()
        
        self.assertIn('total_bags', stats)
        self.assertIn('total_links', stats)
        self.assertIn('total_mismatches', stats)
        self.assertIn('cameras', stats)
        self.assertIn('timestamp', stats)
    
    def test_search_baggage(self):
        results = self.pipeline.search_baggage("suitcase", method='description')
        
        self.assertIsInstance(results, list)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def test_end_to_end_detection_and_linking(self):
        """Test complete flow from detection to linking"""
        pipeline = BaggageLinking()
        
        # Create synthetic frame
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Add some colored regions to simulate objects
        cv2.rectangle(frame, (200, 100), (300, 400), (0, 255, 0), -1)   # Person
        cv2.rectangle(frame, (320, 350), (380, 450), (255, 0, 0), -1)   # Bag
        
        # Process
        result = pipeline.process_frame(frame, camera_id="CAM_001", frame_id=0)
        
        # Verify result structure
        self.assertIsInstance(result, dict)
        self.assertGreater(result['processing_time_ms'], 0)
    
    def test_multiple_frame_processing(self):
        """Test processing multiple frames sequentially"""
        pipeline = BaggageLinking()
        
        for frame_id in range(3):
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.rectangle(frame, (100 + frame_id*50, 100), 
                         (300 + frame_id*50, 400), (0, 255, 0), -1)
            
            result = pipeline.process_frame(frame, camera_id="CAM_001", frame_id=frame_id)
            
            self.assertIsNotNone(result)
            self.assertEqual(result['frame_id'], frame_id)


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    test_classes = [
        TestBoundingBox,
        TestColorHistogram,
        TestDetection,
        TestEmbeddingExtractor,
        TestPersonBagLinking,
        TestHashIDGenerator,
        TestBaggageProfile,
        TestMismatchDetector,
        TestDescriptionSearchEngine,
        TestYOLODetectionEngine,
        TestBaggageLinkingPipeline,
        TestIntegration,
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
