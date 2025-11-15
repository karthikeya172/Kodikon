"""
Vision Pipeline Examples
Demonstrates person-bag linking, mismatch detection, and search capabilities
"""

import numpy as np
import cv2
from datetime import datetime
from typing import List

from vision import (
    BaggageLinking,
    ObjectClass,
    LinkingStatus,
    BoundingBox,
    Detection,
    ColorHistogram,
    PersonBagLink,
)


def example_1_basic_detection():
    """Example 1: Basic YOLO detection in frame"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic YOLO Detection")
    print("="*60)
    
    # Create dummy frame
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Add some colored rectangles to simulate objects
    cv2.rectangle(frame, (100, 100), (300, 400), (0, 255, 0), -1)  # Green
    cv2.rectangle(frame, (400, 150), (550, 350), (0, 0, 255), -1)   # Red
    cv2.rectangle(frame, (600, 200), (800, 450), (255, 0, 0), -1)   # Blue
    
    # Initialize pipeline
    baggage_linking = BaggageLinking({
        'confidence_threshold': 0.5,
        'yolo_model': 'yolov8n'
    })
    
    print("\nProcessing frame...")
    result = baggage_linking.process_frame(frame, camera_id="CAM001", frame_id=0)
    
    print(f"\nDetections found: {len(result['detections'])}")
    print(f"  - Persons: {len(result['persons'])}")
    print(f"  - Bags: {len(result['bags'])}")
    print(f"Processing time: {result['processing_time_ms']:.2f} ms")
    
    # Print detection details
    for i, det in enumerate(result['detections']):
        print(f"\nDetection {i+1}:")
        print(f"  Class: {det.class_name.value}")
        print(f"  Confidence: {det.confidence:.2%}")
        print(f"  Bbox: ({det.bbox.x1:.0f}, {det.bbox.y1:.0f}, {det.bbox.x2:.0f}, {det.bbox.y2:.0f})")
        print(f"  Size: {det.bbox.width():.0f}x{det.bbox.height():.0f}")


def example_2_person_bag_linking():
    """Example 2: Person-bag linking in registration camera"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Person-Bag Linking")
    print("="*60)
    
    baggage_linking = BaggageLinking()
    
    # Create dummy frame with person and bag near each other
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(frame, (200, 100), (300, 400), (0, 255, 0), -1)   # Person (green)
    cv2.rectangle(frame, (320, 350), (380, 450), (255, 0, 0), -1)   # Bag near person (blue)
    
    print("\nProcessing frame with person and nearby bag...")
    result = baggage_linking.process_frame(frame, camera_id="CAM_REGISTRATION", frame_id=1)
    
    print(f"\nDetections: {len(result['detections'])}")
    print(f"Links found: {len(result['links'])}")
    
    for i, link in enumerate(result['links']):
        print(f"\nLink {i+1}:")
        print(f"  Person: {link.person_id}")
        print(f"  Bag: {link.bag_id}")
        print(f"  Status: {link.status.value}")
        print(f"  Overall confidence: {link.overall_score():.2%}")
        print(f"  - Spatial: {link.spatial_distance:.1f}px")
        print(f"  - Feature similarity: {link.feature_similarity:.2%}")
        print(f"  - Color similarity: {link.color_similarity:.2%}")


def example_3_mismatch_detection():
    """Example 3: Baggage mismatch detection in surveillance"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Mismatch Detection")
    print("="*60)
    
    baggage_linking = BaggageLinking()
    
    # Step 1: Register person-bag link at CAM1
    print("\nStep 1: Register person-bag at CAM_REGISTRATION")
    frame1 = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(frame1, (200, 100), (300, 400), (0, 255, 0), -1)
    cv2.rectangle(frame1, (320, 350), (380, 450), (255, 0, 0), -1)
    
    result1 = baggage_linking.process_frame(frame1, camera_id="CAM_REGISTRATION", frame_id=0)
    print(f"  Links registered: {len(result1['links'])}")
    
    # Step 2: Check at surveillance camera
    print("\nStep 2: Check at surveillance camera")
    frame2 = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(frame2, (150, 100), (250, 400), (0, 255, 0), -1)   # Same person
    cv2.rectangle(frame2, (100, 350), (160, 450), (0, 255, 255), -1) # Different bag (yellow)
    
    result2 = baggage_linking.process_frame(frame2, camera_id="CAM_SURVEILLANCE_1", frame_id=1)
    print(f"  Detections: {len(result2['detections'])}")
    print(f"  Mismatches detected: {len(result2['mismatches'])}")
    
    for mismatch in result2['mismatches']:
        print(f"\n  ⚠️  MISMATCH DETECTED:")
        print(f"      Person: {mismatch['person_id']}")
        print(f"      Expected: {mismatch['expected_bag']}")
        print(f"      Observed: {mismatch['current_bag']}")
        print(f"      Reason: {mismatch['reason']}")


def example_4_color_histogram():
    """Example 4: Color-based baggage identification"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Color-Based Identification")
    print("="*60)
    
    from vision import ColorDescriptor
    
    # Create frames with different colored objects
    frames = []
    colors = [
        ("Red bag", (0, 0, 255), (50, 50, 50)),
        ("Blue bag", (255, 0, 0), (50, 50, 50)),
        ("Green bag", (0, 255, 0), (50, 50, 50)),
    ]
    
    print("\nCreating baggage samples with different colors...")
    for name, color, _ in colors:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.rectangle(frame, (200, 200), (400, 400), color, -1)
        frames.append((name, frame))
        print(f"  - {name}")
    
    # Extract histograms
    bbox = BoundingBox(200, 200, 400, 400)
    print("\nExtracting color histograms...")
    histograms = []
    for name, frame in frames:
        hist = ColorDescriptor.extract_histogram(frame, bbox)
        histograms.append((name, hist))
    
    # Compare histograms
    print("\nComparing histograms (0=different, 1=same):")
    for i, (name1, hist1) in enumerate(histograms):
        for j, (name2, hist2) in enumerate(histograms[i+1:], start=i+1):
            similarity = ColorDescriptor.histogram_distance(hist1, hist2)
            print(f"  {name1} vs {name2}: {similarity:.3f}")


def example_5_embedding_search():
    """Example 5: Baggage search by embedding similarity"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Embedding-Based Search")
    print("="*60)
    
    from vision import DescriptionSearchEngine, BaggageProfile
    
    # Create search engine
    search_engine = DescriptionSearchEngine()
    
    # Add sample baggage profiles
    print("\nAdding baggage profiles to database...")
    for i in range(5):
        profile = BaggageProfile(
            bag_id=f"BAG_{i:03d}",
            hash_id=f"HASH_{i:03d}",
            class_name=ObjectClass.SUITCASE if i % 2 == 0 else ObjectClass.BACKPACK,
            color_histogram=ColorHistogram(),
            embedding=np.random.randn(512),
            description=f"Red suitcase" if i % 2 == 0 else "Blue backpack",
            person_id=f"PERSON_{i}"
        )
        search_engine.add_baggage(profile)
        print(f"  Added: {profile.bag_id} - {profile.description}")
    
    # Search by description
    print("\nSearching by description: 'suitcase'")
    results = search_engine.search_by_description("suitcase", top_k=3)
    for i, profile in enumerate(results, 1):
        print(f"  {i}. {profile.bag_id}: {profile.description}")


def example_6_detection_statistics():
    """Example 6: System statistics and tracking"""
    print("\n" + "="*60)
    print("EXAMPLE 6: System Statistics")
    print("="*60)
    
    baggage_linking = BaggageLinking()
    
    # Process multiple frames
    print("\nProcessing multiple frames...")
    for frame_id in range(3):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        # Add random objects
        for _ in range(np.random.randint(1, 4)):
            x1 = np.random.randint(0, 1000)
            y1 = np.random.randint(0, 500)
            cv2.rectangle(frame, (x1, y1), (x1+150, y1+150), 
                         (np.random.randint(0, 256), np.random.randint(0, 256), 
                          np.random.randint(0, 256)), -1)
        
        result = baggage_linking.process_frame(
            frame, camera_id="CAM_TEST", frame_id=frame_id
        )
        print(f"  Frame {frame_id}: {len(result['detections'])} detections, "
              f"{len(result['links'])} links")
    
    # Get statistics
    stats = baggage_linking.get_statistics()
    print("\nSystem Statistics:")
    print(f"  Total bags: {stats['total_bags']}")
    print(f"  Total links: {stats['total_links']}")
    print(f"  Total mismatches: {stats['total_mismatches']}")
    print(f"  Cameras: {stats['cameras']}")
    print(f"  Timestamp: {stats['timestamp']}")


def example_7_hash_id_generation():
    """Example 7: Hash ID generation for baggage"""
    print("\n" + "="*60)
    print("EXAMPLE 7: Hash ID Generation")
    print("="*60)
    
    from vision import HashIDGenerator
    
    print("\nGenerating hash IDs for baggage...")
    
    # Create sample detections
    for i in range(3):
        embedding = np.random.randn(512)
        detection = Detection(
            class_name=ObjectClass.SUITCASE,
            bbox=BoundingBox(100 + i*50, 100 + i*50, 300 + i*50, 300 + i*50),
            confidence=0.85 + i*0.01,
            embedding=embedding,
            color_histogram=ColorHistogram(),
            camera_id="CAM_001",
            frame_id=i
        )
        
        hash_id = HashIDGenerator.generate_hash_id(detection)
        bag_id = HashIDGenerator.generate_bag_id("CAM_001", i, 0)
        
        print(f"\nBaggage {i+1}:")
        print(f"  Bag ID: {bag_id}")
        print(f"  Hash ID: {hash_id}")
        print(f"  Class: {detection.class_name.value}")
        print(f"  Confidence: {detection.confidence:.2%}")


def example_8_bounding_box_operations():
    """Example 8: Bounding box geometry operations"""
    print("\n" + "="*60)
    print("EXAMPLE 8: Bounding Box Operations")
    print("="*60)
    
    # Create sample bounding boxes
    bbox1 = BoundingBox(100, 100, 300, 400)  # Person
    bbox2 = BoundingBox(320, 350, 380, 450)  # Bag nearby
    bbox3 = BoundingBox(600, 100, 800, 300)  # Bag far away
    
    print("\nBounding Box 1 (Person):")
    print(f"  Position: ({bbox1.x1:.0f}, {bbox1.y1:.0f}, {bbox1.x2:.0f}, {bbox1.y2:.0f})")
    print(f"  Size: {bbox1.width():.0f}x{bbox1.height():.0f}")
    print(f"  Area: {bbox1.area():.0f} pixels²")
    print(f"  Center: ({bbox1.center()[0]:.0f}, {bbox1.center()[1]:.0f})")
    
    print("\nBounding Box 2 (Bag - Close):")
    print(f"  Distance to Box 1: {bbox1.distance_to(bbox2):.1f} pixels")
    print(f"  IoU with Box 1: {bbox1.iou(bbox2):.3f}")
    
    print("\nBounding Box 3 (Bag - Far):")
    print(f"  Distance to Box 1: {bbox1.distance_to(bbox3):.1f} pixels")
    print(f"  IoU with Box 1: {bbox1.iou(bbox3):.3f}")


def run_all_examples():
    """Run all examples"""
    examples = [
        ("Basic Detection", example_1_basic_detection),
        ("Person-Bag Linking", example_2_person_bag_linking),
        ("Mismatch Detection", example_3_mismatch_detection),
        ("Color Histogram", example_4_color_histogram),
        ("Embedding Search", example_5_embedding_search),
        ("Statistics", example_6_detection_statistics),
        ("Hash Generation", example_7_hash_id_generation),
        ("Bbox Operations", example_8_bounding_box_operations),
    ]
    
    print("\n" + "="*60)
    print("VISION PIPELINE EXAMPLES")
    print("="*60)
    print(f"Total examples: {len(examples)}\n")
    
    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("ALL EXAMPLES COMPLETED")
    print("="*60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        example_num = int(sys.argv[1])
        examples = [
            example_1_basic_detection,
            example_2_person_bag_linking,
            example_3_mismatch_detection,
            example_4_color_histogram,
            example_5_embedding_search,
            example_6_detection_statistics,
            example_7_hash_id_generation,
            example_8_bounding_box_operations,
        ]
        if 1 <= example_num <= len(examples):
            examples[example_num - 1]()
        else:
            print(f"Invalid example. Choose 1-{len(examples)}")
    else:
        run_all_examples()
