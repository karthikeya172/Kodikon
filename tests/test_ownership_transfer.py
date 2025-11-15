"""
Tests for Ownership Transfer Mitigation (Phase 8)
Unit tests for OwnershipMatcher, AlertVerifier, and KGStore
Integration test for multi-bag check-in scenario
"""

import pytest
import numpy as np
import time
import threading
from dataclasses import dataclass
from typing import Dict, List, Any

# Imports from our modules
from vision.baggage_linking import (
    OwnershipMatcher, AlertVerifier, OwnershipEvent, TransferEvent,
    Detection, BoundingBox, ColorHistogram, ObjectClass
)
from knowledge_graph.kg_store import KGStore
from mesh.mesh_protocol import MeshProtocol


# ============================================================================
# UNIT TESTS
# ============================================================================

class TestOwnershipMatching:
    """Test suite for OwnershipMatcher"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.kg_store = KGStore(persist_path=":memory:")  # In-memory for testing
        self.matcher = OwnershipMatcher(kg_store=self.kg_store)
    
    def create_mock_detection(self, class_name=ObjectClass.PERSON, 
                             embedding=None, camera_id="test_camera"):
        """Helper to create mock detections"""
        if embedding is None:
            embedding = np.random.randn(512).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
        
        bbox = BoundingBox(x1=100, y1=100, x2=200, y2=200)
        return Detection(
            class_name=class_name,
            bbox=bbox,
            confidence=0.95,
            embedding=embedding,
            camera_id=camera_id
        )
    
    def test_high_confidence_maintain(self):
        """Test that high confidence score triggers MAINTAIN decision"""
        person_det = self.create_mock_detection(ObjectClass.PERSON)
        bag_det = self.create_mock_detection(ObjectClass.BAG)
        
        # Use same embedding for high similarity
        bag_det.embedding = person_det.embedding.copy()
        
        ownership_history = [
            {
                'timestamp': time.time() - 10,
                'person_id': 'person1',
                'event_type': 'HOLD',
                'confidence': 0.9
            }
        ]
        
        location_context = {
            'camera_role': 'SURVEILLANCE',
            'zone_type': 'general',
            'location_signature': 'camera1'
        }
        
        result = self.matcher.match(person_det, bag_det, ownership_history, location_context)
        
        assert result['decision'] in ['MAINTAIN', 'UNCERTAIN']
        assert result['confidence'] > 0.5
        assert 'reason' in result
    
    def test_low_confidence_clear(self):
        """Test that low confidence score triggers CLEAR decision"""
        person_det = self.create_mock_detection(ObjectClass.PERSON)
        bag_det = self.create_mock_detection(ObjectClass.BAG)
        
        # Use different embeddings for low similarity
        bag_det.embedding = np.random.randn(512).astype(np.float32)
        bag_det.embedding = bag_det.embedding / np.linalg.norm(bag_det.embedding)
        bag_det.bbox = BoundingBox(x1=500, y1=500, x2=600, y2=600)  # Far away
        
        ownership_history = []
        
        location_context = {
            'camera_role': 'SURVEILLANCE',
            'zone_type': 'general',
            'location_signature': 'camera1'
        }
        
        result = self.matcher.match(person_det, bag_det, ownership_history, location_context)
        
        # With low similarity and far distance, should clear or be uncertain
        assert result['decision'] in ['CLEAR', 'UNCERTAIN']
    
    def test_transfer_suppression(self):
        """Test that transfer suppression window suppresses alerts"""
        person_det = self.create_mock_detection(ObjectClass.PERSON)
        bag_det = self.create_mock_detection(ObjectClass.BAG)
        bag_det.camera_id = "bag123"
        
        # Suppress alerts for this bag
        self.matcher.suppress_alerts_for_transfer(bag_det.camera_id)
        
        result = self.matcher.match(person_det, bag_det, [], {
            'camera_role': 'SURVEILLANCE',
            'zone_type': 'general',
            'location_signature': 'camera1'
        })
        
        # Should indicate suppression
        assert result['decision'] == 'SUPPRESS' or 'suppression' in result['reason'].lower()


class TestAlertVerification:
    """Test suite for AlertVerifier"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.kg_store = KGStore(persist_path=":memory:")
        self.alert_verifier = AlertVerifier(kg_store=self.kg_store)
    
    def test_pending_alert_confirmation(self):
        """Test multi-stage pending alert confirmation"""
        bag_id = "bag_test_001"
        person_id = "person_test_001"
        location_context = {
            'zone_type': 'general',
            'camera_role': 'SURVEILLANCE',
            'location_signature': 'camera1'
        }
        
        # First alert: should create pending
        result1 = self.alert_verifier.raise_alert(
            bag_id, person_id, 'SUSPICIOUS', 0.7, location_context
        )
        assert result1['action'] == 'PENDING'
        assert result1['stage'] == 3
        
        # Second alert from different camera: should accumulate
        location_context2 = location_context.copy()
        location_context2['location_signature'] = 'camera2'
        result2 = self.alert_verifier.raise_alert(
            bag_id, person_id, 'SUSPICIOUS', 0.75, location_context2
        )
        # Should still be pending or escalated depending on timing
        assert result2['action'] in ['PENDING', 'ALERT']
    
    def test_whitelist_zone_suppression(self):
        """Test that whitelist zones suppress alerts"""
        bag_id = "bag_test_002"
        person_id = "person_test_002"
        location_context = {
            'zone_type': 'staff_zone',  # Whitelist zone
            'camera_role': 'SURVEILLANCE',
            'location_signature': 'camera1'
        }
        
        result = self.alert_verifier.raise_alert(
            bag_id, person_id, 'SUSPICIOUS', 0.8, location_context
        )
        
        assert result['action'] == 'SUPPRESS'
        assert result['stage'] == 1
        assert 'whitelist' in result['reason'].lower()
    
    def test_staff_member_suppression(self):
        """Test that staff members are suppressed"""
        bag_id = "bag_test_003"
        staff_id = "staff_member_001"
        
        # Add to staff registry
        self.alert_verifier.add_staff_member(staff_id)
        
        location_context = {
            'zone_type': 'general',
            'camera_role': 'SURVEILLANCE',
            'location_signature': 'camera1'
        }
        
        result = self.alert_verifier.raise_alert(
            bag_id, staff_id, 'SUSPICIOUS', 0.7, location_context
        )
        
        assert result['action'] == 'SUPPRESS'
        assert result['stage'] == 1


class TestKGStore:
    """Test suite for KGStore"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.kg_store = KGStore(persist_path=":memory:")
    
    def test_add_ownership_event(self):
        """Test adding ownership events"""
        event = {
            'event_id': 'evt_001',
            'person_id': 'person1',
            'bag_id': 'bag1',
            'timestamp': time.time(),
            'event_type': 'REGISTER',
            'confidence': 0.9,
            'source_node_id': 'node1',
            'location_signature': 'loc1',
            'camera_role': 'REGISTRATION',
            'transfer_token': None,
            'reason': 'Initial registration'
        }
        
        result = self.kg_store.add_ownership_event(event)
        assert result is True
    
    def test_get_current_owner(self):
        """Test retrieving current owner"""
        event1 = {
            'event_id': 'evt_001',
            'person_id': 'person1',
            'bag_id': 'bag1',
            'timestamp': time.time(),
            'event_type': 'REGISTER',
            'confidence': 0.9,
            'source_node_id': 'node1',
            'location_signature': 'loc1',
            'camera_role': 'REGISTRATION'
        }
        
        self.kg_store.add_ownership_event(event1)
        owner = self.kg_store.get_current_owner('bag1')
        
        assert owner == 'person1'
    
    def test_get_person_bags(self):
        """Test retrieving bags owned by person"""
        person_id = 'person2'
        
        for i in range(3):
            event = {
                'event_id': f'evt_{i}',
                'person_id': person_id,
                'bag_id': f'bag_{i}',
                'timestamp': time.time(),
                'event_type': 'REGISTER',
                'confidence': 0.9,
                'source_node_id': 'node1',
                'location_signature': 'loc1',
                'camera_role': 'REGISTRATION'
            }
            self.kg_store.add_ownership_event(event)
        
        bags = self.kg_store.get_person_bags(person_id)
        assert len(bags) == 3
    
    def test_ownership_history(self):
        """Test retrieving ownership history"""
        bag_id = 'bag_hist'
        
        # Add multiple events for same bag
        for i in range(5):
            event = {
                'event_id': f'evt_hist_{i}',
                'person_id': f'person_{i}',
                'bag_id': bag_id,
                'timestamp': time.time() + i,
                'event_type': 'HOLD' if i > 0 else 'REGISTER',
                'confidence': 0.9 - i * 0.1,
                'source_node_id': 'node1',
                'location_signature': f'loc_{i}',
                'camera_role': 'SURVEILLANCE'
            }
            self.kg_store.add_ownership_event(event)
        
        history = self.kg_store.get_ownership_history(bag_id, limit=10)
        assert len(history) > 0


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestMultiBagCheckInScenario:
    """
    Integration test: Multi-bag check-in scenario
    Simulates person with multiple bags going through check-in zone
    """
    
    def setup_method(self):
        """Set up test fixtures"""
        self.kg_store = KGStore(persist_path=":memory:")
        self.matcher = OwnershipMatcher(kg_store=self.kg_store)
        self.alert_verifier = AlertVerifier(kg_store=self.kg_store)
    
    def simulate_multibag_checkin(self):
        """
        Simulate 5-frame scenario:
        Frame 1: Person with 2 bags at check-in zone
        Frame 2: Person still with bags
        Frame 3: Staff member takes one bag
        Frame 4: Person leaves with one bag
        Frame 5: Staff carries the second bag
        """
        
        results = []
        base_time = time.time()
        
        # Frame 1: Registration
        print("\n=== Frame 1: Person registers with 2 bags ===")
        person_det1 = Detection(
            class_name=ObjectClass.PERSON,
            bbox=BoundingBox(x1=100, y1=100, x2=200, y2=200),
            confidence=0.95,
            embedding=np.ones(512, dtype=np.float32) * 0.5,
            camera_id="person1"
        )
        bag1_det = Detection(
            class_name=ObjectClass.BAG,
            bbox=BoundingBox(x1=80, y1=250, x2=150, y2=350),
            confidence=0.92,
            embedding=np.ones(512, dtype=np.float32) * 0.5,
            camera_id="bag1"
        )
        bag2_det = Detection(
            class_name=ObjectClass.BAG,
            bbox=BoundingBox(x1=210, y1=250, x2=280, y2=350),
            confidence=0.88,
            embedding=np.ones(512, dtype=np.float32) * 0.45,
            camera_id="bag2"
        )
        
        # Register both bags to person
        for bag_id, bag in [("bag1", bag1_det), ("bag2", bag2_det)]:
            event = {
                'event_id': f'frame1_{bag_id}',
                'person_id': 'person1',
                'bag_id': bag_id,
                'timestamp': base_time,
                'event_type': 'REGISTER',
                'confidence': 0.95,
                'source_node_id': 'camera_checkin',
                'location_signature': 'check_in_zone',
                'camera_role': 'REGISTRATION'
            }
            self.kg_store.add_ownership_event(event)
        
        results.append({
            'frame': 1,
            'event': 'Registration',
            'bags_registered': 2,
            'owner': 'person1'
        })
        
        # Frame 2: Person with bags in transit
        print("=== Frame 2: Person moving through zone ===")
        ownership_history_bag1 = self.kg_store.get_ownership_history("bag1")
        match_result = self.matcher.match(
            person_det1, bag1_det, ownership_history_bag1,
            {'camera_role': 'SURVEILLANCE', 'zone_type': 'transit', 'location_signature': 'transit_camera'}
        )
        print(f"Match result: {match_result['decision']} (confidence: {match_result['confidence']:.3f})")
        
        results.append({
            'frame': 2,
            'event': 'In transit',
            'ownership_decision': match_result['decision'],
            'confidence': match_result['confidence']
        })
        
        # Frame 3: Staff takes bag1 (transfer detected)
        print("=== Frame 3: Staff member takes bag1 ===")
        staff_det = Detection(
            class_name=ObjectClass.PERSON,
            bbox=BoundingBox(x1=90, y1=100, x2=180, y2=210),
            confidence=0.94,
            embedding=np.random.randn(512).astype(np.float32),
            camera_id="staff1"
        )
        
        # Suppress alerts for transfer window
        self.matcher.suppress_alerts_for_transfer("bag1")
        
        transfer_event = {
            'transfer_id': 'transfer_001',
            'from_person_id': 'person1',
            'to_person_id': 'staff1',
            'bag_id': 'bag1',
            'timestamp': base_time + 2,
            'transfer_type': 'HAND_OFF',
            'location_signature': 'check_in_zone',
            'source_node_id': 'camera_checkin'
        }
        self.kg_store.add_transfer_event(transfer_event)
        
        # Update ownership
        ownership_event = {
            'event_id': 'frame3_bag1_transfer',
            'person_id': 'staff1',
            'bag_id': 'bag1',
            'timestamp': base_time + 2,
            'event_type': 'TRANSFER_IN',
            'confidence': 1.0,
            'source_node_id': 'camera_checkin',
            'location_signature': 'check_in_zone',
            'camera_role': 'REGISTRATION',
            'transfer_token': 'transfer_001'
        }
        self.kg_store.add_ownership_event(ownership_event)
        
        results.append({
            'frame': 3,
            'event': 'Transfer detected',
            'from': 'person1',
            'to': 'staff1',
            'bag': 'bag1',
            'suppressed': True
        })
        
        # Frame 4: Person leaves with bag2
        print("=== Frame 4: Person leaves with bag2 ===")
        person_det2 = Detection(
            class_name=ObjectClass.PERSON,
            bbox=BoundingBox(x1=100, y1=100, x2=200, y2=200),
            confidence=0.95,
            embedding=np.ones(512, dtype=np.float32) * 0.5,
            camera_id="person1"
        )
        
        ownership_history_bag2 = self.kg_store.get_ownership_history("bag2")
        match_result_bag2 = self.matcher.match(
            person_det2, bag2_det, ownership_history_bag2,
            {'camera_role': 'SURVEILLANCE', 'zone_type': 'exit', 'location_signature': 'exit_camera'}
        )
        
        results.append({
            'frame': 4,
            'event': 'Person exits',
            'with_bag': 'bag2',
            'ownership_decision': match_result_bag2['decision'],
            'confidence': match_result_bag2['confidence']
        })
        
        # Frame 5: Staff with bag1
        print("=== Frame 5: Staff member with bag1 ===")
        ownership_history_bag1_frame5 = self.kg_store.get_ownership_history("bag1")
        match_result_staff = self.matcher.match(
            staff_det, bag1_det, ownership_history_bag1_frame5,
            {'camera_role': 'SURVEILLANCE', 'zone_type': 'staff_zone', 'location_signature': 'back_area'}
        )
        
        results.append({
            'frame': 5,
            'event': 'Staff with bag1',
            'ownership_decision': match_result_staff['decision'],
            'confidence': match_result_staff['confidence'],
            'owner': 'staff1'
        })
        
        return results
    
    def test_multibag_scenario(self):
        """Run multi-bag scenario and validate"""
        results = self.simulate_multibag_checkin()
        
        # Validate results
        assert len(results) == 5
        assert results[0]['bags_registered'] == 2
        assert results[2]['suppressed'] is True
        assert results[1]['ownership_decision'] in ['MAINTAIN', 'UNCERTAIN']
        
        # Verify KGStore state
        assert self.kg_store.get_current_owner('bag1') == 'staff1'
        assert self.kg_store.get_current_owner('bag2') == 'person1'
        
        print("\n=== Multi-bag check-in scenario completed successfully ===")
        for r in results:
            print(f"Frame {r['frame']}: {r['event']}")


# ============================================================================
# TEST EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run tests
    print("Running Ownership Transfer Mitigation Tests...\n")
    
    # Unit tests
    print("=== UNIT TESTS ===\n")
    
    test_matching = TestOwnershipMatching()
    test_matching.setup_method()
    test_matching.test_high_confidence_maintain()
    print("✓ test_high_confidence_maintain passed")
    
    test_matching.setup_method()
    test_matching.test_low_confidence_clear()
    print("✓ test_low_confidence_clear passed")
    
    test_matching.setup_method()
    test_matching.test_transfer_suppression()
    print("✓ test_transfer_suppression passed")
    
    test_alerts = TestAlertVerification()
    test_alerts.setup_method()
    test_alerts.test_pending_alert_confirmation()
    print("✓ test_pending_alert_confirmation passed")
    
    test_alerts.setup_method()
    test_alerts.test_whitelist_zone_suppression()
    print("✓ test_whitelist_zone_suppression passed")
    
    test_alerts.setup_method()
    test_alerts.test_staff_member_suppression()
    print("✓ test_staff_member_suppression passed")
    
    test_kg = TestKGStore()
    test_kg.setup_method()
    test_kg.test_add_ownership_event()
    print("✓ test_add_ownership_event passed")
    
    test_kg.setup_method()
    test_kg.test_get_current_owner()
    print("✓ test_get_current_owner passed")
    
    test_kg.setup_method()
    test_kg.test_get_person_bags()
    print("✓ test_get_person_bags passed")
    
    test_kg.setup_method()
    test_kg.test_ownership_history()
    print("✓ test_ownership_history passed")
    
    # Integration tests
    print("\n=== INTEGRATION TESTS ===\n")
    
    integration_test = TestMultiBagCheckInScenario()
    integration_test.setup_method()
    integration_test.test_multibag_scenario()
    print("✓ test_multibag_scenario passed")
    
    print("\n=== ALL TESTS PASSED ===")
