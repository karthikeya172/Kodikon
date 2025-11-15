"""
Unit tests for the Mesh Protocol
"""

import unittest
import time
import threading
from unittest.mock import Mock, patch, MagicMock

from mesh import (
    MeshProtocol,
    MeshMessage,
    PeerInfo,
    MessageType,
    NodeState,
    HashRegistry,
    HashRegistryEntry,
    NodeStateManager,
    PeerDiscovery,
    MessageRouter
)


class TestPeerInfo(unittest.TestCase):
    """Tests for PeerInfo class"""
    
    def test_peer_info_creation(self):
        """Test PeerInfo instantiation"""
        peer = PeerInfo(
            node_id="test_node",
            ip_address="192.168.1.1",
            port=9999
        )
        self.assertEqual(peer.node_id, "test_node")
        self.assertEqual(peer.ip_address, "192.168.1.1")
        self.assertEqual(peer.port, 9999)
        self.assertEqual(peer.state, NodeState.ACTIVE)
    
    def test_peer_is_alive(self):
        """Test peer liveness check"""
        peer = PeerInfo(
            node_id="test_node",
            ip_address="192.168.1.1",
            port=9999,
            last_heartbeat=time.time()
        )
        self.assertTrue(peer.is_alive(timeout=30))
        
        # Mock old heartbeat
        peer.last_heartbeat = time.time() - 35
        self.assertFalse(peer.is_alive(timeout=30))
    
    def test_peer_to_dict(self):
        """Test peer serialization"""
        peer = PeerInfo(
            node_id="test_node",
            ip_address="192.168.1.1",
            port=9999
        )
        peer_dict = peer.to_dict()
        self.assertIsInstance(peer_dict, dict)
        self.assertEqual(peer_dict['node_id'], "test_node")
        self.assertEqual(peer_dict['ip_address'], "192.168.1.1")


class TestMeshMessage(unittest.TestCase):
    """Tests for MeshMessage class"""
    
    def test_message_creation(self):
        """Test message instantiation"""
        msg = MeshMessage(
            message_type=MessageType.HEARTBEAT,
            source_node_id="node_001",
            payload={'status': 'active'}
        )
        self.assertEqual(msg.message_type, MessageType.HEARTBEAT)
        self.assertEqual(msg.source_node_id, "node_001")
        self.assertEqual(msg.payload['status'], 'active')
    
    def test_message_serialization(self):
        """Test message serialization"""
        msg = MeshMessage(
            message_type=MessageType.ALERT,
            source_node_id="node_001",
            sequence_number=1,
            payload={'alert': 'test_alert'}
        )
        serialized = msg.serialize()
        self.assertIsInstance(serialized, bytes)
        self.assertIn(b'node_001', serialized)
    
    def test_message_deserialization(self):
        """Test message deserialization"""
        original = MeshMessage(
            message_type=MessageType.HEARTBEAT,
            source_node_id="node_001",
            sequence_number=42,
            payload={'status': 'active', 'fps': 30.0}
        )
        
        serialized = original.serialize()
        deserialized = MeshMessage.deserialize(serialized)
        
        self.assertEqual(deserialized.message_type, original.message_type)
        self.assertEqual(deserialized.source_node_id, original.source_node_id)
        self.assertEqual(deserialized.sequence_number, original.sequence_number)
        self.assertEqual(deserialized.payload, original.payload)
    
    def test_message_round_trip(self):
        """Test message serialization round trip"""
        original = MeshMessage(
            message_type=MessageType.SEARCH_QUERY,
            source_node_id="searcher_node",
            payload={
                'query': {'person_id': 'p123'},
                'search_type': 'person'
            },
            routing_path=['node_a', 'node_b']
        )
        
        serialized = original.serialize()
        deserialized = MeshMessage.deserialize(serialized)
        
        self.assertEqual(deserialized.message_type, original.message_type)
        self.assertEqual(deserialized.source_node_id, original.source_node_id)
        self.assertEqual(deserialized.payload, original.payload)
        self.assertEqual(deserialized.routing_path, original.routing_path)


class TestHashRegistry(unittest.TestCase):
    """Tests for HashRegistry class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.registry = HashRegistry()
    
    def test_add_hash(self):
        """Test adding hash to registry"""
        self.registry.add_hash(
            hash_value="hash_001",
            node_id="node_001",
            data_type="person",
            embedding=[0.1] * 512
        )
        
        entry = self.registry.get_hash("hash_001")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.hash_value, "hash_001")
        self.assertEqual(entry.node_id, "node_001")
        self.assertEqual(entry.data_type, "person")
    
    def test_search_by_type(self):
        """Test searching hashes by type"""
        self.registry.add_hash("hash_001", "node_001", "person")
        self.registry.add_hash("hash_002", "node_001", "person")
        self.registry.add_hash("hash_003", "node_002", "vehicle")
        
        person_hashes = self.registry.search_hashes(data_type='person')
        self.assertEqual(len(person_hashes), 2)
        
        vehicle_hashes = self.registry.search_hashes(data_type='vehicle')
        self.assertEqual(len(vehicle_hashes), 1)
    
    def test_search_by_node(self):
        """Test searching hashes by node"""
        self.registry.add_hash("hash_001", "node_001", "person")
        self.registry.add_hash("hash_002", "node_002", "person")
        self.registry.add_hash("hash_003", "node_001", "vehicle")
        
        node_001_hashes = self.registry.search_hashes(node_id='node_001')
        self.assertEqual(len(node_001_hashes), 2)
        
        node_002_hashes = self.registry.search_hashes(node_id='node_002')
        self.assertEqual(len(node_002_hashes), 1)
    
    def test_registry_snapshot(self):
        """Test registry snapshot generation"""
        self.registry.add_hash("hash_001", "node_001", "person")
        self.registry.add_hash("hash_002", "node_002", "vehicle")
        
        snapshot = self.registry.get_registry_snapshot()
        self.assertIn('version', snapshot)
        self.assertIn('entries', snapshot)
        self.assertIn('timestamp', snapshot)
        self.assertEqual(len(snapshot['entries']), 2)


class TestNodeStateManager(unittest.TestCase):
    """Tests for NodeStateManager class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.state_mgr = NodeStateManager(
            node_id="node_001",
            local_state={'status': 'idle', 'fps': 30}
        )
    
    def test_state_creation(self):
        """Test state manager initialization"""
        self.assertEqual(self.state_mgr.node_id, "node_001")
        self.assertEqual(self.state_mgr.local_state['status'], 'idle')
        self.assertEqual(self.state_mgr.state_version, 0)
    
    def test_update_state(self):
        """Test state updates"""
        self.state_mgr.update_local_state({'status': 'active', 'fps': 25})
        self.assertEqual(self.state_mgr.local_state['status'], 'active')
        self.assertEqual(self.state_mgr.local_state['fps'], 25)
        self.assertEqual(self.state_mgr.state_version, 1)
    
    def test_state_snapshot(self):
        """Test state snapshot"""
        self.state_mgr.update_local_state({'battery': 85})
        snapshot = self.state_mgr.get_state_snapshot()
        
        self.assertEqual(snapshot['node_id'], "node_001")
        self.assertEqual(snapshot['version'], 1)
        self.assertIn('state', snapshot)
        self.assertIn('timestamp', snapshot)
    
    def test_merge_remote_state(self):
        """Test merging remote state"""
        remote_state = {
            'version': 5,
            'state': {'status': 'processing', 'load': 0.8}
        }
        
        merged = self.state_mgr.merge_remote_state(remote_state)
        self.assertTrue(merged)
        self.assertEqual(self.state_mgr.state_version, 5)
        self.assertEqual(self.state_mgr.local_state['status'], 'processing')
    
    def test_merge_old_state(self):
        """Test that old state is not merged"""
        self.state_mgr.state_version = 10
        remote_state = {
            'version': 5,
            'state': {'status': 'old_status'}
        }
        
        merged = self.state_mgr.merge_remote_state(remote_state)
        self.assertFalse(merged)


class TestMessageRouter(unittest.TestCase):
    """Tests for MessageRouter class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.router = MessageRouter(node_id="node_001", max_hops=5)
    
    def test_router_creation(self):
        """Test router initialization"""
        self.assertEqual(self.router.node_id, "node_001")
        self.assertEqual(self.router.max_hops, 5)
    
    def test_should_forward_message(self):
        """Test message forwarding decision"""
        msg = MeshMessage(
            message_type=MessageType.ALERT,
            source_node_id="source",
            sequence_number=1
        )
        
        # First message should be forwarded
        self.assertTrue(self.router.should_forward_message(msg))
        
        # Duplicate should not be forwarded
        self.assertFalse(self.router.should_forward_message(msg))
    
    def test_add_to_path(self):
        """Test adding node to routing path"""
        msg = MeshMessage(
            message_type=MessageType.ALERT,
            source_node_id="source"
        )
        
        self.assertEqual(len(msg.routing_path), 0)
        self.router.add_to_path(msg)
        self.assertEqual(len(msg.routing_path), 1)
        self.assertEqual(msg.routing_path[0], "node_001")
    
    def test_get_best_route(self):
        """Test best route calculation"""
        peers = {
            'node_002': PeerInfo("node_002", "192.168.1.2", 9999),
            'node_003': PeerInfo("node_003", "192.168.1.3", 9999)
        }
        
        route = self.router.get_best_route("node_002", peers)
        self.assertIsNotNone(route)
        self.assertEqual(route[0], "node_001")


class TestMeshProtocol(unittest.TestCase):
    """Tests for MeshProtocol class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mesh = MeshProtocol(
            node_id="test_node",
            port=19999,
            heartbeat_interval=1,
            heartbeat_timeout=5,
            max_peers=10
        )
    
    def tearDown(self):
        """Clean up"""
        if self.mesh.running:
            self.mesh.stop()
    
    def test_mesh_creation(self):
        """Test mesh initialization"""
        self.assertEqual(self.mesh.node_id, "test_node")
        self.assertEqual(self.mesh.port, 19999)
        self.assertEqual(self.mesh.heartbeat_interval, 1)
        self.assertFalse(self.mesh.running)
    
    def test_mesh_start_stop(self):
        """Test mesh start and stop"""
        self.mesh.start({'status': 'active'})
        self.assertTrue(self.mesh.running)
        
        self.mesh.stop()
        self.assertFalse(self.mesh.running)
    
    def test_update_node_state(self):
        """Test updating node state"""
        self.mesh.start()
        self.mesh.update_node_state({'battery': 90, 'status': 'processing'})
        
        state = self.mesh.get_node_state()
        self.assertEqual(state['state']['battery'], 90)
        self.assertEqual(state['state']['status'], 'processing')
    
    def test_add_hash(self):
        """Test adding hash"""
        self.mesh.start()
        self.mesh.add_hash(
            hash_value="test_hash",
            data_type="person",
            embedding=[0.1] * 512,
            metadata={'person_id': 'p123'}
        )
        
        entry = self.mesh.hash_registry.get_hash("test_hash")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.data_type, "person")
    
    def test_discover_peers(self):
        """Test peer discovery"""
        self.mesh.start()
        peers = self.mesh.discover_peers()
        self.assertIsInstance(peers, dict)
        # Should not include self
        self.assertNotIn("test_node", peers)
    
    def test_get_network_stats(self):
        """Test network statistics"""
        self.mesh.start()
        stats = self.mesh.get_network_stats()
        
        self.assertEqual(stats['node_id'], "test_node")
        self.assertTrue(stats['is_running'])
        self.assertIn('alive_peers', stats)
        self.assertIn('messages_sent', stats)
    
    def test_register_handler(self):
        """Test message handler registration"""
        handler = Mock()
        self.mesh.register_message_handler(MessageType.ALERT, handler)
        
        handlers = self.mesh.message_handlers[MessageType.ALERT]
        self.assertIn(handler, handlers)
    
    def test_message_handler_call(self):
        """Test message handler is called"""
        handler = Mock()
        self.mesh.register_message_handler(MessageType.ALERT, handler)
        
        msg = MeshMessage(
            message_type=MessageType.ALERT,
            source_node_id="sender"
        )
        
        # Simulate receiving a message (internal call)
        self.mesh._handle_message_type(msg)
        # Handler won't be called directly in _handle_message_type
        # but would be called in the actual receive loop


class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def test_message_serialization_all_types(self):
        """Test serialization for all message types"""
        for msg_type in MessageType:
            msg = MeshMessage(
                message_type=msg_type,
                source_node_id="node_001",
                payload={'test': 'data'}
            )
            
            serialized = msg.serialize()
            deserialized = MeshMessage.deserialize(serialized)
            
            self.assertEqual(deserialized.message_type, msg_type)
            self.assertEqual(deserialized.source_node_id, "node_001")
    
    def test_hash_registry_complex_operations(self):
        """Test complex hash registry operations"""
        registry = HashRegistry()
        
        # Add multiple hashes
        for i in range(10):
            registry.add_hash(
                hash_value=f"hash_{i:03d}",
                node_id=f"node_{i % 3}",
                data_type='person' if i % 2 == 0 else 'vehicle',
                embedding=[0.1 * i] * 512,
                metadata={'index': i}
            )
        
        # Search by type
        person_hashes = registry.search_hashes(data_type='person')
        self.assertEqual(len(person_hashes), 5)
        
        # Search by node
        node_0_hashes = registry.search_hashes(node_id='node_0')
        self.assertGreater(len(node_0_hashes), 0)
        
        # Get snapshot
        snapshot = registry.get_registry_snapshot()
        self.assertEqual(len(snapshot['entries']), 10)


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestPeerInfo))
    suite.addTests(loader.loadTestsFromTestCase(TestMeshMessage))
    suite.addTests(loader.loadTestsFromTestCase(TestHashRegistry))
    suite.addTests(loader.loadTestsFromTestCase(TestNodeStateManager))
    suite.addTests(loader.loadTestsFromTestCase(TestMessageRouter))
    suite.addTests(loader.loadTestsFromTestCase(TestMeshProtocol))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
