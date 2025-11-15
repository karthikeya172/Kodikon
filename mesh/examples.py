"""
Mesh Protocol Examples and Usage Demonstrations
"""

import time
import logging
from mesh import (
    MeshProtocol, 
    MessageType, 
    NodeState,
    MeshMessage,
    PeerInfo
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_1_basic_setup():
    """Example 1: Basic mesh network setup"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Mesh Network Setup")
    print("="*60)
    
    # Create mesh protocol instance
    mesh = MeshProtocol(
        node_id="node_001",
        port=9999,
        heartbeat_interval=5,
        heartbeat_timeout=30,
        max_peers=10
    )
    
    # Define local node state
    local_state = {
        'processing_fps': 30.0,
        'battery_level': 85,
        'embedding_dim': 512,
        'reid_model': 'osnet_x1_0',
        'yolo_model': 'yolov8n',
        'location': 'entrance_camera'
    }
    
    # Start the mesh
    mesh.start(local_state)
    logger.info("Mesh protocol started successfully")
    
    # Let it run for a bit
    time.sleep(3)
    
    # Get stats
    stats = mesh.get_network_stats()
    print(f"\nNetwork Statistics:")
    print(f"  Node ID: {stats['node_id']}")
    print(f"  Is Running: {stats['is_running']}")
    print(f"  Alive Peers: {stats['alive_peers']}")
    print(f"  Total Peers: {stats['total_peers']}")
    print(f"  Messages Sent: {stats['messages_sent']}")
    print(f"  Heartbeats Sent: {stats['heartbeats_sent']}")
    
    mesh.stop()
    print("\nMesh stopped")


def example_2_peer_discovery():
    """Example 2: Peer discovery and monitoring"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Peer Discovery and Monitoring")
    print("="*60)
    
    mesh = MeshProtocol(node_id="node_discovery", port=9998)
    mesh.start()
    
    logger.info("Starting peer discovery simulation...")
    time.sleep(5)
    
    # Get discovered peers
    peers = mesh.discover_peers()
    print(f"\nDiscovered Peers ({len(peers)}):")
    for node_id, peer_info in peers.items():
        alive = peer_info.is_alive(30)
        print(f"  - {node_id}:")
        print(f"    Address: {peer_info.ip_address}:{peer_info.port}")
        print(f"    State: {peer_info.state.name}")
        print(f"    Alive: {alive}")
        print(f"    Last Heartbeat: {time.time() - peer_info.last_heartbeat:.2f}s ago")
    
    mesh.stop()


def example_3_state_management():
    """Example 3: Node state management and synchronization"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Node State Management")
    print("="*60)
    
    mesh = MeshProtocol(node_id="node_state_mgr", port=9997)
    
    initial_state = {
        'status': 'idle',
        'processing_fps': 30.0,
        'battery_level': 100
    }
    
    mesh.start(initial_state)
    logger.info("Node started with initial state")
    
    # Get initial state
    state = mesh.get_node_state()
    print(f"\nInitial State:")
    print(f"  Version: {state['version']}")
    print(f"  State Data: {state['state']}")
    
    # Update state
    print(f"\nUpdating state...")
    mesh.update_node_state({
        'status': 'processing_search',
        'processing_fps': 25.5,
        'battery_level': 92,
        'active_searches': 1
    })
    
    # Get updated state
    state = mesh.get_node_state()
    print(f"\nUpdated State:")
    print(f"  Version: {state['version']}")
    print(f"  State Data: {state['state']}")
    
    time.sleep(2)
    mesh.stop()


def example_4_hash_registry():
    """Example 4: Distributed hash registry"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Distributed Hash Registry")
    print("="*60)
    
    mesh = MeshProtocol(node_id="node_hash_registry", port=9996)
    mesh.start()
    
    # Add hashes to registry
    print("\nAdding hashes to registry...")
    
    for i in range(5):
        hash_val = f"hash_{i:03d}_{'a'*20}"
        embedding = [0.1 * (i + 1)] * 512  # Simplified embedding
        
        mesh.add_hash(
            hash_value=hash_val,
            data_type='person' if i % 2 == 0 else 'vehicle',
            embedding=embedding,
            metadata={
                'confidence': 0.95 - (i * 0.05),
                'timestamp': time.time(),
                'location': f'zone_{i}'
            }
        )
        print(f"  Added hash {i+1}: {hash_val}")
    
    # Search hashes
    print(f"\nSearching hashes...")
    person_hashes = mesh.hash_registry.search_hashes(data_type='person')
    print(f"  Found {len(person_hashes)} person hashes")
    for entry in person_hashes:
        print(f"    - {entry.hash_value[:20]}... from {entry.node_id}")
    
    # Get registry snapshot
    snapshot = mesh.hash_registry.get_registry_snapshot()
    print(f"\nRegistry Snapshot:")
    print(f"  Version: {snapshot['version']}")
    print(f"  Entry Count: {len(snapshot['entries'])}")
    print(f"  Timestamp: {snapshot['timestamp']}")
    
    time.sleep(1)
    mesh.stop()


def example_5_alerts_and_broadcasts():
    """Example 5: Alert broadcasting"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Alert Broadcasting")
    print("="*60)
    
    mesh = MeshProtocol(node_id="node_alerts", port=9995)
    
    # Register alert handler
    alert_count = [0]
    
    def handle_alert(message):
        alert_count[0] += 1
        print(f"\n[ALERT RECEIVED] From: {message.source_node_id}")
        print(f"  Priority: {message.payload.get('priority')}")
        print(f"  Data: {message.payload.get('alert')}")
    
    mesh.register_message_handler(MessageType.ALERT, handle_alert)
    mesh.start()
    
    # Broadcast different priority alerts
    alerts = [
        {
            'type': 'security_alert',
            'description': 'Unauthorized person detected',
            'location': 'entrance_gate',
            'priority': 'critical'
        },
        {
            'type': 'system_alert',
            'description': 'Low battery on camera 2',
            'location': 'corridor_b',
            'priority': 'high'
        },
        {
            'type': 'notification',
            'description': 'Search completed successfully',
            'location': 'system',
            'priority': 'normal'
        }
    ]
    
    print("\nBroadcasting alerts...")
    for alert in alerts:
        priority = alert.pop('priority')
        mesh.broadcast_alert(alert, priority=priority)
        print(f"  Broadcast {alert['type']} with priority: {priority}")
        time.sleep(1)
    
    time.sleep(2)
    mesh.stop()


def example_6_search_queries():
    """Example 6: Search query broadcasting"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Search Query Broadcasting")
    print("="*60)
    
    mesh = MeshProtocol(node_id="node_search", port=9994)
    
    # Register search handler
    def handle_search(message):
        print(f"\n[SEARCH QUERY RECEIVED] From: {message.source_node_id}")
        print(f"  Type: {message.payload.get('search_type')}")
        print(f"  Query: {message.payload.get('query')}")
    
    mesh.register_message_handler(MessageType.SEARCH_QUERY, handle_search)
    mesh.start()
    
    # Create and broadcast search queries
    searches = [
        {
            'query': {
                'person_id': 'person_123',
                'time_range': (time.time() - 3600, time.time()),
                'location': 'building_a',
                'min_confidence': 0.8
            },
            'search_type': 'person_search'
        },
        {
            'query': {
                'vehicle_type': 'sedan',
                'color': 'black',
                'plate_pattern': 'ABC*',
                'time_range': (time.time() - 7200, time.time())
            },
            'search_type': 'vehicle_search'
        },
        {
            'query': {
                'embedding': [0.1] * 512,
                'similarity_threshold': 0.7,
                'limit': 10
            },
            'search_type': 'embedding_search'
        }
    ]
    
    print("\nBroadcasting searches...")
    for search in searches:
        mesh.broadcast_search_query(
            query=search['query'],
            search_type=search['search_type']
        )
        print(f"  Broadcast {search['search_type']}")
        time.sleep(1)
    
    time.sleep(2)
    mesh.stop()


def example_7_message_routing():
    """Example 7: Message routing through mesh"""
    print("\n" + "="*60)
    print("EXAMPLE 7: Message Routing")
    print("="*60)
    
    mesh = MeshProtocol(node_id="node_router", port=9993)
    mesh.start()
    
    # Create custom message
    message = MeshMessage(
        message_type=MessageType.ROUTE_BROADCAST,
        source_node_id="source_node",
        payload={
            'data': 'important_data',
            'destination': 'target_node',
            'priority': 'high'
        }
    )
    
    print(f"\nCreated message:")
    print(f"  Type: {message.message_type.name}")
    print(f"  Source: {message.source_node_id}")
    print(f"  Sequence: {message.sequence_number}")
    print(f"  Payload: {message.payload}")
    
    # Serialize
    serialized = message.serialize()
    print(f"\nSerialized message size: {len(serialized)} bytes")
    
    # Deserialize
    deserialized = MeshMessage.deserialize(serialized)
    print(f"\nDeserialized message:")
    print(f"  Type: {deserialized.message_type.name}")
    print(f"  Source: {deserialized.source_node_id}")
    print(f"  Payload: {deserialized.payload}")
    
    # Send message
    print(f"\nSending message...")
    mesh.send_message(message)
    
    time.sleep(2)
    mesh.stop()


def example_8_network_stats():
    """Example 8: Network statistics and monitoring"""
    print("\n" + "="*60)
    print("EXAMPLE 8: Network Statistics and Monitoring")
    print("="*60)
    
    mesh = MeshProtocol(node_id="node_stats", port=9992)
    mesh.start()
    
    print("\nSimulating network activity...")
    for i in range(3):
        # Send some messages
        for j in range(2):
            mesh.broadcast_alert({
                'type': 'test_alert',
                'index': f'{i}_{j}'
            })
        
        # Get stats
        stats = mesh.get_network_stats()
        print(f"\nStats (iteration {i+1}):")
        print(f"  Running: {stats['is_running']}")
        print(f"  Peers (alive/total): {stats['alive_peers']}/{stats['total_peers']}")
        print(f"  Messages Sent: {stats['messages_sent']}")
        print(f"  Messages Received: {stats['messages_received']}")
        print(f"  Messages Routed: {stats['messages_routed']}")
        print(f"  Heartbeats Sent: {stats['heartbeats_sent']}")
        print(f"  Peers Discovered: {stats['peers_discovered']}")
        
        time.sleep(2)
    
    mesh.stop()


def run_all_examples():
    """Run all examples"""
    examples = [
        ("Basic Setup", example_1_basic_setup),
        ("Peer Discovery", example_2_peer_discovery),
        ("State Management", example_3_state_management),
        ("Hash Registry", example_4_hash_registry),
        ("Alerts", example_5_alerts_and_broadcasts),
        ("Search Queries", example_6_search_queries),
        ("Message Routing", example_7_message_routing),
        ("Network Stats", example_8_network_stats),
    ]
    
    print("\n" + "="*60)
    print("MESH PROTOCOL EXAMPLES")
    print("="*60)
    print(f"Total examples: {len(examples)}\n")
    
    for name, func in examples:
        try:
            func()
            time.sleep(1)
        except Exception as e:
            logger.error(f"Error in {name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("ALL EXAMPLES COMPLETED")
    print("="*60)


if __name__ == "__main__":
    # Run individual example or all
    import sys
    
    if len(sys.argv) > 1:
        example_num = int(sys.argv[1])
        examples = [
            example_1_basic_setup,
            example_2_peer_discovery,
            example_3_state_management,
            example_4_hash_registry,
            example_5_alerts_and_broadcasts,
            example_6_search_queries,
            example_7_message_routing,
            example_8_network_stats,
        ]
        if 1 <= example_num <= len(examples):
            examples[example_num - 1]()
        else:
            print(f"Invalid example number. Choose 1-{len(examples)}")
    else:
        run_all_examples()
