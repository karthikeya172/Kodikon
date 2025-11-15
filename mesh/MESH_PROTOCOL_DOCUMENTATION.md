# Mesh Protocol Implementation Documentation

## Overview

The Mesh Protocol is a comprehensive peer-to-peer UDP-based network implementation designed for distributed node communication in the Kodikon system. It provides robust peer discovery, state synchronization, message routing, and distributed hash registry management.

## Architecture Components

### 1. **Core Classes**

#### `MeshProtocol` (Main Interface)
The central component managing all mesh operations:
- **Initialization**: `MeshProtocol(node_id, port=9999, heartbeat_interval=5, heartbeat_timeout=30, max_peers=10)`
- **Key Methods**:
  - `start(local_state=None)`: Initialize and start the mesh network
  - `stop()`: Gracefully shutdown the mesh
  - `send_message(message, target_ip, target_port)`: Send targeted or broadcast messages
  - `broadcast_alert(alert_data, priority)`: Send urgent alerts to all peers
  - `broadcast_search_query(query, search_type)`: Distribute search requests
  - `propagate_hash_registry()`: Sync hash registry across network
  - `discover_peers()`: Get list of discovered peers
  - `get_network_stats()`: Retrieve network statistics

#### `MeshMessage`
Serializable message structure for all network communication:
- **Fields**: message_type, source_node_id, timestamp, sequence_number, payload, routing_path
- **Methods**:
  - `serialize()`: Convert to bytes for transmission
  - `deserialize(data)`: Reconstruct from bytes

#### `PeerInfo`
Information container for discovered peers:
- **Fields**: node_id, ip_address, port, state, last_heartbeat, embedding_dim, reid_model, yolo_model, processing_fps, battery_level
- **Methods**:
  - `is_alive(timeout)`: Check if peer is responsive
  - `to_dict()`: Serialize for transmission

### 2. **Supporting Components**

#### `PeerDiscovery`
Handles peer detection and announcement:
- UDP broadcast-based discovery
- Automatic peer registration
- Configurable broadcast intervals

#### `NodeStateManager`
Manages local and remote node state:
- Versioned state snapshots
- Thread-safe state updates
- Remote state merging with conflict resolution

#### `HashRegistry`
Distributed hash registry for embeddings and object hashes:
- Thread-safe entry management
- Search by data type or node ID
- Registry versioning for consistency
- Snapshot generation for propagation

#### `MessageRouter`
Intelligent message routing through the mesh:
- Anti-loop detection (seen message tracking)
- Best-path routing to destinations
- Configurable hop limits (default: 5)
- Route caching for optimization

### 3. **Message Types**

```python
MessageType.HEARTBEAT          # Node status and aliveness beacon
MessageType.PEER_DISCOVERY     # Announcement of new nodes
MessageType.NODE_STATE_SYNC    # State updates and synchronization
MessageType.SEARCH_QUERY       # Distributed search requests
MessageType.ALERT              # Priority alerts to all peers
MessageType.HASH_REGISTRY      # Hash registry propagation
MessageType.ROUTE_BROADCAST    # Multi-hop broadcast
MessageType.ACK                # Message acknowledgment
```

### 4. **Node States**

```python
NodeState.ACTIVE       # Node processing normally
NodeState.IDLE         # Node idle/waiting
NodeState.PROCESSING   # Node actively processing
NodeState.OFFLINE      # Node offline/unreachable
```

## Features

### Peer Discovery
- **Automatic Discovery**: UDP broadcast every 5 seconds (configurable)
- **Dynamic Registration**: Peers added up to max limit (default: 10)
- **State Tracking**: Each peer maintains state, IP, port, and last heartbeat time

### Heartbeats & Liveness Monitoring
- **Periodic Heartbeats**: Sent at configured intervals with current node state
- **Timeout Detection**: Peers considered dead after `heartbeat_timeout` seconds (default: 30)
- **Automatic Cleanup**: Dead peers automatically removed from network

### Node State Synchronization
- **Versioned State**: Each node maintains incrementing version numbers
- **Smart Merging**: Remote state only applied if newer than local
- **Periodic Sync**: Full state broadcast every 2 * heartbeat_interval seconds
- **Thread-Safe Operations**: All state updates protected by locks

### Message Routing
- **Smart Routing**: Automatic path discovery and multi-hop support
- **Loop Prevention**: Duplicate detection prevents message loops
- **Path Tracking**: Each message maintains routing path history
- **Route Caching**: Recently used routes cached for efficiency

### Hash Registry
- **Distributed Storage**: Hashes stored and synchronized across network
- **Metadata Support**: Each hash can store embeddings and custom metadata
- **Search Capability**: Query by data type, node ID, or hash value
- **Consistency**: Version numbers ensure latest data propagation

### Alerts & Search Broadcasting
- **Priority Levels**: Alerts support low, normal, high, critical levels
- **Mesh-Wide Distribution**: Alerts automatically forwarded through network
- **Search Queries**: Distribute search parameters to all peers for collective processing
- **Flexible Payloads**: Custom JSON payloads for any data type

### Packet Serialization
- **JSON-Based**: Human-readable, language-agnostic format
- **Compression-Ready**: Format supports future compression
- **Type Safety**: Message fields validated during deserialization
- **UTF-8 Encoding**: Standard UTF-8 for text data

## Usage Examples

### Basic Initialization

```python
from mesh import MeshProtocol, NodeState

# Create and start mesh protocol
mesh = MeshProtocol(
    node_id="node_001",
    port=9999,
    heartbeat_interval=5,
    heartbeat_timeout=30,
    max_peers=10
)

# Initialize local state
local_state = {
    'processing_fps': 30.0,
    'battery_level': 85,
    'embedding_dim': 512
}

mesh.start(local_state)
```

### Peer Discovery

```python
# Get all discovered peers
peers = mesh.discover_peers()
for node_id, peer_info in peers.items():
    print(f"Peer: {node_id} at {peer_info.ip_address}:{peer_info.port}")

# Check specific peer
peer = mesh.get_peer_info("node_002")
if peer and peer.is_alive(timeout=30):
    print(f"Peer {node_id} is alive")
```

### State Management

```python
# Update local node state
mesh.update_node_state({
    'processing_fps': 25.5,
    'battery_level': 78,
    'status': 'processing_search'
})

# Get current state
state = mesh.get_node_state()
print(f"Node state version: {state['version']}")
```

### Hash Registry

```python
# Add hash to local registry
mesh.add_hash(
    hash_value="abc123def456",
    data_type="person",
    embedding=[0.1, 0.2, 0.3, ...],  # 512-dim embedding
    metadata={'person_id': 'p123', 'confidence': 0.95}
)

# Search hashes
person_hashes = mesh.hash_registry.search_hashes(data_type='person')

# Propagate registry to peers
mesh.propagate_hash_registry()
```

### Alert Broadcasting

```python
# Send high-priority alert
mesh.broadcast_alert(
    alert_data={
        'type': 'security_alert',
        'description': 'Unauthorized person detected',
        'location': 'entrance_gate',
        'timestamp': time.time()
    },
    priority='critical'
)
```

### Search Broadcasting

```python
# Broadcast search query
mesh.broadcast_search_query(
    query={
        'person_id': 'p123',
        'time_range': (start_time, end_time),
        'location': 'building_a'
    },
    search_type='person_search'
)
```

### Message Handlers

```python
from mesh import MessageType

# Register custom handler
def handle_search_results(message):
    print(f"Search result from {message.source_node_id}: {message.payload}")

mesh.register_message_handler(MessageType.SEARCH_QUERY, handle_search_results)
```

### Network Statistics

```python
# Get network statistics
stats = mesh.get_network_stats()
print(f"Alive peers: {stats['alive_peers']}")
print(f"Messages sent: {stats['messages_sent']}")
print(f"Messages received: {stats['messages_received']}")
print(f"Messages routed: {stats['messages_routed']}")
```

## Threading Model

### Background Threads

1. **Heartbeat Loop**: Sends periodic heartbeats with node state
2. **Liveness Check Loop**: Monitors peer health and removes dead peers
3. **State Sync Loop**: Periodic node state and hash registry sync
4. **Receive Loop**: Listens for incoming UDP messages
5. **Message Handler Threads**: Process received messages asynchronously

All threads are daemon threads and automatically cleanup on shutdown.

## Thread Safety

- **Peers Dictionary**: Protected by `peers_lock` (RLock)
- **Node State**: Protected by `state_lock` (RLock)
- **Hash Registry**: Protected by `registry_lock` (RLock)
- **Statistics**: Protected by `stats_lock` (RLock)
- **Sequence Numbers**: Protected by `seq_lock` (RLock)
- **Message Router**: Protected by `seen_lock` (RLock)

## Configuration

From `config/defaults.yaml`:

```yaml
mesh:
  broadcast_interval: 5          # Seconds between broadcasts
  heartbeat_timeout: 30          # Seconds before considering peer dead
  max_peers: 10                  # Maximum tracked peers
  udp_port: 9999                 # UDP port for mesh communication
```

## Performance Characteristics

- **Memory**: O(max_peers) for peer tracking, O(hash_count) for registry
- **Network Overhead**: Single heartbeat per interval per peer (typically <100 bytes)
- **Latency**: Sub-second message delivery for nearby peers
- **Scalability**: Designed for 10-50 nodes per network segment

## Error Handling

- **Network Errors**: Gracefully handled, logged, and recovery attempted
- **Serialization Errors**: Invalid messages logged and skipped
- **Threading Issues**: All thread operations wrapped in try-except
- **Resource Cleanup**: Automatic cleanup on stop() or exception

## Security Considerations

- **No Authentication**: Currently uses IP-based trust (suitable for local networks)
- **No Encryption**: Messages sent in plain JSON (recommended for local networks only)
- **Future Enhancements**: Can add HMAC signatures, TLS encryption for public networks

## Logging

The mesh protocol uses Python's standard logging module:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
# Mesh logs go to 'mesh.mesh_protocol' logger
```

Log levels:
- `DEBUG`: Detailed mesh operations
- `INFO`: Node join/leave, mesh start/stop
- `WARNING`: Peer removal, failed messages
- `ERROR`: Network errors, handler failures

## Testing

See `tests/` directory for comprehensive unit and integration tests covering:
- Peer discovery and registration
- Message serialization/deserialization
- State synchronization
- Hash registry operations
- Message routing
- Liveness monitoring

## Future Enhancements

1. **Message Compression**: GZIP compression for large payloads
2. **Authentication**: HMAC or certificate-based peer verification
3. **Encryption**: TLS or ChaCha20-Poly1305 for message encryption
4. **Persistent Registry**: SQLite storage for hash registry durability
5. **Load Balancing**: Distribute search queries based on peer capacity
6. **Latency Optimization**: Multicast for discovery on high-latency networks
7. **Bandwidth Management**: QoS and traffic shaping
