# Mesh Protocol - Peer-to-Peer UDP Network

A comprehensive peer-to-peer UDP-based mesh network implementation for distributed node communication in the Kodikon system. Designed for scalable, resilient communication between camera nodes, servers, and processing units.

## Features

### Core Networking
- **UDP-Based Communication**: Low-latency, connectionless messaging
- **Peer Discovery**: Automatic node discovery via UDP broadcast
- **Heartbeat Monitoring**: Continuous node liveness monitoring
- **Message Routing**: Multi-hop message delivery through mesh
- **Anti-Loop Detection**: Prevents message duplication and loops

### State Management
- **Node State Sync**: Distributed state synchronization with versioning
- **State Consistency**: Smart merging of remote state with conflict resolution
- **Atomic Updates**: Thread-safe state operations with locks

### Hash Registry
- **Distributed Storage**: Network-wide hash and embedding storage
- **Rich Metadata**: Support for embeddings, confidence scores, and custom data
- **Efficient Search**: Query by data type, node, or specific hash
- **Automatic Propagation**: Registry updates shared across network

### Alerts & Queries
- **Priority Alerts**: Broadcast critical alerts with priority levels
- **Search Distribution**: Distribute search queries to all nodes
- **Flexible Payloads**: JSON-based extensible message format

## Quick Start

### Installation

```bash
# No additional dependencies - uses only Python stdlib
# Existing project already has required packages
pip install -r requirements.txt
```

### Basic Usage

```python
from mesh import MeshProtocol

# Create mesh instance
mesh = MeshProtocol(
    node_id="camera_001",
    port=9999,
    heartbeat_interval=5,
    heartbeat_timeout=30
)

# Start the mesh
mesh.start(local_state={
    'fps': 30,
    'battery': 85,
    'location': 'entrance'
})

# Discover peers
peers = mesh.discover_peers()
print(f"Found {len(peers)} peers")

# Broadcast alert
mesh.broadcast_alert({
    'type': 'intrusion_alert',
    'confidence': 0.98
}, priority='critical')

# Add hash to registry
mesh.add_hash(
    hash_value="face_hash_123",
    data_type="person",
    embedding=embedding_vector,
    metadata={'person_id': 'p001'}
)

# Get network stats
stats = mesh.get_network_stats()
print(f"Active peers: {stats['alive_peers']}")

# Graceful shutdown
mesh.stop()
```

## Architecture

### Message Types

```python
MessageType.HEARTBEAT          # Node status beacon (1)
MessageType.PEER_DISCOVERY     # New node announcement (2)
MessageType.NODE_STATE_SYNC    # State synchronization (3)
MessageType.SEARCH_QUERY       # Distributed search (4)
MessageType.ALERT              # Priority alert (5)
MessageType.HASH_REGISTRY      # Hash registry update (6)
MessageType.ROUTE_BROADCAST    # Multi-hop broadcast (7)
MessageType.ACK                # Acknowledgment (8)
```

### Components

#### MeshProtocol (Main Interface)
The central component managing all mesh operations. Handles:
- Peer discovery and registration
- Heartbeat sending and receiving
- State synchronization
- Message routing
- Alert and search broadcasting
- Hash registry management

#### PeerInfo
Information about discovered peers:
- Node ID, IP address, port
- Operational state (ACTIVE, IDLE, PROCESSING, OFFLINE)
- Last heartbeat timestamp
- Node capabilities (embedding_dim, models, fps, battery)

#### MeshMessage
Serializable message structure:
- Message type and source
- Timestamp and sequence number
- JSON payload
- Routing path history

#### HashRegistry
Distributed hash storage with:
- Hash-to-entry mapping
- Search by type/node/hash
- Version tracking for consistency
- Metadata support (embeddings, confidence, etc.)

#### NodeStateManager
Versioned state management:
- Local state snapshot generation
- Remote state merging with version checking
- Thread-safe updates

#### MessageRouter
Intelligent message routing:
- Path discovery to destinations
- Duplicate detection
- Route caching
- Configurable hop limits

## Configuration

From `config/defaults.yaml`:

```yaml
mesh:
  broadcast_interval: 5      # Seconds between discovery broadcasts
  heartbeat_timeout: 30      # Seconds to consider peer dead
  max_peers: 10             # Maximum peers to track
  udp_port: 9999            # UDP port for mesh
```

## API Reference

### MeshProtocol Class

#### Initialization
```python
MeshProtocol(
    node_id: str,                  # Unique node identifier
    port: int = 9999,             # UDP port
    heartbeat_interval: int = 5,  # Heartbeat interval (seconds)
    heartbeat_timeout: int = 30,  # Peer timeout (seconds)
    max_peers: int = 10           # Maximum tracked peers
)
```

#### Core Methods

**Network Management:**
- `start(local_state=None)` - Initialize and start mesh
- `stop()` - Gracefully shutdown
- `discover_peers()` -> Dict[str, PeerInfo] - Get discovered peers
- `get_peer_info(node_id) -> Optional[PeerInfo]` - Get specific peer
- `get_network_stats() -> dict` - Get network statistics

**State Management:**
- `update_node_state(updates: dict)` - Update local state
- `get_node_state() -> dict` - Get current state snapshot

**Hash Registry:**
- `add_hash(hash_value, data_type, embedding, metadata)` - Add hash entry
- `propagate_hash_registry()` - Sync registry to peers

**Broadcasting:**
- `send_message(message, target_ip, target_port)` - Send message
- `broadcast_alert(alert_data, priority)` - Send alert
- `broadcast_search_query(query, search_type)` - Send search

**Handlers:**
- `register_message_handler(message_type, handler)` - Register handler
- `get_network_stats()` - Get statistics

### PeerInfo Class

```python
@dataclass
class PeerInfo:
    node_id: str
    ip_address: str
    port: int
    state: NodeState = NodeState.ACTIVE
    last_heartbeat: float
    embedding_dim: int = 512
    reid_model: str = "osnet_x1_0"
    yolo_model: str = "yolov8n"
    processing_fps: float = 30.0
    battery_level: Optional[float] = None
    
    def is_alive(self, timeout: float) -> bool:
        """Check if peer is responsive"""
    
    def to_dict(self) -> dict:
        """Serialize for transmission"""
```

## Examples

### Example 1: Basic Setup
```python
from mesh import MeshProtocol

mesh = MeshProtocol("node_001", port=9999)
mesh.start({'status': 'active', 'fps': 30})
time.sleep(5)
mesh.stop()
```

### Example 2: Peer Discovery
```python
mesh = MeshProtocol("node_discovery", port=9998)
mesh.start()
time.sleep(5)

peers = mesh.discover_peers()
for node_id, peer_info in peers.items():
    print(f"{node_id} at {peer_info.ip_address}:{peer_info.port}")
    print(f"  State: {peer_info.state.name}")
    print(f"  Battery: {peer_info.battery_level}%")
```

### Example 3: Hash Registry
```python
mesh = MeshProtocol("node_hash", port=9996)
mesh.start()

# Add embeddings
mesh.add_hash(
    hash_value="person_123_hash",
    data_type="person",
    embedding=embedding_vector,
    metadata={'person_id': 'p123', 'confidence': 0.95}
)

# Search
person_hashes = mesh.hash_registry.search_hashes(data_type='person')
print(f"Found {len(person_hashes)} person hashes")
```

### Example 4: Alerts
```python
mesh = MeshProtocol("node_alerts", port=9995)
mesh.start()

# High-priority alert
mesh.broadcast_alert({
    'type': 'security_alert',
    'description': 'Unauthorized person detected',
    'location': 'gate_1',
    'confidence': 0.98
}, priority='critical')

# Medium-priority notification
mesh.broadcast_alert({
    'type': 'info',
    'message': 'Scanning complete'
}, priority='normal')
```

### Example 5: Search Broadcasting
```python
mesh = MeshProtocol("node_search", port=9994)
mesh.start()

# Search query
mesh.broadcast_search_query({
    'person_id': 'p123',
    'start_time': time.time() - 3600,
    'end_time': time.time(),
    'location': 'building_a'
}, search_type='person_search')
```

### Example 6: Custom Message Handlers
```python
from mesh import MessageType, MeshMessage

def handle_alert(message):
    print(f"Alert from {message.source_node_id}:")
    print(f"  {message.payload}")

def handle_search(message):
    print(f"Search query: {message.payload['query']}")

mesh = MeshProtocol("node_handlers", port=9993)
mesh.register_message_handler(MessageType.ALERT, handle_alert)
mesh.register_message_handler(MessageType.SEARCH_QUERY, handle_search)
mesh.start()
```

## Threading Model

The mesh protocol runs several daemon threads:

1. **Heartbeat Loop** - Sends periodic heartbeats (interval: `heartbeat_interval`)
2. **Liveness Check Loop** - Monitors peer health (interval: `heartbeat_timeout / 2`)
3. **State Sync Loop** - Syncs state and registry (interval: `heartbeat_interval * 2`)
4. **Receive Loop** - Listens for UDP messages
5. **Message Handlers** - Process received messages asynchronously

All threads:
- Clean up automatically on shutdown
- Handle exceptions gracefully
- Are thread-safe with proper locking

## Thread Safety

All shared data structures are protected:
- **Peers Dictionary**: `peers_lock` (threading.Lock)
- **Node State**: `state_lock` (threading.Lock)
- **Hash Registry**: `registry_lock` (threading.Lock)
- **Statistics**: `stats_lock` (threading.Lock)
- **Sequence Numbers**: `seq_lock` (threading.Lock)

Safe to call API methods from multiple threads.

## Performance

### Network Overhead
- **Heartbeat**: ~100 bytes every 5 seconds per peer
- **Discovery**: ~80 bytes every 5 seconds
- **State Sync**: ~200 bytes every 10 seconds
- **Message**: Varies with payload size

### Memory Usage
- Per peer: ~500 bytes
- Per hash entry: ~2KB (with embedding)
- Message cache: Bounded to 10,000 recent messages

### Latency
- Local network: <10ms typical
- Multi-hop: <50ms per hop
- Message routing: Sub-millisecond overhead

### Scalability
- Tested with 10-50 nodes per segment
- Scales to thousands with network segmentation
- Designed for edge networks, not internet-scale

## Testing

Run unit tests:

```bash
# Run all tests
python -m pytest tests/test_mesh_protocol.py -v

# Run specific test class
python -m pytest tests/test_mesh_protocol.py::TestMeshProtocol -v

# Run with coverage
python -m pytest tests/test_mesh_protocol.py --cov=mesh --cov-report=html
```

Test coverage includes:
- Message serialization/deserialization
- Peer discovery and management
- State synchronization
- Hash registry operations
- Message routing
- Liveness monitoring
- Thread safety
- Error handling

## Examples

Run examples:

```bash
# Run all examples
python mesh/examples.py

# Run specific example (1-8)
python mesh/examples.py 1    # Basic setup
python mesh/examples.py 3    # State management
python mesh/examples.py 4    # Hash registry
```

## Logging

Enable logging to see detailed mesh operations:

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Mesh logs to 'mesh.mesh_protocol' logger
mesh_logger = logging.getLogger('mesh.mesh_protocol')
mesh_logger.setLevel(logging.DEBUG)
```

Log levels:
- `DEBUG`: Detailed mesh operations, message flows
- `INFO`: Node discovery, mesh start/stop
- `WARNING`: Peer removal, timeout warnings
- `ERROR`: Network errors, handler failures

## Security Considerations

### Current Implementation
- **No Authentication**: Uses IP-based trust (suitable for local networks)
- **No Encryption**: Plain JSON messages (recommended for local networks)
- **No Message Signing**: No HMAC or signature verification

### Recommendations
- Deploy on trusted local networks only
- Use network segmentation/VLANs for isolation
- Use IP whitelist for peer validation in untrusted environments

### Future Enhancements
- HMAC message signing for integrity
- TLS encryption for sensitive deployments
- Certificate-based peer authentication
- Message compression for bandwidth optimization

## Troubleshooting

### Peers not discovering each other
- Check UDP port is not blocked by firewall
- Verify peers are on same network segment or broadcast domain
- Check `broadcast_interval` setting

### High message latency
- Check network congestion
- Reduce `max_peers` or split into smaller mesh segments
- Check peer machine CPU/memory

### Messages not being routed
- Verify peers are alive (check `get_network_stats()`)
- Check message hop limit (default: 5)
- Verify routing paths with `routing_path` in message

### Memory usage growing
- Check message handler doesn't leak resources
- Monitor hash registry size
- Tune `max_peers` if needed

## Integration with Kodikon

### With Camera Streaming
```python
# In streaming module
mesh.broadcast_alert({
    'type': 'detection',
    'object': detection_type,
    'confidence': confidence
})
```

### With Vision Module
```python
# In vision module
mesh.add_hash(
    hash_value=hash_of_embedding,
    data_type='person',
    embedding=embedding_vector,
    metadata={'source_camera': camera_id}
)
```

### With Integrated Runtime
```python
# In integrated runtime
mesh.broadcast_search_query({
    'target_embedding': query_embedding,
    'time_range': (start, end)
}, search_type='embedding_search')
```

## Files

- `mesh_protocol.py` - Core implementation (717 lines)
- `__init__.py` - Module exports
- `examples.py` - Usage examples and demonstrations
- `MESH_PROTOCOL_DOCUMENTATION.md` - Detailed documentation
- `../tests/test_mesh_protocol.py` - Comprehensive test suite

## License

Part of Kodikon project. See main LICENSE file.

## Support

For issues or questions:
1. Check documentation in `MESH_PROTOCOL_DOCUMENTATION.md`
2. Review examples in `examples.py`
3. Check test cases in `test_mesh_protocol.py`
4. Enable debug logging for detailed diagnostics
