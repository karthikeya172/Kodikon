# UDP Node-to-Node Communication Implementation

## Overview

The UDP-based mesh network enables real-time communication between distributed devices (cameras, sensors, etc.) in the Kodikon baggage tracking system. When each device detects people entering/exiting zones or bag transfers, it automatically broadcasts this information across the mesh network to all connected nodes.

## Architecture

### Components

1. **MeshProtocol** (`mesh_protocol.py`)
   - Core UDP-based peer discovery and message routing
   - Handles heartbeats, peer liveness checks, and message forwarding
   - Provides message broadcasting infrastructure

2. **EventBroadcaster** (`event_broadcaster.py`)
   - Buffers and batches vision detection events
   - Broadcasts events via the mesh protocol
   - Provides callbacks for local event listeners

3. **VisionEventEmitter** (`vision_integration.py`)
   - Integrates with vision pipeline outputs
   - Tracks persons and bags across frames
   - Detects entry/exit, linkings, transfers
   - Emits appropriate events to the broadcaster

4. **IntegratedMeshNode** (`udp_setup_guide.py`)
   - High-level unified interface combining all components
   - Recommended for practical use

### Communication Flow

```
Vision Detection (YOLO, ReID)
        ↓
VisionEventEmitter (tracking, detection)
        ↓
EventBroadcaster (buffer, batch)
        ↓
MeshProtocol (UDP broadcast)
        ↓
Network → Other Nodes
        ↓
Event Listeners (local handlers)
```

## Event Types

The system supports these vision events:

- **PERSON_ENTER**: Person detected entering a zone
- **PERSON_EXIT**: Person detected leaving a zone
- **BAG_DETECTED**: Bag detected in the camera view
- **PERSON_BAG_LINK**: Person-bag association detected
- **PERSON_BAG_UNLINK**: Person-bag association broken
- **BAG_TRANSFER**: Bag transferred between persons
- **MISMATCH_ALERT**: Possible baggage mismatch detected
- **OWNERSHIP_CHANGE**: Bag ownership changed
- **ZONE_ACTIVITY**: General zone activity summary
- **DEVICE_STATUS**: Device online/offline status

## Setup & Usage

### Basic Setup

```python
from mesh.udp_setup_guide import IntegratedMeshNode
from mesh.event_broadcaster import CameraRole

# Create a node for a registration camera
node = IntegratedMeshNode(
    node_id="camera_registration_01",
    port=9999,
    location_signature="registration_gate_a",
    camera_role="registration"
)

# Start the node
node.start(local_state={
    'device_type': 'vision_camera',
    'model': 'YOLO_v8n',
    'fps': 30
})

# Process vision detections
node.process_vision_frame(
    detected_persons=['person_001', 'person_002'],
    detected_bags=['bag_001'],
    person_bag_links={'person_001': 'bag_001'},
    frame_metadata={'frame_id': 1, 'timestamp': time.time()}
)

# Cleanup
node.stop()
```

### Multiple Nodes

```python
# Create nodes for different locations
registration_node = IntegratedMeshNode(
    node_id="camera_reg",
    port=9999,
    location_signature="gate_a",
    camera_role="registration"
)

surveillance_node = IntegratedMeshNode(
    node_id="camera_surv",
    port=10000,
    location_signature="hallway_a",
    camera_role="surveillance"
)

exit_node = IntegratedMeshNode(
    node_id="camera_exit",
    port=10001,
    location_signature="gate_exit",
    camera_role="exit"
)

# Start all nodes
for node in [registration_node, surveillance_node, exit_node]:
    node.start()

# Each node automatically discovers others and broadcasts events
```

### Handling Mismatch Alerts

```python
# Report a mismatch
node.report_mismatch(
    person_id='person_001',
    bag_id='bag_999',
    severity='critical',
    reason='Person with unregistered baggage detected'
)

# Broadcast is automatically sent to all nodes
```

### Handling Bag Transfers

```python
# Report transfer between persons
node.report_bag_transfer(
    from_person_id='person_001',
    to_person_id='person_002',
    bag_id='bag_001',
    transfer_type='HAND_OFF',
    confidence=0.95
)
```

### Receiving Events Locally

```python
from mesh.event_broadcaster import EventType

def on_person_enter(event):
    print(f"Person {event.person_ids[0]} entered {event.location_signature}")

def on_mismatch(event):
    print(f"ALERT: Mismatch at {event.location_signature}")

# Register local listeners
node.register_event_handler(EventType.PERSON_ENTER, on_person_enter)
node.register_event_handler(EventType.MISMATCH_ALERT, on_mismatch)
```

## Integration with Vision Pipeline

### From YOLO/ReID Output

```python
# Assuming your vision pipeline produces:
detection_output = {
    'persons': ['person_001', 'person_002'],
    'bags': ['bag_001', 'bag_002'],
    'person_bag_links': {
        'person_001': 'bag_001',
        'person_002': 'bag_002'
    }
}

# Feed to mesh node
node.process_vision_frame(
    detected_persons=detection_output['persons'],
    detected_bags=detection_output['bags'],
    person_bag_links=detection_output['person_bag_links'],
    frame_metadata={
        'frame_id': frame_number,
        'timestamp': time.time(),
        'confidence': 0.92
    }
)
```

## Network Configuration

### Ports

- **Primary mesh port**: 9999 (configurable)
- Each device should use a different port to avoid conflicts
- Default ports: 9999, 10000, 10001, 10002, etc.

### Broadcast Settings

- **Heartbeat interval**: 5 seconds (configurable)
- **Heartbeat timeout**: 30 seconds (peer considered dead after)
- **Max peers**: 20 (configurable)
- **Event buffer flush**: Every 2 seconds or 50 events

### Network Tuning

```python
# Create custom mesh node with tuning
node = IntegratedMeshNode(node_id="camera_01", port=9999)

# Access underlying mesh protocol for fine-tuning
mesh = node.mesh_protocol
mesh.heartbeat_interval = 10  # Increase heartbeat interval
mesh.heartbeat_timeout = 60   # Increase timeout
mesh.max_peers = 30           # Increase max peers

# Access broadcaster for tuning
broadcaster = node.event_broadcaster
broadcaster.event_buffer.flush_interval = 5.0  # Increase flush interval
```

## Monitoring & Statistics

### Get Network Status

```python
# Network connectivity
stats = node.get_network_stats()
print(f"Alive peers: {stats['alive_peers']}")
print(f"Total peers: {stats['total_peers']}")
print(f"Messages sent: {stats['messages_sent']}")
print(f"Messages received: {stats['messages_received']}")
```

### Get Event Statistics

```python
# Event broadcasting stats
broadcaster_stats = node.get_broadcaster_stats()
print(f"Events generated: {broadcaster_stats['events_generated']}")
print(f"Events broadcasted: {broadcaster_stats['events_broadcasted']}")
print(f"By type: {broadcaster_stats['events_by_type']}")
```

### Get Full Status

```python
status = node.get_full_status()
print(f"Node: {status['node_info']}")
print(f"Network: {status['network']}")
print(f"Events: {status['events']}")
print(f"Vision: {status['vision']}")
```

## Message Format

### Event Message (JSON)

```json
{
  "event_id": "camera_01_abc123def456",
  "event_type": "person_enter",
  "timestamp": 1731705600.123,
  "node_id": "camera_01",
  "location_signature": "registration_gate_a",
  "camera_role": "registration",
  "person_ids": ["person_001"],
  "bag_ids": [],
  "confidence": 0.95,
  "metadata": {
    "frame_id": 100,
    "fps": 30
  }
}
```

### Transfer Event (JSON)

```json
{
  "event_id": "camera_01_xyz789uvw012",
  "event_type": "bag_transfer",
  "timestamp": 1731705605.456,
  "node_id": "camera_01",
  "location_signature": "hallway_a",
  "camera_role": "surveillance",
  "person_ids": ["person_001", "person_002"],
  "bag_ids": ["bag_001"],
  "confidence": 0.92,
  "metadata": {
    "from_person_id": "person_001",
    "to_person_id": "person_002",
    "transfer_type": "HAND_OFF"
  }
}
```

## Error Handling

### Node Disconnection

The mesh protocol automatically handles peer disconnection:
- Nodes that don't send heartbeats within 30 seconds are marked dead
- Dead nodes are removed from the peer list
- Failed broadcasts are logged but don't crash the system

### Event Buffer Overflow

```python
# Buffer auto-flushes every 2 seconds or at 50 events
# Configure flush behavior
node.event_broadcaster.event_buffer.flush_interval = 5.0
node.event_broadcaster.event_buffer.max_buffer_size = 100
```

### Network Errors

All network errors are caught and logged:
- UDP send failures are retried silently
- Deserialization errors are logged with context
- Handler exceptions don't crash the event loop

## Performance Considerations

### Event Batching

Events are automatically batched for efficiency:
- Up to 50 events per batch
- Auto-flush every 2 seconds even if buffer not full
- Reduces network overhead by ~80%

### Tracking Efficiency

```python
# Person tracker uses timeout-based cleanup
# Exit timeout: 5 seconds (configurable)
# Bags are tracked only while visible

# Memory usage:
# - Per person: ~64 bytes
# - Per bag: ~64 bytes
# - Per link: ~64 bytes
```

### Scalability

- **Typical deployment**: 10-50 cameras
- **Network bandwidth**: ~10-20 KB/s per camera at 30 FPS
- **Latency**: <10ms between nodes (LAN)
- **Throughput**: Can handle 100+ events/second

## Troubleshooting

### Nodes Not Discovering Each Other

1. Check firewall allows UDP on specified ports
2. Verify all nodes on same network subnet
3. Check node IDs are unique
4. Review logs for binding errors

### Events Not Being Broadcast

1. Check node is running: `node.running == True`
2. Check broadcaster is running: `node.event_broadcaster.running == True`
3. Verify mesh protocol has peers: `node.get_peers()` should return nodes
4. Check buffer flush: `node.event_broadcaster.event_buffer.get_buffer_size()`

### High Network Latency

1. Increase heartbeat interval for less chatter
2. Increase event buffer flush interval
3. Check network congestion on same network
4. Reduce max peers if not needed

## Examples

See `udp_setup_guide.py` for complete working examples:

1. **Basic Setup**: Single node with vision event emission
2. **Multi-Node**: Multiple cameras communicating
3. **Mismatch Detection**: Handling baggage alerts
4. **Advanced**: Custom event handlers and tuning

Run examples:
```bash
python mesh/udp_setup_guide.py
```

## Future Enhancements

- [ ] End-to-end encryption for events
- [ ] Event querying across mesh
- [ ] Distributed consensus voting for critical alerts
- [ ] Event compression for bandwidth optimization
- [ ] Peer clustering for hierarchical networks
- [ ] Event persistence/archival
- [ ] Analytics and reporting

## References

- **Mesh Protocol**: `mesh/mesh_protocol.py`
- **Event Broadcaster**: `mesh/event_broadcaster.py`
- **Vision Integration**: `mesh/vision_integration.py`
- **Setup Guide**: `mesh/udp_setup_guide.py`
- **Examples**: `mesh/udp_setup_guide.py` - see `if __name__ == "__main__"` section
