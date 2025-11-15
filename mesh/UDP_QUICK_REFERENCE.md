# UDP Network Quick Reference

## 30-Second Setup

```python
from mesh.udp_setup_guide import IntegratedMeshNode

# Create and start node
node = IntegratedMeshNode("camera_01", port=9999, 
                         location_signature="zone_a", 
                         camera_role="registration")
node.start()

# Send detection events
node.process_vision_frame(
    detected_persons=['p1', 'p2'],
    detected_bags=['b1'],
    person_bag_links={'p1': 'b1'}
)

node.stop()
```

## Common Operations

### Initialize a Node

```python
from mesh.udp_setup_guide import IntegratedMeshNode

# For registration/entry camera
node = IntegratedMeshNode(
    node_id="reg_camera_01",
    port=9999,
    location_signature="entry_gate",
    camera_role="registration"  # registration|surveillance|exit|checkpoint|general
)
node.start()
```

### Process Vision Frame

```python
# Called from your vision pipeline
node.process_vision_frame(
    detected_persons=['person_001', 'person_002'],  # List of IDs
    detected_bags=['bag_001', 'bag_002'],           # List of IDs
    person_bag_links={'person_001': 'bag_001'},     # Dict of ID -> ID
    frame_metadata={'frame_id': 1, 'timestamp': time.time()}
)
```

### Report Mismatch

```python
node.report_mismatch(
    person_id='person_001',
    bag_id='bag_999',
    severity='critical',  # low|medium|high|critical
    reason='Unregistered baggage'
)
```

### Report Transfer

```python
node.report_bag_transfer(
    from_person_id='person_001',
    to_person_id='person_002',
    bag_id='bag_001',
    transfer_type='HAND_OFF',  # HAND_OFF|DROP_OFF|PICKUP|EXCHANGE
    confidence=0.95
)
```

### Listen to Local Events

```python
from mesh.event_broadcaster import EventType

def on_person_enter(event):
    print(f"Person {event.person_ids[0]} entered {event.location_signature}")

def on_bag_transfer(event):
    from_id = event.metadata['from_person_id']
    to_id = event.metadata['to_person_id']
    print(f"Bag transfer: {from_id} -> {to_id}")

node.register_event_handler(EventType.PERSON_ENTER, on_person_enter)
node.register_event_handler(EventType.BAG_TRANSFER, on_bag_transfer)
```

### Get Network Status

```python
peers = node.get_peer_list()  # ['camera_01', 'camera_02', ...]
stats = node.get_network_stats()  # Peers, messages, etc.
full_status = node.get_full_status()  # Complete status
```

## Event Types

```python
from mesh.event_broadcaster import EventType

EventType.PERSON_ENTER           # Person entering zone
EventType.PERSON_EXIT            # Person leaving zone  
EventType.BAG_DETECTED           # Bag detected
EventType.PERSON_BAG_LINK        # Person picked up bag
EventType.PERSON_BAG_UNLINK      # Person released bag
EventType.BAG_TRANSFER           # Bag transferred between persons
EventType.MISMATCH_ALERT         # Mismatch detected
EventType.OWNERSHIP_CHANGE       # Bag ownership changed
EventType.ZONE_ACTIVITY          # Zone activity summary
EventType.DEVICE_STATUS          # Device online/offline
```

## Camera Roles

```python
from mesh.event_broadcaster import CameraRole

CameraRole.REGISTRATION  # Entry/registration point
CameraRole.SURVEILLANCE  # General surveillance
CameraRole.EXIT          # Exit point
CameraRole.CHECKPOINT    # Verification/checkpoint
CameraRole.GENERAL       # General/unknown
```

## Integration Points

### From YOLO Detection

```python
# Your YOLO pipeline outputs:
results = yolo_model(frame)

# Extract detections
detected_persons = [p.id for p in results.persons]
detected_bags = [b.id for b in results.bags]

# Send to mesh
node.process_vision_frame(
    detected_persons=detected_persons,
    detected_bags=detected_bags,
    person_bag_links={...}
)
```

### From ReID Linking

```python
# Your ReID pipeline outputs person-bag links:
links = reid_linker.link(persons, bags)

# Extract as dict
person_bag_links = {p['id']: b['id'] for p, b in links}

# Send to mesh
node.process_vision_frame(
    detected_persons=[...],
    detected_bags=[...],
    person_bag_links=person_bag_links
)
```

### From Alert System

```python
# Your alert system detects mismatch:
if detect_mismatch(person, bag):
    node.report_mismatch(
        person_id=person.id,
        bag_id=bag.id,
        severity='critical'
    )
```

## Multi-Node Setup

```python
nodes = {
    'registration': IntegratedMeshNode('cam_reg', 9999, 'gate_a', 'registration'),
    'surveillance': IntegratedMeshNode('cam_surv', 10000, 'hallway_a', 'surveillance'),
    'exit': IntegratedMeshNode('cam_exit', 10001, 'gate_exit', 'exit')
}

# Start all
for node in nodes.values():
    node.start()

# Each node auto-discovers others
time.sleep(3)

# Send detection from each
nodes['registration'].process_vision_frame(['p1'], ['b1'], {'p1': 'b1'})
nodes['surveillance'].process_vision_frame(['p1'], ['b1'], {'p1': 'b1'})
nodes['exit'].process_vision_frame(['p1'], ['b1'], {'p1': 'b1'})

# Stop all
for node in nodes.values():
    node.stop()
```

## Network Configuration

```python
# Default: UDP port 9999 for mesh communication
# If running multiple nodes locally, use different ports:
# - Node 1: port 9999
# - Node 2: port 10000
# - Node 3: port 10001

# Custom heartbeat settings
node.mesh_protocol.heartbeat_interval = 10  # seconds
node.mesh_protocol.heartbeat_timeout = 60   # seconds
node.mesh_protocol.max_peers = 30

# Custom event buffer settings
node.event_broadcaster.event_buffer.flush_interval = 5.0  # seconds
node.event_broadcaster.event_buffer.max_buffer_size = 100  # events
```

## Monitoring

```python
# Check if running
assert node.running == True
assert node.event_broadcaster.running == True
assert node.mesh_protocol.running == True

# Get peers
peers = node.get_peer_list()
print(f"Connected to {len(peers)} peers")

# Get statistics
stats = node.get_broadcaster_stats()
print(f"Events sent: {stats['events_broadcasted']}")
print(f"Events generated: {stats['events_generated']}")

# Get full status
status = node.get_full_status()
print(json.dumps(status, indent=2))
```

## Error Handling

```python
try:
    node.start()
except Exception as e:
    print(f"Error starting node: {e}")

try:
    node.process_vision_frame(
        detected_persons=['p1'],
        detected_bags=['b1']
    )
except Exception as e:
    print(f"Error processing frame: {e}")

try:
    node.stop()
except Exception as e:
    print(f"Error stopping node: {e}")
```

## Common Patterns

### Pattern 1: Continuous Frame Processing

```python
import time
from mesh.udp_setup_guide import IntegratedMeshNode

node = IntegratedMeshNode('camera_01', port=9999, 
                         location_signature='zone_a')
node.start()

frame_id = 0
while True:
    # Your vision pipeline
    persons = detect_persons(frame)  # Returns ['p1', 'p2', ...]
    bags = detect_bags(frame)        # Returns ['b1', 'b2', ...]
    links = link_persons_bags(persons, bags)  # Returns {'p1': 'b1', ...}
    
    # Broadcast to mesh
    node.process_vision_frame(
        detected_persons=persons,
        detected_bags=bags,
        person_bag_links=links,
        frame_metadata={'frame_id': frame_id}
    )
    
    frame_id += 1
    time.sleep(1/30)  # 30 FPS

node.stop()
```

### Pattern 2: Zone-Specific Alerts

```python
from mesh.event_broadcaster import EventType

def on_alert(event):
    if event.location_signature == 'security_zone':
        print(f"SECURITY ALERT: {event.event_type.value}")
        trigger_security_response()

node.register_event_handler(EventType.MISMATCH_ALERT, on_alert)
node.register_event_handler(EventType.BAG_TRANSFER, on_alert)
```

### Pattern 3: Person Tracking Across Zones

```python
# Multiple nodes tracking same person
nodes = {
    'entry': IntegratedMeshNode('cam_entry', port=9999, 
                               location_signature='entry'),
    'hallway': IntegratedMeshNode('cam_hallway', port=10000,
                                 location_signature='hallway'),
    'exit': IntegratedMeshNode('cam_exit', port=10001,
                              location_signature='exit')
}

for node in nodes.values():
    node.start()

# Track person through zones
def track_person(person_id, zone_path):
    for zone_name, detected_in_zone in zone_path:
        if detected_in_zone:
            nodes[zone_name].process_vision_frame(
                detected_persons=[person_id],
                detected_bags=[],
                frame_metadata={'tracking': True}
            )

for node in nodes.values():
    node.stop()
```

## Debugging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Check mesh status
mesh = node.mesh_protocol
print(f"Running: {mesh.running}")
print(f"Peers: {len(mesh.discover_peers())}")
print(f"Stats: {mesh.get_network_stats()}")

# Check broadcaster
broadcaster = node.event_broadcaster
print(f"Running: {broadcaster.running}")
print(f"Buffer size: {broadcaster.event_buffer.get_buffer_size()}")
print(f"Stats: {broadcaster.get_stats()}")

# Check vision emitter
emitter = node.vision_emitter
print(f"State: {emitter.get_current_state()}")
```

## Files

- **Core**: `mesh/mesh_protocol.py` - UDP mesh network
- **Events**: `mesh/event_broadcaster.py` - Event broadcasting
- **Integration**: `mesh/vision_integration.py` - Vision pipeline integration
- **Setup**: `mesh/udp_setup_guide.py` - Complete setup and examples
- **Docs**: `mesh/UDP_SETUP.md` - Full documentation

## Running Examples

```bash
# From Kodikon root directory
python mesh/udp_setup_guide.py
```

This runs three example scenarios:
1. Basic single-node setup
2. Multi-node communication
3. Mismatch detection and alerting
