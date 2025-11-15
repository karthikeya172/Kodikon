# Integrated System Orchestrator - Full Documentation

## Overview

The `IntegratedSystem` class is the central orchestrator that coordinates all subsystems of the Kodikon baggage tracking platform:
- **YOLO Detection**: Real-time person and baggage detection
- **Embedding Extraction**: Deep learning-based feature extraction for ReID
- **Power Management**: Adaptive performance based on activity density
- **Mesh Networking**: Peer-to-peer communication and data synchronization
- **UI Visualization**: Real-time overlays with system metrics
- **Search Interface**: Query-based baggage search across the network

## Architecture

```
IntegratedSystem (Main Orchestrator)
├── CameraWorker Threads (per camera)
│   └── Frame Capture Queue
├── Vision Pipeline
│   ├── YOLODetectionEngine
│   ├── EmbeddingExtractor
│   └── ColorDescriptor
├── Power Management
│   ├── MotionAnalyzer
│   └── ObjectDensityAnalyzer
├── Mesh Network
│   ├── Peer Discovery
│   ├── Message Router
│   └── Hash Registry
└── Processing Loops
    ├── Processing Loop (detection + linking)
    ├── Visualization Loop (UI rendering)
    ├── Mesh Sync Loop (state synchronization)
    └── Search Handler Loop (query processing)
```

## Core Components

### 1. System State Management

**Enums:**
- `SystemState`: STOPPED, INITIALIZING, RUNNING, PROCESSING, ERROR, SHUTTING_DOWN
- `CameraState`: IDLE, CAPTURING, PROCESSING, ERROR

**Metrics:**
- `SystemMetrics`: Tracks total frames, detections, links, mismatches, performance

### 2. Camera Capture (Multi-threaded)

**CameraWorker Class:**
```python
camera = CameraWorker("camera-0", source=0, fps_config=(30, 1))
camera.start()  # Runs in background thread

# Get frames from queue
frame_id, frame, timestamp = camera.get_frame(timeout=0.1)
```

**Features:**
- Non-blocking frame queuing with drop-on-full logic
- Frame skipping for power management
- Real-time FPS tracking
- Graceful error handling

### 3. Processing Pipeline

**Frame Processing Steps:**
1. Motion analysis (optical flow)
2. YOLO detection (conditional based on power mode)
3. Embedding extraction (person and bag ReID)
4. Color histogram extraction
5. Person-bag linking
6. Mismatch detection
7. Alert generation
8. Adaptive power mode update

**Person-Bag Linking Algorithm:**
- Spatial distance metric (pixel distance)
- Feature similarity (embedding dot product)
- Color histogram similarity (Bhattacharyya distance)
- Weighted score: 40% feature + 30% spatial + 30% color

**Mismatch Detection:**
- Identifies baggage items without linked persons
- Triggers alerts for security monitoring

### 4. Power Management Integration

**Adaptive Power Modes:**
- **ECO**: 10 FPS, 640x480, YOLO every 30 frames
- **BALANCED**: 20 FPS, 1280x720, YOLO every 10 frames
- **PERFORMANCE**: 30 FPS, 1920x1080, YOLO every 3 frames

**Triggers:**
- Activity density (motion + object count)
- Battery level
- User override possible

### 5. Mesh Network Integration

**Message Types Handled:**
- `SEARCH_QUERY`: Incoming baggage searches
- `ALERT`: Security alerts from other nodes
- `NODE_STATE_SYNC`: State synchronization

**Broadcast:**
- Local detections → peer nodes
- Alerts → all connected peers
- State updates → periodic sync

### 6. Baggage Profile Management

**BaggageProfile:**
- Unique hash ID (SHA256 from embedding)
- Visual descriptors (color histogram, embedding)
- Detection history across cameras
- Person-bag associations
- Mismatch counter

### 7. Search Interface

**Search Criteria:**
- Description text matching
- Color histogram similarity
- Embedding similarity
- Multi-criteria weighted scoring

**Search Result Scoring:**
```python
score = (
    0.5 * description_match +
    0.3 * color_similarity +
    0.2 * embedding_similarity
)
```

### 8. UI Overlays

**Real-time Overlays:**
- FPS counter
- Current power mode
- Connected peer count
- Detection and link counts
- Latest alert message

**Keyboard Controls:**
- `q`: Quit system
- `s`: Trigger interactive search

## Usage

### Basic Initialization

```python
from integrated_runtime.integrated_system import IntegratedSystem
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Create system
system = IntegratedSystem(
    config_path="config/defaults.yaml",
    node_id="baggage-tracker-01"
)

# Start and run (blocking)
system.run()
```

### Programmatic Control

```python
# Initialize subsystems
system.initialize()

# Start background loops
system.start()

# Run custom processing
while system.running:
    # Your custom logic here
    time.sleep(0.1)

# Graceful shutdown
system.shutdown()
```

### Configuration

**YAML Structure:**
```yaml
camera:
  fps: 30
  width: 1280
  height: 720

yolo:
  model: "yolov8n"
  confidence_threshold: 0.5

reid:
  model: "osnet_x1_0"
  embedding_dim: 512

power:
  mode: "balanced"
  min_fps: 10
  max_fps: 30

mesh:
  udp_port: 9999
  heartbeat_interval: 5
  heartbeat_timeout: 30
```

### Search API

```python
# By description
results = system.search_by_description("red backpack")

# Results format
# [
#   {
#     'bag_id': 'b_123_456',
#     'hash_id': 'a1b2c3d4...',
#     'class_name': 'backpack',
#     'description': 'red backpack',
#     'person_id': 'person_001',
#     'first_seen': '2025-11-15T10:30:00',
#     'last_seen': '2025-11-15T10:35:00',
#     'camera_ids': ['camera-0'],
#     'mismatch_count': 0
#   },
#   ...
# ]
```

### Alert Handling

```python
# Create alert (broadcasted to network)
system._create_alert(
    "Suspicious baggage detected at checkpoint",
    priority="high",
    data=mismatch_link
)
```

## Threading Model

**Background Threads:**
1. **CameraWorker** (per camera): Captures frames continuously
2. **Processing Loop**: Processes frames through pipeline
3. **Visualization Loop**: Renders UI with overlays
4. **Mesh Sync Loop**: Periodic state synchronization
5. **Search Handler Loop**: Processes search queries
6. **Mesh Protocol Threads** (5+): Peer discovery, heartbeats, message routing

**Thread Safety:**
- `profiles_lock`: Protects baggage profile dictionary
- `alerts_lock`: Protects alerts deque
- `mesh.peers_lock`: Protects peer information
- `queue.Queue`: Thread-safe frame buffering

## Performance Considerations

### Frame Processing
- **Motion Analysis**: ~5-10ms per frame
- **YOLO Detection**: ~50-200ms (depends on model and resolution)
- **Embedding Extraction**: ~30-100ms per detection
- **Color Histogram**: ~5-10ms per detection

### Bottlenecks
- YOLO inference (mitigated by adaptive intervals)
- Embedding extraction (parallelizable per detection)
- Mesh broadcasting (async)

### Optimization Tips
1. Use smaller YOLO models (nano/small) for real-time performance
2. Adjust power mode based on activity patterns
3. Enable frame skipping in low-activity scenarios
4. Batch embedding extraction for multiple detections

## Error Handling

**Graceful Degradation:**
- Missing ReID model → use color histogram fallback
- Camera error → automatic reconnection
- Mesh network loss → continue local processing
- CUDA unavailable → fallback to CPU

**Signal Handling:**
- SIGINT (Ctrl+C) → graceful shutdown
- SIGTERM → graceful shutdown
- Automatic cleanup of all resources

## Example: Extended Usage

```python
import time
from integrated_runtime.integrated_system import IntegratedSystem

# Create system
system = IntegratedSystem(node_id="entrance-cam-1")
system.initialize()
system.start()

try:
    # Monitor for 1 hour
    start_time = time.time()
    while time.time() - start_time < 3600:
        # Print metrics every 30 seconds
        if int(time.time()) % 30 == 0:
            print(f"Detections: {system.metrics.total_detections}")
            print(f"Links: {system.metrics.total_links_found}")
            print(f"Mismatches: {system.metrics.total_mismatches}")
            print(f"Mode: {system.metrics.current_power_mode.name}")
        
        time.sleep(1)

finally:
    system.shutdown()
```

## Integration Points

### With Vision Module
- YOLO detection engine
- Embedding extractor
- Color descriptor
- Baggage profile management

### With Power Module
- Motion analyzer
- Activity density calculation
- Adaptive mode switching
- FPS/resolution scaling

### With Mesh Module
- Peer discovery
- Message routing
- Hash registry
- State synchronization

### With Backend
- Provide frame stream via WebSocket
- REST API for search queries
- Real-time alert webhooks
- Metrics export

## Future Enhancements

1. **Multi-Camera Support**: Extend to handle 4+ cameras with load balancing
2. **GPU Batch Processing**: Batch embeddings for multiple detections
3. **Distributed Tracking**: Cross-camera trajectory prediction
4. **Analytics Dashboard**: Real-time metrics dashboard
5. **ML Model Updates**: Over-the-air model deployment
6. **Edge Inference**: Quantized models for faster inference
7. **Redundancy**: Failover to backup nodes
8. **Custom Rules Engine**: User-defined linking and alert rules

## Troubleshooting

**System fails to initialize:**
- Check CUDA/GPU availability
- Verify config file path
- Check model files exist

**Low FPS:**
- Reduce resolution in config
- Switch to smaller YOLO model
- Disable real-time visualization
- Enable power ECO mode

**High memory usage:**
- Reduce frame queue size
- Limit baggage profile history
- Disable visualization

**Mesh not connecting:**
- Check firewall UDP port
- Verify peer IP addresses
- Check network connectivity

## API Reference

### IntegratedSystem Methods

**Lifecycle:**
- `initialize()`: Setup all subsystems
- `start()`: Begin processing loops
- `run()`: Blocking call (initialize + start + loop)
- `shutdown()`: Clean shutdown

**Search:**
- `search_by_description(str) → List[Dict]`: Search baggage by description
- `_search_baggage(Dict) → List[BaggageProfile]`: Internal search with multi-criteria

**Alerts:**
- `_create_alert(str, priority, data)`: Create and broadcast alert

**Metrics:**
- `metrics.total_frames_processed`
- `metrics.total_detections`
- `metrics.total_links_found`
- `metrics.total_mismatches`
- `metrics.avg_processing_time_ms`
- `metrics.active_peers`
- `metrics.current_power_mode`

## Performance Metrics

**Typical Performance (NVIDIA GPU):**
- Frame rate: 20-30 FPS (BALANCED mode)
- Detection latency: 50-100ms per frame
- Link finding latency: 20-50ms
- End-to-end frame processing: 100-200ms

**CPU-only Performance:**
- Frame rate: 5-15 FPS (depending on resolution)
- Detection latency: 200-500ms
- Reduced YOLO interval recommended

## Security Considerations

1. **Mesh Authentication**: Currently trust-based (enhance with crypto)
2. **Data Privacy**: Embeddings are lossy (recommend anonymization)
3. **Access Control**: Add node authentication for peer discovery
4. **Encryption**: Add TLS for mesh communication
5. **Rate Limiting**: Protect against DoS on search queries
