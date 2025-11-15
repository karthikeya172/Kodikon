# Kodikon Integrated Runtime

## Overview

The `integrated_runtime` module is the core orchestrator of the Kodikon baggage tracking system. It coordinates:

- **Real-time Camera Capture**: Multi-threaded video capture with frame queuing
- **YOLO Detection**: Person and baggage detection with confidence filtering
- **Person-Bag Linking**: Multi-metric similarity matching
- **Mesh Networking**: Peer-to-peer communication and data synchronization
- **Power Management**: Adaptive frame rate and resolution scaling
- **UI Visualization**: Real-time overlays with system metrics
- **Search Interface**: Query-based baggage profile search
- **System Lifecycle**: Graceful initialization, operation, and shutdown

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from integrated_runtime.integrated_system import IntegratedSystem

system = IntegratedSystem()
system.run()  # Blocking call, press Ctrl+C to stop
```

### Non-blocking Control

```python
system = IntegratedSystem()
system.initialize()
system.start()

# Do other work here
import time
time.sleep(300)

system.shutdown()
```

## Architecture

### Main Components

1. **IntegratedSystem** - Main orchestrator class
2. **CameraWorker** - Per-camera capture thread
3. **Processing Loop** - Detection, linking, and metrics
4. **Visualization Loop** - UI rendering
5. **Mesh Sync Loop** - Network synchronization
6. **Search Handler Loop** - Query processing

### Threading Model

```
Main Thread
├── CameraWorker Threads (1 per camera)
├── Processing Loop Thread
├── Visualization Loop Thread
├── Mesh Sync Loop Thread
├── Search Handler Loop Thread
└── Mesh Protocol Threads (5+)
    ├── Discovery Thread
    ├── Heartbeat Thread
    ├── Receive Thread
    ├── Liveness Check Thread
    └── State Sync Thread
```

## Features

### YOLO Detection

- Configurable model (yolov8n, yolov8s, yolov8m)
- CUDA/CPU support
- Class filtering (person, bag, backpack, suitcase, handbag)
- Confidence thresholding
- Conditional processing based on power mode

### Person-Bag Linking

Multi-metric similarity algorithm:
- **40%** Deep feature similarity (embedding dot product)
- **30%** Spatial proximity (pixel distance)
- **30%** Color histogram similarity

Score threshold: 0.5

### Mismatch Detection

- Identifies unlinked baggage items
- Triggers security alerts
- Tracks mismatch history

### Power Management

Three adaptive modes:

| Mode | FPS | Resolution | YOLO Interval | Use Case |
|------|-----|------------|---------------|----------|
| ECO | 10 | 640x480 | Every 30 frames | Low activity |
| BALANCED | 20 | 1280x720 | Every 10 frames | Normal operation |
| PERFORMANCE | 30 | 1920x1080 | Every 3 frames | High activity |

### Mesh Network

- Peer discovery via UDP broadcast
- Message routing with TTL
- Hash registry for baggage tracking
- Alert broadcasting
- State synchronization

### Search Interface

Multi-criteria search:
- Text description matching
- Color histogram similarity
- Embedding similarity
- Top-10 result ranking

### UI Overlays

Real-time visualization:
- FPS counter
- Power mode indicator
- Connected peers count
- Detection and link statistics
- Latest alert message

### Keyboard Controls

- `q` - Quit system
- `s` - Interactive search interface

## Configuration

Edit `config/defaults.yaml`:

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

## API Reference

### IntegratedSystem Class

#### Lifecycle Methods

```python
def initialize()
    """Setup all subsystems"""

def start()
    """Begin processing loops (non-blocking)"""

def run()
    """Initialize, start, and run (blocking)"""

def shutdown()
    """Clean shutdown of all subsystems"""
```

#### Search Methods

```python
def search_by_description(description: str) -> List[Dict]
    """Search baggage by text description"""
    # Returns list of matching BaggageProfile dicts
```

#### Metrics

```python
system.metrics.total_frames_processed
system.metrics.total_detections
system.metrics.total_links_found
system.metrics.total_mismatches
system.metrics.avg_processing_time_ms
system.metrics.active_peers
system.metrics.current_power_mode
system.metrics.battery_level
system.metrics.alerts_count
```

## Usage Examples

### Example 1: Basic Operation

```python
from integrated_runtime.integrated_system import IntegratedSystem
import logging

logging.basicConfig(level=logging.INFO)

system = IntegratedSystem()
system.run()  # Blocking
```

### Example 2: Custom Configuration

```python
system = IntegratedSystem(
    config_path="config/production.yaml",
    node_id="entrance-camera-01"
)
system.run()
```

### Example 3: Integration with FastAPI

```python
from integrated_runtime.integrated_system import IntegratedSystem
from fastapi import FastAPI
import threading

# Start orchestrator in background
system = IntegratedSystem()
thread = threading.Thread(target=system.run, daemon=False)
thread.start()

app = FastAPI()

@app.get("/metrics")
async def metrics():
    return {
        'detections': system.metrics.total_detections,
        'links': system.metrics.total_links_found,
        'mode': system.metrics.current_power_mode.name
    }

@app.get("/search/{description}")
async def search(description: str):
    return system.search_by_description(description)
```

### Example 4: Monitoring

```python
system = IntegratedSystem()
system.initialize()
system.start()

import time
while system.running:
    print(f"FPS: {1000.0 / system.metrics.avg_processing_time_ms:.1f}")
    print(f"Detections: {system.metrics.total_detections}")
    time.sleep(5)

system.shutdown()
```

## Performance

### Typical Performance (GPU)

| Metric | Value |
|--------|-------|
| Processing FPS | 20-30 |
| Detection Latency | 50-100ms |
| Linking Latency | 20-50ms |
| Memory Usage | 2-4GB |

### Optimization Tips

1. Use smaller YOLO models (nano/small) for real-time performance
2. Reduce resolution in ECO mode for mobile devices
3. Enable frame skipping for less critical scenarios
4. Batch embedding extraction for multiple detections
5. Disable visualization for production

## Troubleshooting

### Issue: Low FPS

**Solution:**
- Reduce resolution in config
- Switch to yolov8n model
- Disable visualization
- Use ECO power mode

### Issue: High Memory Usage

**Solution:**
- Reduce frame queue size (modify CameraWorker)
- Limit baggage profile history
- Disable embedding caching

### Issue: Mesh Network Not Connecting

**Solution:**
- Check firewall allows UDP on port 9999
- Verify peer IP addresses
- Check network connectivity
- Review mesh logs

### Issue: CUDA Out of Memory

**Solution:**
- Use CPU device (set in config)
- Reduce batch size
- Use smaller YOLO model
- Reduce input resolution

## Files

```
integrated_runtime/
├── __init__.py
├── integrated_system.py         # Main orchestrator (966 lines)
├── IMPLEMENTATION_SUMMARY.md    # Implementation overview
├── ORCHESTRATOR_DOCUMENTATION.md # Detailed documentation
├── quick_start.py               # Usage examples
└── README.md                    # This file
```

## System State Machine

```
┌─────────┐
│ STOPPED │ <─── shutdown()
└────┬────┘
     │ initialize()
     ▼
┌──────────────┐
│ INITIALIZING │
└────┬─────────┘
     │ success
     ▼
┌─────────┐
│ RUNNING │ <─── start()
└────┬────┘
     │ error
     ▼
┌───────┐
│ ERROR │
└───┬───┘
    │ shutdown()
    ▼
┌──────────────┐
│ SHUTTING_DOWN│
└──────┬───────┘
       │
       ▼
    STOPPED
```

## Threading Considerations

### Thread Safety

- Frame queue: Built-in thread-safe queue
- Profiles: Protected by `profiles_lock`
- Alerts: Protected by `alerts_lock`
- Mesh peers: Protected by `mesh.peers_lock`

### Graceful Shutdown

The system handles shutdown signals (SIGINT, SIGTERM) and performs:

1. Set `running = False`
2. Stop all camera workers
3. Mesh protocol cleanup
4. Close visualization windows
5. Release all resources

## Integration Points

### With Vision Module

- YOLODetectionEngine
- EmbeddingExtractor
- ColorDescriptor
- BaggageProfile registry

### With Power Module

- MotionAnalyzer
- ObjectDensityAnalyzer
- PowerModeController

### With Mesh Module

- MeshProtocol
- Message routing
- Peer discovery
- Hash registry

### With Backend

- REST API endpoints
- WebSocket streaming
- Alert webhooks
- Metrics export

## Future Enhancements

- Multi-camera tracking with trajectory prediction
- Distributed learning across peers
- Advanced alert rules engine
- Analytics dashboard
- Database persistence
- Edge model optimization
- Redundant node failover

## Support

For issues, questions, or contributions:
- Check IMPLEMENTATION_SUMMARY.md for detailed component info
- Review ORCHESTRATOR_DOCUMENTATION.md for API details
- See quick_start.py for usage examples
- Check logs for detailed error messages

## License

Part of Kodikon Baggage Tracking System
