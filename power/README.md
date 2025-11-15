# Dynamic Power Management System

## Overview

The Dynamic Power Management System provides intelligent battery optimization for the Kodikon distributed surveillance network. It combines local adaptive control with network-level coordination to maximize battery life while maintaining surveillance quality.

## Architecture

### Two-Level System

**Level 1: Local Power Mode Controller**
- Analyzes local motion and object density
- Adaptively adjusts FPS, resolution, and YOLO detection intervals
- Responds to battery state and active tracking
- Makes real-time decisions based on local conditions

**Level 2: Collaborative Network Manager**
- Coordinates power across all mesh nodes
- Performs load balancing
- Manages battery emergencies
- Optimizes network-wide power efficiency

## Features

### 1. Motion Analysis
- Optical flow-based motion detection
- Configurable motion thresholds
- Historical motion tracking
- Automatic motion area calculation

### 2. Object Density Analysis
- Calculates detection area ratio
- Tracks object count trends
- Density-weighted activity scoring
- Historical averaging for stability

### 3. Activity Density Calculation
- **Motion Score**: 0-1 from optical flow analysis
- **Object Density Score**: 0-1 from detection coverage
- **Combined Density**: Weighted average (40% motion, 60% objects)
- **Activity Levels**: VERY_LOW, LOW, MODERATE, HIGH, VERY_HIGH

### 4. Adaptive FPS Control
- **ECO**: 10 FPS
- **BALANCED**: 20 FPS
- **PERFORMANCE**: 30 FPS
- Automatic transitions based on activity and battery

### 5. Resolution Switching
- **ECO**: 640x480 (307K pixels)
- **BALANCED**: 1280x720 (922K pixels)
- **PERFORMANCE**: 1920x1080 (2.1M pixels)
- Maintains aspect ratio on custom scales

### 6. YOLO Detection Interval Control
- **ECO**: Every 30 frames (~0.3 Hz at 10 FPS)
- **BALANCED**: Every 10 frames (~2 Hz at 20 FPS)
- **PERFORMANCE**: Every 3 frames (~10 Hz at 30 FPS)
- Massive power savings in detection-heavy workloads

### 7. Tracking-Driven High-Mode Override
- Maintains performance mode while tracking active objects
- 30-second grace period after tracking ends
- Ensures tracking accuracy and reliability
- Automatic return to power-conscious mode

### 8. Network Load Balancing
- Multiple strategies: Round-robin, Least-loaded, Density-aware, Capability-aware
- Distributes high-load tasks to available nodes
- Avoids overloading critical nodes
- Predicts battery depletion time

### 9. Battery Emergency Management
- Automatic ECO mode when battery drops below reserve
- Progressive degradation as battery depletes
- Emergency alerts for critical levels
- Network-coordinated shutdown support

## Core Classes

### PowerModeController
Main local controller for power optimization.

```python
from power import PowerModeController, PowerConfig

# Create controller with default config
controller = PowerModeController()

# Or with custom config
config = PowerConfig(
    fps_eco=10,
    fps_balanced=20,
    fps_performance=30,
    battery_reserve=15.0
)
controller = PowerModeController(config)

# Update battery level
controller.update_battery_level(75.5)

# Analyze frame with detections
detections = [
    {'bbox': [100, 100, 200, 200], 'class': 'person', 'confidence': 0.95},
    {'bbox': [300, 150, 450, 300], 'class': 'backpack', 'confidence': 0.87}
]
activity = controller.analyze_frame(frame, detections)

# Update tracking state
controller.update_tracking(active_track_count=2)

# Update power mode
mode = controller.update_power_mode()

# Get current settings
fps = controller.get_current_fps()  # Returns float
width, height = controller.get_current_resolution()  # Returns (int, int)
interval = controller.get_yolo_interval()  # Returns int

# Check if YOLO should run on this frame
if controller.should_run_yolo(frame_number=42):
    # Run YOLO detection
    pass

# Get statistics
stats = controller.get_power_stats()
```

### CollaborativePowerManager
Network-level power management and coordination.

```python
from power import CollaborativePowerManager, NodePowerMetrics

# Create manager
manager = CollaborativePowerManager(local_node_id="camera_001")

# Register node metrics
metrics = NodePowerMetrics(
    node_id="camera_001",
    current_mode="balanced",
    battery_level=75.5,
    fps=20.0,
    resolution_width=1280,
    resolution_height=720,
    yolo_interval=10,
    activity_density=0.45,
    active_tracks=2,
    cpu_usage=35.0,
    memory_usage=42.0
)
manager.register_node_metrics(metrics)

# Analyze network health
health = manager.analyze_network_health()

# Detect battery emergencies
emergencies = manager.detect_battery_emergencies()

# Get power recommendation for node
allocation = manager.recommend_power_allocation("camera_001", metrics)

# Perform load balancing
lb_decision = manager.balance_load_across_network()

# Network-wide optimization
results = manager.optimize_network_power()

# Get stats
stats = manager.get_network_stats()
```

### NetworkPowerCoordinator
Bridges local controller and network manager, handles mesh integration.

```python
from power import NetworkPowerCoordinator
from mesh import MeshProtocol

# Create mesh
mesh = MeshProtocol("camera_001", port=9999)
mesh.start()

# Create coordinator
coordinator = NetworkPowerCoordinator("camera_001", mesh_protocol=mesh)
coordinator.set_local_controller(power_controller)

# Update metrics from local controller
metrics = coordinator.update_metrics()

# Broadcast to mesh network
coordinator.broadcast_metrics_to_mesh()

# Receive metrics from peer
coordinator.receive_peer_metrics("camera_002", {
    'current_mode': 'balanced',
    'battery_level': 85.0,
    'fps': 20.0,
    # ... more fields
})

# Run optimization cycle
results = coordinator.run_optimization_cycle()
```

## Power Modes

### ECO Mode
- **FPS**: 10
- **Resolution**: 640x480
- **YOLO Interval**: 30 frames
- **Use Case**: Idle monitoring, battery preservation
- **Power Consumption**: ~2W average

### BALANCED Mode
- **FPS**: 20
- **Resolution**: 1280x720
- **YOLO Interval**: 10 frames
- **Use Case**: Normal surveillance
- **Power Consumption**: ~5W average

### PERFORMANCE Mode
- **FPS**: 30
- **Resolution**: 1920x1080
- **YOLO Interval**: 3 frames
- **Use Case**: Active tracking, high-security areas
- **Power Consumption**: ~8W average

## Activity Levels

| Level | Density | Motion | Objects | Use Cases |
|-------|---------|--------|---------|-----------|
| VERY_LOW | <0.05 | Minimal | <5% area | Empty scene, night mode |
| LOW | 0.05-0.2 | Light | 5-20% area | Sparse activity, idle periods |
| MODERATE | 0.2-0.5 | Regular | 20-50% area | Normal surveillance |
| HIGH | 0.5-0.8 | Heavy | 50-80% area | Crowded, fast motion |
| VERY_HIGH | >0.8 | Extreme | >80% area | Dense crowd, emergency |

## Configuration

### Default Configuration
```yaml
power:
  mode: "balanced"           # eco, balanced, performance
  min_fps: 10                # Minimum FPS to maintain
  max_fps: 30                # Maximum FPS
  motion_threshold: 0.3      # Optical flow threshold
  battery_reserve: 15        # Min battery % to reserve
```

### Custom Configuration
```python
from power import PowerConfig, ResolutionConfig

config = PowerConfig(
    # FPS settings
    fps_eco=8,
    fps_balanced=18,
    fps_performance=28,
    
    # Resolution settings
    resolution_eco=ResolutionConfig(480, 360),
    resolution_balanced=ResolutionConfig(1280, 720),
    resolution_performance=ResolutionConfig(1920, 1080),
    
    # YOLO intervals
    yolo_interval_eco=40,
    yolo_interval_balanced=12,
    yolo_interval_performance=2,
    
    # Thresholds
    motion_threshold=0.25,
    battery_reserve=20.0,
    battery_eco_threshold=25.0,
    battery_balanced_threshold=60.0
)

controller = PowerModeController(config)
```

## Usage Patterns

### Pattern 1: Basic Power Management
```python
from power import PowerModeController
import cv2

controller = PowerModeController()
cap = cv2.VideoCapture(0)
frame_number = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Update battery (from system)
    battery = get_battery_level()
    controller.update_battery_level(battery)
    
    # Analyze frame
    detections = run_detection(frame) if controller.should_run_yolo(frame_number) else []
    controller.analyze_frame(frame, detections)
    
    # Update power mode
    controller.update_power_mode()
    
    # Get current settings
    fps = controller.get_current_fps()
    width, height = controller.get_current_resolution()
    
    # Resize frame
    frame = cv2.resize(frame, (width, height))
    
    # Process frame at current FPS
    process_frame(frame)
    
    frame_number += 1
    cap.release()
```

### Pattern 2: Network Coordination
```python
from power import NetworkPowerCoordinator, PowerModeController
from mesh import MeshProtocol
import threading

# Setup power controller
power_controller = PowerModeController()

# Setup mesh
mesh = MeshProtocol("camera_001")
mesh.start()

# Setup coordinator
coordinator = NetworkPowerCoordinator("camera_001", mesh)
coordinator.set_local_controller(power_controller)

# Background optimization thread
def optimization_loop():
    while True:
        results = coordinator.run_optimization_cycle()
        time.sleep(10)  # Every 10 seconds

threading.Thread(target=optimization_loop, daemon=True).start()

# Main processing loop
frame_number = 0
while True:
    # ... capture frame ...
    
    # Update controller
    power_controller.update_battery_level(battery)
    power_controller.analyze_frame(frame, detections)
    power_controller.update_power_mode()
    
    frame_number += 1
```

### Pattern 3: Tracking-Driven Optimization
```python
from power import PowerModeController

controller = PowerModeController()

tracker = initialize_tracker()

while True:
    # ... capture frame ...
    
    # Run tracking
    tracked_objects = tracker.update(frame)
    
    # Update controller with active tracks
    controller.update_tracking(len(tracked_objects))
    
    # Analyze frame
    controller.analyze_frame(frame, detections)
    controller.update_power_mode()
    
    # High-mode override if tracking active
    if len(tracked_objects) > 0:
        logger.info("Maintaining high performance for tracking")
```

## Performance Impact

### Power Savings Example
Assuming 8-hour operational day:

**Without Power Management** (Constant 30 FPS, 1920x1080):
- Power: 8W average
- Daily Energy: 64 Wh
- Battery Life: ~8 hours

**With Power Management**:
- Daytime (high activity): 2 hours @ 8W = 16 Wh
- Afternoon (moderate): 4 hours @ 5W = 20 Wh
- Evening (low activity): 2 hours @ 2W = 4 Wh
- **Daily Energy**: 40 Wh (~38% savings)
- **Battery Life**: ~12-13 hours

## Metrics and Monitoring

### Activity Density Metrics
- `motion_score`: 0-1, from optical flow
- `object_density`: 0-1, from detection coverage
- `combined_density`: 0-1, weighted combination
- `activity_level`: Categorical level (VERY_LOW to VERY_HIGH)

### Power Metrics
- `current_mode`: Power mode name
- `battery_level`: Current battery %
- `current_fps`: Target FPS
- `current_resolution`: Width x Height
- `yolo_interval`: Frames between detections
- `power_score`: 0-1, current power consumption estimate

### Network Metrics
- `healthy_nodes`: Nodes with good battery
- `warning_nodes`: Nodes with low battery
- `critical_nodes`: Nodes with critical battery
- `network_load`: Average system load
- `average_power_score`: Network average power consumption

## Integration Points

### With Streaming Module
- Get current FPS requirement
- Get resolution requirements
- Frame capture loop integration

### With Vision Module
- YOLO detection scheduling
- Detection-based activity analysis
- Object density calculation

### With Mesh Network
- Broadcast power metrics
- Receive peer metrics
- Coordinate load balancing
- Share power recommendations

### With Integrated Runtime
- Update power mode based on tasks
- Get current resource availability
- Coordinate system-wide power

## Advanced Features

### Battery Depletion Prediction
```python
# Predict when battery will be depleted
seconds_remaining = manager.predict_battery_depletion_time("camera_001")
hours_remaining = seconds_remaining / 3600

if hours_remaining < 1:
    # Take emergency action
    trigger_low_battery_protocol()
```

### Load Migration Suggestions
```python
# Get suggestions for load migration
migration_plan = manager.suggest_load_migration(
    source_node="camera_001",
    target_nodes=["camera_002", "camera_003"]
)

# migration_plan contains:
# - targets with available capacity
# - estimated migration time
# - expected power savings
```

### Network Health Analysis
```python
health = manager.analyze_network_health()

if health['critical_nodes'] > 2:
    # Too many nodes in critical state
    activate_emergency_power_mode()

if health['network_load'] > 0.8:
    # Network overloaded
    enable_load_balancing()
```

## Troubleshooting

### Issue: Not Switching to Eco Mode
**Solution**: Check battery level is actually changing. Verify motion threshold is appropriate for scene.

### Issue: Tracking Quality Degrading
**Solution**: Ensure tracking override is enabled. Verify active track count is being updated correctly.

### Issue: High Power Consumption
**Solution**: Check if in PERFORMANCE mode. Review activity density calculations. Verify YOLO interval is appropriate.

### Issue: Network Coordination Not Working
**Solution**: Verify mesh network is operational. Check metrics broadcast interval. Ensure load balancing strategy is appropriate.

## Future Enhancements

1. **Machine Learning**: Predict optimal power settings based on historical patterns
2. **Thermal Management**: Factor in device temperature in power decisions
3. **Predictive Load Balancing**: Anticipate peak loads and pre-distribute work
4. **Edge Computing**: Distribute processing to less power-intensive nodes
5. **Battery Health Tracking**: Monitor battery degradation over time
6. **Cloud Sync**: Sync power decisions with cloud analytics platform

---

**Status**: Production Ready  
**Last Updated**: November 15, 2025
