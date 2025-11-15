# IP Webcam Streaming Module - Documentation

## Overview

The `phone_stream_viewer.py` module provides a lightweight, efficient IP Webcam reader for Android phone camera feeds with support for multi-feed grid display and optional YOLO object detection inference.

## Features

### Core Capabilities

- **Threaded Frame Capture**: Asynchronous frame capture from multiple IP Webcam streams
- **Multi-Feed Grid Display**: Automatically arranges multiple streams in optimal grid layout
- **Auto-Reconnection**: Handles network interruptions with configurable retry logic
- **Low Latency**: Minimized buffer size for real-time viewing
- **Optional YOLO Inference**: Real-time object detection overlay on video feeds

### Architecture

```plaintext
PhoneStreamViewer (Main Orchestrator)
├── StreamGridDisplay (Rendering)
│   └── Grid layout calculation & stitching
├── WebcamStream (Per-feed Handler) [Multiple]
│   ├── Frame Capture Thread
│   ├── Auto-reconnection Logic
│   ├── YOLO Inference Thread (Optional)
│   └── Frame & Detection Buffers
└── YOLO Model (Shared/Per-stream)
```

## Installation

### Prerequisites

```bash
# Core dependencies (already in requirements.txt)
pip install opencv-python numpy

# Optional: For YOLO inference
pip install ultralytics torch torchvision
```

### Quick Setup

```bash
# Install in development mode
cd Kodikon
pip install -r requirements.txt
```

## Usage

### 1. Command-Line Usage

#### Single Stream

```bash
python streaming/phone_stream_viewer.py \
  --url http://192.168.1.100:8080/video \
  --name "Phone1"
```

#### Multiple Streams (Command-line)

```bash
python streaming/phone_stream_viewer.py \
  --url http://192.168.1.100:8080/video --name "Phone1" \
  --url http://192.168.1.101:8080/video --name "Phone2" \
  --url http://192.168.1.102:8080/video --name "Phone3"
```

#### Multiple Streams with YOLO

```bash
python streaming/phone_stream_viewer.py \
  --config streaming/config_example.json \
  --yolo \
  --confidence 0.5
```

### 2. Configuration File

Create `streams.json`:

```json
{
  "streams": [
    {
      "url": "http://192.168.1.100:8080/video",
      "name": "Entrance",
      "enable_yolo": true,
      "confidence_threshold": 0.5
    },
    {
      "url": "http://192.168.1.101:8080/video",
      "name": "Exit",
      "enable_yolo": false
    }
  ]
}
```

Then run:

```bash
python streaming/phone_stream_viewer.py --config streams.json --yolo
```

### 3. Programmatic Usage

```python
from streaming import PhoneStreamViewer, StreamConfig

# Create stream configurations
configs = [
    StreamConfig(
        url="http://192.168.1.100:8080/video",
        name="Phone1",
        enable_yolo=True,
        confidence_threshold=0.5
    ),
    StreamConfig(
        url="http://192.168.1.101:8080/video",
        name="Phone2",
        enable_yolo=True
    )
]

# Initialize viewer
viewer = PhoneStreamViewer(configs, enable_yolo=True)

# Run the viewer
viewer.run()
```

## Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit the viewer |
| `r` | Reset/reconnect all streams |
| `s` | Save screenshot of current grid |

## Configuration Parameters

### StreamConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | str | - | IP Webcam stream URL |
| `name` | str | - | Display name for stream |
| `max_retries` | int | 3 | Max connection attempts |
| `retry_delay` | int | 2 | Delay between retries (seconds) |
| `enable_yolo` | bool | False | Enable YOLO inference |
| `confidence_threshold` | float | 0.5 | YOLO detection confidence |

### Command-Line Options

```plaintext
--url URL              IP Webcam stream URL (repeatable)
--name NAME            Stream display name (repeatable)
--config FILE          JSON configuration file
--yolo                 Enable YOLO inference
--confidence FLOAT     YOLO confidence threshold (0-1)
--retries INT          Max connection retries
--retry-delay INT      Seconds between retries
```

## Frame Display

### Grid Layout

- 1 stream: Single window
- 2 streams: 1x2 grid
- 3 streams: Optimal layout (e.g., 2x2 with padding)
- 4+ streams: Dynamic grid calculation

### Frame Information

Each frame displays:

- **Stream Name**: Top-left corner
- **Connection Status**: "CONNECTED" (green) or "RECONNECTING" (red)
- **Frame Count**: Total frames captured for this stream
- **Detection Count**: Number of objects detected (if YOLO enabled)
- **Bounding Boxes**: Overlaid detection boxes with class labels and confidence

## YOLO Inference

### Model Selection

- Uses YOLOv8 nano model (`yolov8n.pt`) for lightweight inference
- Automatically downloads on first run
- ~6-7 MB model size

### Performance Considerations

- Inference runs in separate thread per stream
- Processing throttled to ~10 Hz to reduce CPU usage
- Minimal impact on frame capture performance

### Detection Classes

Detects standard COCO dataset classes:

- person, bicycle, car, dog, cat, etc. (80 classes)

### Customization

To use different YOLO model or custom weights:

```python
# In WebcamStream._initialize_yolo()
self.yolo_model = YOLO("yolov8m.pt")  # Medium model
# or
self.yolo_model = YOLO("/path/to/custom_weights.pt")
```

## Troubleshooting

### Connection Issues

```bash
# Test stream URL with ffmpeg
ffmpeg -i "http://192.168.1.100:8080/video" -t 5 -f null -

# Increase retry attempts and delay
python streaming/phone_stream_viewer.py \
  --url http://192.168.1.100:8080/video \
  --retries 5 --retry-delay 3
```

### High CPU Usage

- Disable YOLO inference if not needed
- Reduce number of concurrent streams
- Check network bandwidth

### Low Frame Rate

- Reduce stream resolution on phone
- Bring phone closer to WiFi router
- Disable YOLO inference
- Close other bandwidth-consuming applications

## Examples

### Example 1: Airport Baggage Linking

```bash
python streaming/phone_stream_viewer.py \
  --url http://10.197.139.108:8080/video --name "Check-in" \
  --url http://10.197.139.199:8080/video --name "Baggage Claim" \
  --yolo --confidence 0.6
```

### Example 2: Multi-Location Surveillance

```json
{
  "streams": [
    {"url": "http://location1:8080/video", "name": "Hall A", "enable_yolo": true},
    {"url": "http://location2:8080/video", "name": "Hall B", "enable_yolo": true},
    {"url": "http://location3:8080/video", "name": "Hall C", "enable_yolo": false},
    {"url": "http://location4:8080/video", "name": "Hall D", "enable_yolo": false}
  ]
}
```

### Example 3: Development/Testing

```python
from streaming import WebcamStream, StreamConfig

# Single stream for testing
config = StreamConfig(
    url="http://localhost:8080/video",
    name="TestStream",
    enable_yolo=False
)

stream = WebcamStream(config).start()

# Capture frames
while True:
    frame = stream.read()
    if frame is not None:
        print(f"Frame shape: {frame.shape}")
    time.sleep(0.1)
```

## Performance Metrics

### Tested Specifications

- **CPU Usage**: ~15-20% (4 streams, no YOLO)
- **Memory**: ~200-300 MB (4 streams, no YOLO)
- **Network**: ~2-4 Mbps per stream (depending on resolution/quality)
- **Latency**: ~200-500 ms per stream

### With YOLO (4 streams, YOLOv8n)

- **CPU Usage**: ~60-80%
- **Memory**: ~1-1.5 GB
- **Latency**: ~300-800 ms

## Integration with Kodikon

### With Vision Module

```python
from streaming import PhoneStreamViewer, StreamConfig
from vision import BaggageLinking

# Setup viewer
configs = [StreamConfig(...)]
viewer = PhoneStreamViewer(configs, enable_yolo=True)

# Can feed detections to vision module for linking
detections = viewer.streams[0].get_detections()
for det in detections:
    if det['class'] == 'person':
        # Process person detection for bag linking
        pass
```

## Logging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Log output includes:

- Stream connection status
- Frame capture FPS
- YOLO inference results
- Reconnection attempts

## Notes

- Requires IP Webcam app installed on Android phone
- Phone must be on same network as viewer
- Ensure sufficient network bandwidth for multiple streams
- YOLO inference optional and can be toggled per stream

## Future Enhancements

- [ ] Recording support (mp4 output)
- [ ] Stream statistics dashboard
- [ ] Custom YOLO model support
- [ ] Edge device optimization (TensorRT, ONNX)
- [ ] WebRTC support for remote viewing
- [ ] Multi-node distributed streaming

