# Streaming Module - Quick Reference

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start (3 ways)

### 1. Command-Line (Simplest)

```bash
# Single stream
python streaming/phone_stream_viewer.py \
  --url http://192.168.1.100:8080/video \
  --name "MyPhone"

# Multiple streams
python streaming/phone_stream_viewer.py \
  --url http://192.168.1.100:8080/video --name "Phone1" \
  --url http://192.168.1.101:8080/video --name "Phone2"

# With YOLO detection
python streaming/phone_stream_viewer.py \
  --config streaming/config_example.json --yolo
```

### 2. Configuration File

Create `my_config.json`:

```json
{
  "streams": [
    {"url": "http://192.168.1.100:8080/video", "name": "Phone1", "enable_yolo": true},
    {"url": "http://192.168.1.101:8080/video", "name": "Phone2", "enable_yolo": false}
  ]
}
```

Run:

```bash
python streaming/phone_stream_viewer.py --config my_config.json --yolo
```

### 3. Python Code

```python
from streaming import PhoneStreamViewer, StreamConfig

configs = [
    StreamConfig(
        url="http://192.168.1.100:8080/video",
        name="Phone1",
        enable_yolo=True
    )
]

viewer = PhoneStreamViewer(configs, enable_yolo=True)
viewer.run()  # Press 'q' to quit
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Reconnect all streams |
| `s` | Save screenshot |

## Getting Stream URLs

### Android IP Webcam App

1. Install "IP Webcam" from Google Play Store
2. Open app, tap "Start server"
3. Copy URL shown in app (e.g., `http://192.168.1.100:8080/video`)
4. Use URL in streaming viewer

## Common Options

```bash
--url URL              Stream URL (repeatable)
--name NAME            Display name (repeatable)
--config FILE          JSON config file
--yolo                 Enable YOLO detection
--confidence 0.5       YOLO confidence (0-1)
--retries 3            Max connection attempts
--retry-delay 2        Seconds between retries
```

## Examples

### Single Stream Viewer

```bash
python streaming/phone_stream_viewer.py \
  --url http://192.168.1.100:8080/video \
  --name "MyPhone"
```

### 4-Stream Airport Setup

```bash
python streaming/phone_stream_viewer.py \
  --url http://check-in:8080/video --name "CheckIn" \
  --url http://baggage:8080/video --name "Baggage" \
  --url http://gate1:8080/video --name "Gate1" \
  --url http://gate2:8080/video --name "Gate2" \
  --yolo --confidence 0.6
```

### Python Script Integration

```python
from streaming import StreamConfig, WebcamStream
import cv2

# Single stream
config = StreamConfig(
    url="http://192.168.1.100:8080/video",
    name="Camera1"
)

stream = WebcamStream(config).start()

# Capture frames
for i in range(100):
    frame = stream.read()
    if frame is not None:
        cv2.imshow("Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

stream.stop()
cv2.destroyAllWindows()
```

## Performance Tips

### Reduce CPU Usage

- Disable YOLO inference: `--yolo` not specified
- Reduce stream resolution on phone
- Use fewer streams

### Improve FPS

- Use wired connection (if possible)
- Reduce number of streams
- Disable YOLO inference
- Close other applications

### Fix Connection Issues

```bash
# Increase retry attempts
python streaming/phone_stream_viewer.py \
  --url http://192.168.1.100:8080/video \
  --retries 10 --retry-delay 5
```

## Troubleshooting

### "Connection refused" error

- Check phone IP address is correct
- Phone and computer on same WiFi network
- IP Webcam app is running on phone
- Firewall not blocking port 8080

### High CPU Usage

- Disable YOLO detection
- Use fewer streams
- Reduce frame rate on phone app

### Low Frame Rate

- Move phone closer to WiFi router
- Check other network usage
- Reduce video quality on phone
- Disable YOLO

## Architecture

```
PhoneStreamViewer (Main)
├── StreamGridDisplay (UI)
└── WebcamStream × N (Per-stream)
    ├── Capture Thread (Frames)
    └── Inference Thread (YOLO)
```

## Classes

### StreamConfig

Configuration dataclass:

```python
StreamConfig(
    url="http://192.168.1.100:8080/video",
    name="Phone1",
    max_retries=3,           # Connection attempts
    retry_delay=2,           # Seconds between retries
    enable_yolo=False,       # Enable YOLO
    confidence_threshold=0.5 # YOLO confidence
)
```

### WebcamStream

Per-stream handler:

```python
stream = WebcamStream(config).start()
frame = stream.read()              # Get current frame
detections = stream.get_detections() # Get YOLO detections
stream.stop()                      # Stop capture
```

### StreamGridDisplay

Grid renderer:

```python
display = StreamGridDisplay()
grid = display.create_grid(frames)  # Stitch frames
display.display(grid)               # Show in window
```

### PhoneStreamViewer

Main orchestrator:

```python
viewer = PhoneStreamViewer(configs, enable_yolo=True)
viewer.run()        # Main loop
viewer.shutdown()   # Cleanup
```

## Run Examples

```bash
python streaming/examples.py

# Select example:
# 1. Single Stream Capture
# 2. Multi-Stream Viewing
# 3. YOLO Detection
# 4. Stream Monitoring
# 5. Custom Processing
# 6. Resilience Testing
# 7. Performance Monitoring
# 8. Vision Integration
```

## Integration with Kodikon

### Access Detections

```python
viewer = PhoneStreamViewer(configs, enable_yolo=True)

# Get detections from stream
for stream in viewer.streams:
    detections = stream.get_detections()
    for det in detections:
        print(f"Found {det['class']} at {det['bbox']}")
```

### Feed to Vision Module

```python
from streaming import PhoneStreamViewer
from vision import BaggageLinking

configs = [StreamConfig(...)]
viewer = PhoneStreamViewer(configs, enable_yolo=True)

linking = BaggageLinking()
frame = viewer.streams[0].read()
# linked_pairs = linking.process(frame, detections)
```

## Logging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Files

- `phone_stream_viewer.py`: Main implementation
- `__init__.py`: Module exports
- `README.md`: Full documentation
- `examples.py`: 8 usage examples
- `config_example.json`: Configuration template
- `IMPLEMENTATION_SUMMARY.md`: Implementation details

## Status

✅ Complete and ready to use

All features implemented:
- Threaded frame capture
- Multi-feed grid display
- YOLO inference (optional)
- Auto-reconnection
- Full documentation
- Examples included
