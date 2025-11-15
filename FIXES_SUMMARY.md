# Fixes Summary

## Issues Fixed

### 1. OpenCV 4.x Compatibility Error
**Problem:** Motion analysis error in `power/power_mode_controller.py`
```
OpenCV(4.12.0) :-1: error: (-
```

**Root Cause:** The code was using OpenCV 3.x API for `cv2.findContours()` which returns 3 values, but OpenCV 4.x only returns 2 values.

**Fix:** Updated line 232 in `power/power_mode_controller.py`:
```python
# Before (OpenCV 3.x)
_, contours, _ = cv2.findContours(...)

# After (OpenCV 4.x)
contours, _ = cv2.findContours(...)
```

### 2. Camera 404 Errors
**Problem:** Cameras cam1 and cam3 returning 404 errors:
```
INFO: 127.0.0.1:51137 - "GET /nodes/frame/command-centre-node?camera_id=cam1&t=1763231232145 HTTP/1.1" 404 Not Found
INFO: 127.0.0.1:53086 - "GET /nodes/frame/command-centre-node?camera_id=cam3&t=1763231232145 HTTP/1.1" 404 Not Found
```

**Root Cause:** Cameras were failing to connect but the endpoint wasn't providing helpful error messages.

**Fix:** Enhanced error handling in `command_centre/routes/nodes.py`:
- Added check for camera existence in dictionary
- Added camera state validation (running/error)
- Increased timeout from 0.1s to 0.5s for frame retrieval
- Added detailed logging for debugging
- Better error messages indicating camera status

### 3. Grid View Display
**Problem:** Need to implement grid view for multiple camera streams like the example code.

**Fix:** The grid view was already implemented in `streaming/phone_stream_viewer.py` in the `StreamGridDisplay` class. Enhanced it with:
- Proper grid layout calculation (rows x cols based on square root)
- Frame padding for incomplete grids
- Automatic resizing to fit screen
- Status indicators for each camera

### 4. Dashboard UI Improvements
**Problem:** Dashboard didn't show camera connection status clearly.

**Fix:** Updated `command_centre/static/dashboard.js` and `dashboard.css`:
- Added visual status indicators (● green for connected, ○ red for offline)
- Added "No Signal" and "Camera Offline" messages
- Better error handling for failed image loads
- Enhanced CSS styling for camera status

## Testing

### Test Camera Grid
Run the test script to verify camera connections:
```bash
python test_camera_grid.py
```

This will:
- Connect to all 3 cameras (cam1, cam2, cam3)
- Display them in a grid layout
- Show connection status for each camera
- Handle disconnections gracefully

### Test Command Centre
Run the command centre to see the dashboard:
```bash
python run_command_centre.py
```

Then open http://localhost:8000 in your browser to see:
- Camera grid with status indicators
- Real-time frame updates
- Connection status for each camera

## Camera Configuration

Cameras are configured in `config/cameras.yaml`:
```yaml
cameras:
  - id: cam1
    url: "http://10.197.139.199:8080/video"
    enabled: true
  - id: cam2
    url: "http://10.197.139.108:8080/video"
    enabled: true
  - id: cam3
    url: "http://10.197.139.192:8080/video"
    enabled: true
```

## Troubleshooting

If cameras show 404 errors:
1. Check if the camera URLs are accessible
2. Verify the IP addresses and ports are correct
3. Ensure the IP Webcam app is running on the phones
4. Check network connectivity
5. Look at the server logs for detailed error messages

The enhanced error handling will now show:
- "Camera not found" if camera_id doesn't exist
- "Camera not available" if camera is in error state
- "No frame available" if camera is connected but no frames yet
