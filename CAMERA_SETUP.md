# Camera Setup Guide

## IP Camera Configuration

The system supports multiple IP camera streams from phones or IP cameras.

### Configuration File

Edit `config/cameras.yaml` to configure your cameras:

```yaml
cameras:
  - id: cam1
    name: "Camera 1"
    url: "http://10.7.74.56:8080/video"
    enabled: true
    
  - id: cam2
    name: "Camera 2"
    url: "http://10.7.74.165:8080/video"
    enabled: true
    
  - id: cam3
    name: "Camera 3"
    url: "http://10.7.74.168:8080/video"
    enabled: true

fallback_to_local: true
local_camera_id: 0
```

### Using Phone as IP Camera

#### Android - IP Webcam App

1. **Install IP Webcam**
   - Download from Google Play Store: "IP Webcam" by Pavel Khlebovich
   - Or search for "IP Webcam" in Play Store

2. **Configure the App**
   - Open IP Webcam app
   - Scroll down and tap "Start server"
   - Note the URL shown (e.g., `http://192.168.1.100:8080`)

3. **Update Configuration**
   - The video stream URL is: `http://YOUR_PHONE_IP:8080/video`
   - Update `config/cameras.yaml` with this URL

#### iOS - EpocCam or similar

1. **Install EpocCam** (or similar app)
   - Download from App Store

2. **Get Stream URL**
   - Start the app
   - Note the IP address and port
   - Update `config/cameras.yaml`

### Network Requirements

1. **Same Network**
   - All phones/cameras must be on the same WiFi network as the computer
   - Check your computer's IP: `ipconfig` (Windows) or `ifconfig` (Linux/Mac)
   - Check phone's IP in WiFi settings

2. **Firewall**
   - Ensure port 8080 (or your camera's port) is not blocked
   - Windows: Allow through Windows Firewall
   - Router: Check port forwarding if needed

### Testing Camera URLs

Run the test script to verify cameras are accessible:

```bash
python test_camera_urls.py
```

Expected output:
```
✓ ACCESSIBLE: cam1 - http://10.7.74.56:8080/video
✓ ACCESSIBLE: cam2 - http://10.7.74.165:8080/video
✓ ACCESSIBLE: cam3 - http://10.7.74.168:8080/video
```

### Troubleshooting

#### Camera Not Accessible

1. **Check Phone App**
   - Is the camera app running?
   - Is the server started?
   - Check the IP address shown in the app

2. **Check Network**
   ```bash
   # Ping the phone
   ping 10.7.74.56
   
   # Test the URL in browser
   # Open: http://10.7.74.56:8080
   ```

3. **Check Firewall**
   - Temporarily disable firewall to test
   - Add exception for port 8080

4. **Update IP Addresses**
   - Phone IPs may change when reconnecting to WiFi
   - Check current IP in phone's WiFi settings
   - Update `config/cameras.yaml`

#### Camera Opens But No Frames

1. **Check URL Format**
   - Correct: `http://10.7.74.56:8080/video`
   - Wrong: `http://10.7.74.56:8080` (missing /video)

2. **Check App Settings**
   - Some apps have different stream endpoints
   - Try: `/video`, `/videofeed`, `/cam/1/stream`

3. **Check Resolution**
   - Lower resolution in camera app if frames are slow
   - Recommended: 640x480 or 1280x720

### Fallback to Local Camera

If IP cameras fail, the system will automatically use your computer's webcam:

```yaml
fallback_to_local: true
local_camera_id: 0  # 0 = default webcam
```

To disable fallback:
```yaml
fallback_to_local: false
```

### Multiple Camera Setup

The dashboard will display all active cameras in a grid:

```
┌─────────────┬─────────────┬─────────────┐
│   Camera 1  │   Camera 2  │   Camera 3  │
│   (cam1)    │   (cam2)    │   (cam3)    │
│             │             │             │
│  Live Feed  │  Live Feed  │  Live Feed  │
│             │             │             │
│  FPS: 25.3  │  FPS: 28.1  │  FPS: 30.0  │
└─────────────┴─────────────┴─────────────┘
```

### Performance Tips

1. **Reduce Resolution**
   - Lower resolution = faster processing
   - Recommended: 640x480 for 3+ cameras

2. **Adjust FPS**
   - Edit `config/cameras.yaml`:
   ```yaml
   fps: 15  # Lower FPS for better performance
   ```

3. **Disable Unused Cameras**
   ```yaml
   - id: cam3
     enabled: false  # Disable this camera
   ```

### Running the System

Once cameras are configured:

```bash
# Test cameras first
python test_camera_urls.py

# If cameras are accessible, start the system
python run_command_centre.py
```

Access dashboard at: http://localhost:8000

### Camera App Recommendations

**Android:**
- IP Webcam (Free, reliable)
- DroidCam (Free/Paid)
- iVCam (Free/Paid)

**iOS:**
- EpocCam (Free/Paid)
- iVCam (Free/Paid)
- Iriun Webcam (Free)

**Features to Look For:**
- ✓ HTTP/MJPEG streaming
- ✓ Adjustable resolution
- ✓ Adjustable FPS
- ✓ Low latency
- ✓ Stable connection
