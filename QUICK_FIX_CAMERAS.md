# Quick Fix for Camera Issues

## Current Problem

The system is trying to connect to:
- `http://10.7.74.56:8080/video` (cam1)
- `http://10.7.74.165:8080/video` (cam2)
- `http://10.7.74.168:8080/video` (cam3)

But getting connection errors: `Connection to tcp://10.7.74.X:8080 failed`

## Solution Steps

### Step 1: Check Camera App on Phones

1. **Open IP Webcam app on each phone**
2. **Start the server** (scroll down and tap "Start server")
3. **Note the IP address shown** (e.g., `http://192.168.1.100:8080`)

### Step 2: Update Camera Configuration

Edit `config/cameras.yaml` with the correct IP addresses:

```yaml
cameras:
  - id: cam1
    name: "Camera 1"
    url: "http://YOUR_PHONE1_IP:8080/video"  # Update this
    enabled: true
    
  - id: cam2
    name: "Camera 2"
    url: "http://YOUR_PHONE2_IP:8080/video"  # Update this
    enabled: true
    
  - id: cam3
    name: "Camera 3"
    url: "http://YOUR_PHONE3_IP:8080/video"  # Update this
    enabled: true

fallback_to_local: true  # Will use webcam if phones fail
local_camera_id: 0
```

### Step 3: Test Camera URLs

Before running the full system, test if cameras are accessible:

```bash
python test_camera_urls.py
```

You should see:
```
✓ ACCESSIBLE: cam1 - http://...
✓ ACCESSIBLE: cam2 - http://...
✓ ACCESSIBLE: cam3 - http://...
```

### Step 4: Restart the System

```bash
# Stop current system (Ctrl+C)
# Then restart:
python run_command_centre.py
```

## Alternative: Use Local Webcam

If you don't have IP cameras ready, edit `config/cameras.yaml`:

```yaml
cameras: []  # Empty list = no IP cameras

fallback_to_local: true
local_camera_id: 0  # Use computer's webcam
```

Then restart the system.

## Troubleshooting

### Can't Connect to Phone

1. **Check same WiFi network**
   ```bash
   # On computer, check your IP
   ipconfig
   
   # Should be same subnet as phone (e.g., 10.7.74.X or 192.168.1.X)
   ```

2. **Test in browser**
   - Open browser on computer
   - Go to: `http://PHONE_IP:8080`
   - You should see the camera interface

3. **Check firewall**
   - Windows Firewall might be blocking
   - Temporarily disable to test

### Phone IP Changed

Phone IPs can change when reconnecting to WiFi. To get current IP:

**Android:**
- Settings → WiFi → Tap connected network → IP address

**iOS:**
- Settings → WiFi → Tap (i) icon → IP Address

### Using Different Camera App

If using a different app (not IP Webcam), the URL might be different:

- **DroidCam**: `http://IP:4747/video`
- **iVCam**: Check app for URL
- **EpocCam**: Check app for URL

Update the `url` in `config/cameras.yaml` accordingly.

## Quick Test Command

Test a single camera URL:

```bash
# Windows
curl http://10.7.74.56:8080

# Or open in browser
start http://10.7.74.56:8080
```

If you see HTML or video, the camera is accessible!
