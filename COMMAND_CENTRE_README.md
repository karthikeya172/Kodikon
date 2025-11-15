# Command Centre System

## Overview

The Command Centre is a centralized monitoring and control system for distributed baggage tracking nodes. It provides real-time visualization, face search capabilities, and system-wide coordination.

## Architecture

### Backend Components

1. **FastAPI Server** (`command_centre/server.py`)
   - Main HTTP/WebSocket server
   - Handles REST API requests
   - Manages WebSocket connections

2. **WebSocket Manager** (`command_centre/websocket_manager.py`)
   - Broadcasts real-time events to connected clients
   - Manages client connections
   - Event types: node_status, registration, alert, person_in/out, face_result

3. **API Routes**
   - `routes/nodes.py` - Node status and frame retrieval
   - `routes/logs.py` - Live and historical logs
   - `routes/search.py` - Face search and backtracking

### Frontend Components

1. **Dashboard** (`static/index.html`)
   - Node status display
   - Live camera feed grid
   - System logs viewer
   - Face search interface

2. **JavaScript Client** (`static/dashboard.js`)
   - WebSocket client for real-time updates
   - REST API polling
   - UI rendering and event handling

3. **Styling** (`static/dashboard.css`)
   - Dark theme optimized for monitoring
   - Responsive grid layout

## Integration with Existing System

### Patches Applied

#### 1. `integrated_system.py`
- Added `FrameHistoryBuffer` class for frame storage
- Added `get_status_snapshot()` for node status
- Added `get_current_frame_jpeg()` for live frame retrieval
- Added `get_person_event_log()` for event history
- Added `extract_face_embedding()` for face detection
- Added `run_face_backtrack()` for face search
- Added `set_ws_manager()` for WebSocket integration
- Added person event emission on detection

#### 2. `mesh_protocol.py`
- Added `FACE_SEARCH_REQUEST` message type
- Added `FACE_SEARCH_RESULT` message type
- Added handlers for face search messages

#### 3. `baggage_linking.py` / `vision/baggage_linking.py`
- Added `face_embedding` field to `Detection` dataclass
- Added `face_embedding` field to `BaggageProfile` dataclass
- Added `extract_face_embedding_from_detection()` function

## API Endpoints

### REST API

#### Nodes
- `GET /nodes/status` - Get status of all nodes
- `GET /nodes/frame/{node_id}` - Get current frame as JPEG

#### Logs
- `GET /logs/live?limit=50` - Get live logs
- `GET /logs/history?start_time=X&end_time=Y&event_type=Z` - Get historical logs

#### Search
- `POST /search/face` - Upload face image and search
  - Form data: `file` (image), `timestamp` (optional)
- `GET /search/result/{job_id}` - Get search result by job ID

### WebSocket

#### Connection
- `ws://localhost:8000/ws/status` - Real-time status updates

#### Message Types

**node_status**
```json
{
  "type": "node_status",
  "node_id": "...",
  "power_mode": "BALANCED",
  "fps": 25.3,
  "activity": 150,
  "peers": ["node-1", "node-2"],
  "timestamp": 1234567890.123
}
```

**registration_event**
```json
{
  "type": "registration",
  "hash_id": "abc123...",
  "timestamp": 1234567890.123,
  "details": {}
}
```

**alert_event**
```json
{
  "type": "alert",
  "alert_type": "MISMATCH",
  "details": {
    "message": "...",
    "timestamp": 1234567890.123
  }
}
```

**person_in / person_out**
```json
{
  "type": "person_in",
  "person_id": "p_123",
  "timestamp": 1234567890.123,
  "details": {}
}
```

**face_search_result**
```json
{
  "type": "face_result",
  "job_id": "uuid",
  "match_timestamp": 1234567890.123,
  "confidence": 0.85,
  "frame": "<base64>",
  "hash_id": "abc123..."
}
```

## Running the System

### Prerequisites
```bash
pip install fastapi uvicorn websockets
```

### Start Command Centre
```bash
python run_command_centre.py
```

### Access Dashboard
Open browser to: http://localhost:8000

## Features

### 1. Real-Time Monitoring
- Live node status updates
- FPS, power mode, peer count
- Activity metrics

### 2. Live Camera Feeds
- Grid view of all connected nodes
- Auto-refreshing JPEG streams
- 1-second update interval

### 3. System Logs
- Real-time event stream
- Filterable by type (PERSON_IN, PERSON_OUT, ALERT, REGISTRATION)
- Historical log retrieval

### 4. Face Search & Backtracking
- Upload face image
- Search frame history (±5 minutes from timestamp)
- Returns matching frames with confidence scores
- Links to baggage hash_id

### 5. Mesh Network Visualization
- Connected peer count
- Network topology (future enhancement)

## Configuration

### Frame History
- Default: 300 frames (~10 seconds at 30 FPS)
- Configurable in `FrameHistoryBuffer(max_size=300)`

### Log Buffer
- Default: 500 events
- Configurable in `person_events = deque(maxlen=500)`

### WebSocket Reconnection
- Auto-reconnect every 5 seconds on disconnect
- Configurable in `dashboard.js`

## Development

### Adding New Event Types

1. Add message type to `websocket_manager.py`:
```python
async def send_custom_event(self, data: Dict[str, Any]):
    message = {"type": "custom_event", **data}
    await self.broadcast(message)
```

2. Add handler in `dashboard.js`:
```javascript
case 'custom_event':
    this.handleCustomEvent(message);
    break;
```

3. Emit from `integrated_system.py`:
```python
if self.ws_manager:
    asyncio.create_task(self.ws_manager.send_custom_event(data))
```

### Adding New API Endpoints

1. Create route in `command_centre/routes/`:
```python
@router.get("/custom")
async def custom_endpoint():
    return {"data": "..."}
```

2. Include router in `server.py`:
```python
from command_centre.routes import custom
app.include_router(custom.router, prefix="/custom", tags=["custom"])
```

## Troubleshooting

### WebSocket Not Connecting
- Check firewall settings
- Verify port 8000 is not in use
- Check browser console for errors

### No Camera Feeds
- Ensure integrated system is running
- Check camera initialization in logs
- Verify frame capture is working

### Face Search Not Working
- Ensure embedding extractor is loaded
- Check image format (JPEG/PNG)
- Verify face is visible in uploaded image

## Future Enhancements

1. **Multi-Node Support**
   - Aggregate status from multiple nodes
   - Distributed face search across mesh

2. **Advanced Analytics**
   - Person tracking across cameras
   - Dwell time analysis
   - Traffic flow visualization

3. **Alert Management**
   - Alert acknowledgment
   - Alert escalation rules
   - Notification system

4. **User Authentication**
   - Role-based access control
   - Audit logging
   - Session management

5. **Data Export**
   - CSV/JSON export of logs
   - Frame sequence export
   - Report generation
