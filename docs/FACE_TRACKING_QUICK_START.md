# FACE TRACKING QUICK START GUIDE

**Time to implement:** 2-3 hours | **Complexity:** Medium | **Breaking changes:** None

---

## OVERVIEW

Adding face-based person re-identification + timestamped video history + backtrack search to Kodikon without touching YOLO or mesh protocol architecture.

**What you're adding:**
1. ResNet-50 face embeddings (512-dim vectors)
2. Frame ring buffer (300 frames = ~10 sec @ 30fps)
3. Event logging (PERSON_IN/OUT to JSON files)
4. Backtrack search (find people by reference face in history)
5. Mesh broadcasting for distributed face search

---

## STEP-BY-STEP INTEGRATION

### Step 1: Install Dependencies (5 min)

```bash
pip install torchvision>=0.14.0
```

### Step 2: Add Face Embedding Extractor (15 min)

**File:** `vision/baggage_linking.py`  
**Location:** After `EmbeddingExtractor` class (~line 500)  
**Copy:** Entire `FaceEmbeddingExtractor` class from `FACE_TRACKING_CODE_PATCHES.md` → PATCH 1

### Step 3: Add Frame Buffer Classes (20 min)

**File:** `integrated_system.py`  
**Location:** After imports (~line 50)  
**Copy:** `TimestampedFrame`, `FrameHistoryBuffer`, `EventLogger`, `PersonTracker` classes from PATCH 2

### Step 4: Initialize in IntegratedSystem (10 min)

**File:** `integrated_system.py`  
**Locations:**
- `__init__()`: Add 4 new instance variables (from PATCH 3, part 1)
- `initialize()`: Add frame tracking init code (from PATCH 3, part 2)

### Step 5: Track Persons & Log Events (15 min)

**File:** `integrated_system.py`  
**Location:** `_process_frame()` method  
**Action:** Call person tracking + add frames to history (PATCH 4, part 1)  
**Action:** Add person tracking method (PATCH 4, part 2)

### Step 6: Add Backtrack Search (10 min)

**File:** `integrated_system.py`  
**Location:** Add new method to `IntegratedSystem` class  
**Copy:** `backtrack_face_search()` from PATCH 5

### Step 7: Update UI Overlay (10 min)

**File:** `integrated_system.py`  
**Location:** `_draw_overlays()` method  
**Action:** Add face search results display (PATCH 6)

### Step 8: Extend Mesh Protocol (15 min)

**File:** `mesh/mesh_protocol.py`  
**Location 1:** `MessageType` enum → add 3 new types (PATCH 7, part 1)  
**Location 2:** `MeshProtocol` class → add 2 methods (PATCH 7, part 2)

### Step 9: Update Config (5 min)

**File:** `config/defaults.yaml`  
**Action:** Add these sections:

```yaml
face_tracking:
  enabled: true
  embedding_dim: 512
  device: "cuda"

frame_cache:
  dir: "./frame_cache"
  format: "jpg"

event_logging:
  dir: "./event_logs"
  format: "jsonl"
```

---

## FILE EDITING CHECKLIST

| File | Changes | Status |
|------|---------|--------|
| `vision/baggage_linking.py` | +100 lines (FaceEmbeddingExtractor) | ⬜ |
| `integrated_system.py` | +400 lines (FrameHistoryBuffer, PersonTracker, tracking logic) | ⬜ |
| `mesh/mesh_protocol.py` | +25 lines (new message types + methods) | ⬜ |
| `config/defaults.yaml` | +10 lines (config sections) | ⬜ |
| `requirements.txt` | +1 line (torchvision) | ⬜ |

---

## QUICK TEST

After integration, test each component:

```python
# Test 1: Face extractor loads
from vision.baggage_linking import FaceEmbeddingExtractor
extractor = FaceEmbeddingExtractor()
print("✓ Face extractor loaded")

# Test 2: Frame buffer works
from integrated_system import FrameHistoryBuffer
import numpy as np
buffer = FrameHistoryBuffer(max_memory_frames=10)
dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
buffer.append(dummy_frame, 1731702451.0, 0, "camera-0")
print(f"✓ Buffer appended frame: {buffer.get_buffer_stats()['frames']} frames")

# Test 3: Event logger works
from integrated_system import EventLogger
logger = EventLogger()
logger.log_event({'event': 'TEST', 'person_id': 'p_test'})
print("✓ Event logger created event_logs/events_YYYYMMDD.jsonl")

# Test 4: System starts
system = IntegratedSystem()
system.start()
print("✓ System started")
system.shutdown()
```

---

## USAGE EXAMPLES

### Example 1: Use Face Search from CLI

```python
# After system is running...
import numpy as np

# Get a reference face embedding
reference_embedding = np.random.randn(512)  # In practice, extract from uploaded image

# Search backward 5 minutes
matches = system.backtrack_face_search(
    reference_face_embedding=reference_embedding,
    target_timestamp=time.time(),
    search_window_sec=300.0,
    similarity_threshold=0.75
)

for match in matches:
    print(f"Match: {match['similarity']:.2f} @ {match['timestamp']}")
```

### Example 2: Broadcast Face Search Across Mesh

```python
search_request = {
    'search_id': f"search_{int(time.time())}",
    'reference_face_embedding': reference_embedding.tolist(),
    'target_timestamp': time.time(),
    'search_window_sec': 300.0,
    'similarity_threshold': 0.75,
    'requesting_node_id': system.node_id,
    'camera_ids': []  # Empty = all cameras
}

system.mesh.broadcast_face_search_request(search_request)
print("✓ Broadcast search to mesh network")
```

### Example 3: Query Event Logs

```bash
# View today's events
cat event_logs/events_20231115.jsonl | jq '.[] | select(.event == "PERSON_IN")'

# Count person detections
cat event_logs/events_*.jsonl | jq -c '.event' | sort | uniq -c
```

---

## CONFIGURATION TUNING

### For Mobile (Limited Memory)
```yaml
frame_cache:
  max_disk_frames: 2000      # ~1 hour
memory_buffer_size: 150      # ~5 seconds
```

### For High Security (Keep Everything)
```yaml
frame_cache:
  max_disk_frames: 10000     # ~3 hours
memory_buffer_size: 600      # ~20 seconds
compression: "h264"          # Smaller files
```

### For Fast Search (High CPU)
```yaml
face_tracking:
  device: "cuda"
backtrack_search:
  similarity_threshold: 0.70  # Lower = more results
```

---

## TROUBLESHOOTING

### Issue: Face embeddings slow (30+ms per person)

**Solution:**
- Use `device: "cpu"` if GPU not available
- Reduce `memory_buffer_size` to skip fewer frames
- Use quantized ResNet-50 (INT8)

### Issue: Event log files grow huge

**Solution:**
- Switch to CSV format (`log_format: "csv"`)
- Compress old logs: `gzip event_logs/events_*.jsonl`
- Clear older than 7 days: `find event_logs -name "*.jsonl" -mtime +7 -delete`

### Issue: Frame buffer uses too much RAM

**Solution:**
- Reduce `max_memory_frames` (e.g., 100 = 3 seconds)
- Enable disk persistence: `save_on_alert: true`
- Store frames as JPEG (not PNG): `compression: "jpg"`

### Issue: Backtrack search not finding matches

**Solution:**
- Lower `similarity_threshold` (0.70 instead of 0.75)
- Increase `search_window_sec` (600 instead of 300)
- Verify face embeddings are being extracted (check logs)

---

## PERFORMANCE EXPECTATIONS

| Operation | Latency | Notes |
|-----------|---------|-------|
| Face embedding extraction | 15-30ms | Per person per frame |
| Frame history append | <1ms | Ring buffer operation |
| Event logging | <1ms | Write to disk |
| Backtrack search (300 frames) | 5-15 sec | Depends on YOLO speed |
| Mesh broadcast | <2ms | UDP send |

**Memory Usage:**
- 300 frames @ 1280x720 @ RGB: ~60MB
- 512-dim embeddings (100 persons): <1MB
- Event logs (1 hour, 1 detection/sec): ~2MB

---

## FILES CREATED/MODIFIED

```
vision/
  baggage_linking.py        [MODIFIED] +FaceEmbeddingExtractor
integrated_runtime/
  integrated_system.py      [MODIFIED] +FrameHistoryBuffer, +PersonTracker, +event logging
mesh/
  mesh_protocol.py          [MODIFIED] +face search message types
config/
  defaults.yaml             [MODIFIED] +face_tracking config
requirements.txt            [MODIFIED] +torchvision
docs/
  FACE_TRACKING_INTEGRATION_DESIGN.md    [NEW] Full design doc
  FACE_TRACKING_CODE_PATCHES.md          [NEW] Code patches
```

---

## NEXT STEPS (OPTIONAL ENHANCEMENTS)

1. **Dedicated Face Detector:** Replace person bbox with dedicated face detection (MTCNN, RetinaFace)
2. **Database Backend:** Store embeddings in PostgreSQL + vector DB (Qdrant)
3. **Mobile Optimization:** Quantize ResNet-50 to INT8 for 3x speedup
4. **Face Database:** Allow uploading set of reference faces for matching
5. **Real-time Dashboard:** Web UI showing live person tracks + search results
6. **Alert on Match:** Emit alert when reference face found in real-time stream

---

## SUPPORT

- **Issues?** Check event_logs/ for error messages
- **Slow?** Profile with `python -m cProfile -s cumtime integrated_system.py`
- **Memory leak?** Monitor with `psutil` + `tracemalloc`

