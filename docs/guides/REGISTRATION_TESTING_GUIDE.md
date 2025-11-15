# Registration Desk System - Testing Guide

## ✅ Implementation Status: COMPLETE

All components verified and integrated successfully. The registration desk system is ready for end-to-end testing.

---

## 1. Component Verification Results

### Test Summary
```
✅ PASS: baggage_linking          - Registration record management
✅ PASS: mesh_protocol             - Broadcast integration
✅ PASS: udp_setup_guide           - Wrapper methods
✅ PASS: integrated_system         - UI and workflow

Total: 4/4 tests passed
```

### Verified Components

#### baggage_linking.py (478 lines)
- ✅ RegistrationRecord dataclass with serialization
- ✅ Person-bag embedding extraction (ORB + histogram)
- ✅ Color histogram computation (HSV space)
- ✅ Similarity matching (cosine distance)
- ✅ Hash ID generation (SHA256-based)
- ✅ File storage system (registrations/{hash_id}/)

#### mesh/mesh_protocol.py
- ✅ broadcast_hash_registration() method
- ✅ on_hash_registry_received() handler
- ✅ hash_registry_storage dict
- ✅ MessageType.HASH_REGISTRY (type=6)

#### mesh/udp_setup_guide.py
- ✅ broadcast_hash_registration() wrapper
- ✅ get_hash_registry() accessor
- ✅ IntegratedMeshNode integration

#### integrated_runtime/integrated_system.py
- ✅ Registration mode state variables
- ✅ Keyboard handlers (r, SPACE, ESC)
- ✅ UI overlay rendering
- ✅ _process_registration() async method
- ✅ Countdown timer display

---

## 2. Testing Workflow

### Phase 1: Single-Node Registration (Local Testing)

**Setup:**
```bash
# Terminal 1: Start integrated system
cd c:\Users\viswa\GithubClonedRepos\Kodikon
python -m integrated_runtime.integrated_system
```

**Test Steps:**

1. **Enable Registration Mode**
   - Press `r` key
   - Expected: "REGISTRATION MODE: ON" overlay appears in top-left
   - Overlay should show: "R=Toggle | SPACE=Freeze | ESC=Cancel"

2. **Prepare Registration**
   - Position person + bag in camera frame
   - Both should be visible in frame
   - Ensure good lighting for YOLO detection

3. **Freeze and Extract**
   - Press `SPACE` key to freeze frame
   - Expected: "FRAME FROZEN" message appears
   - 1-second countdown: "3... 2... 1..."
   - Progress indicator: "Extracting..." status

4. **Verify Registration Success**
   - Expected: "✓ Registered: {hash_id[:8]}" success message
   - Hash ID format: 12 characters (SHA256 truncated)
   - Example: "a7f3b9c2e156"

5. **Check File Storage**
   ```bash
   # Terminal 2: Verify directory structure
   ls registrations/
   ls registrations/a7f3b9c2e156/  # Or your hash_id
   
   # Should contain:
   # - person.jpg (person image)
   # - bag.jpg (bag image)
   # - record.json (metadata)
   ```

6. **Verify Mesh Broadcast**
   - Check mesh logs for: "Broadcasting hash registration"
   - Expected: Message sent successfully
   - Hash registry updated on local node

7. **Disable Registration Mode**
   - Press `r` key again
   - Expected: "REGISTRATION MODE: OFF" message
   - Return to normal detection mode

### Phase 2: Multi-Node Mesh Broadcast Testing

**Setup:**
```bash
# Terminal 1: Start desk camera (node 1)
python -m integrated_runtime.integrated_system --node-id desk-cam --port 9999

# Terminal 2: Start surveillance camera 1 (node 2)
python -m integrated_runtime.integrated_system --node-id surv-cam-1 --port 9998

# Terminal 3: Start surveillance camera 2 (node 3)
python -m integrated_runtime.integrated_system --node-id surv-cam-2 --port 9997
```

**Test Steps:**

1. **Verify Mesh Connections**
   - All three nodes should connect via UDP mesh
   - Check logs for: "Peer discovered: ..." messages
   - Heartbeats should be received from other nodes

2. **Register on Desk Camera**
   - In Terminal 1: Press `r` to enable registration mode
   - Complete registration steps from Phase 1
   - Note the hash_id generated

3. **Verify Remote Reception**
   - In Terminal 2 & 3: Check logs for broadcast reception
   - Expected: "Received hash registration from desk-cam"
   - Hash registry should be updated on surveillance nodes

4. **Check Multi-Node Hash Registry**
   - Terminal 1:
     ```python
     from mesh.udp_setup_guide import IntegratedMeshNode
     node = IntegratedMeshNode(node_id="desk-cam", port=9999)
     registry = node.get_hash_registry()
     print(f"Local registry: {len(registry)} entries")
     ```
   - Terminal 2 & 3: Same check, should show 1 entry each

### Phase 3: Detection Matching and Annotation

**Setup:**
- One camera in registration mode (desk)
- One camera in detection mode (surveillance)

**Test Steps:**

1. **Register Person-Bag Pair**
   - Desk camera: Complete registration workflow
   - Note hash_id generated

2. **Trigger Detection on Remote Camera**
   - Have the registered person + bag appear on surveillance camera
   - YOLO should detect both objects

3. **Verify Hash Matching**
   - Check logs for: "Matching hash_id against registry"
   - Expected: Hash ID lookup succeeds
   - Annotation should show: "PERSON: {hash_id[:8]}" + confidence

4. **Visual Verification**
   - Bounding box should appear with hash_id annotation
   - Color: Green for registered, red for unknown
   - Text: "{hash_id} (0.95 confidence)"

### Phase 4: Error Recovery and Edge Cases

**Test Scenarios:**

1. **Cancel During Registration**
   - Press `SPACE` to freeze
   - Press `ESC` before extraction completes
   - Expected: Countdown cancels, registration aborted
   - No files created, no broadcast sent

2. **Toggle Registration Mode Multiple Times**
   - Press `r` repeatedly
   - Expected: Mode toggles smoothly without errors
   - No crashes or memory leaks

3. **No Detection in Frame**
   - Press `SPACE` when no person/bag detected
   - Expected: Error message: "No objects detected in frame"
   - Registration cancelled gracefully

4. **Partial Detection**
   - Only person detected, no bag
   - Expected: Warning message
   - Recovery: User can exit registration mode with `r` or `ESC`

5. **Mesh Network Partition**
   - Disconnect surveillance camera from mesh
   - Register on desk camera
   - Expected: Broadcast fails gracefully
   - System continues operating locally

6. **Multiple Rapid Registrations**
   - Complete 3-5 registrations in succession
   - Expected: All hash_ids unique
   - All files stored correctly
   - All broadcasts sent

---

## 3. Logging and Debugging

### Enable Verbose Logging
```python
# In integrated_runtime/integrated_system.py
import logging
logging.basicConfig(level=logging.DEBUG)

# Then run:
python -m integrated_runtime.integrated_system
```

### Key Log Messages to Monitor

**Registration Mode:**
```
INFO - Registration mode toggled: ON
INFO - Frame frozen for registration
INFO - Extracting embeddings...
INFO - Hash ID generated: a7f3b9c2e156
INFO - Images saved to registrations/a7f3b9c2e156/
INFO - Broadcasting hash registration across mesh
INFO - Registration complete
```

**Mesh Broadcast:**
```
INFO - Broadcasting hash registration: {hash_id}
INFO - Sent to {num_peers} peers
INFO - Received hash registration from {node_id}
INFO - Updated hash registry with {hash_id}
```

**Detection Matching:**
```
INFO - Detected person at confidence: 0.92
INFO - Detected bag at confidence: 0.88
INFO - Matching against hash registry
INFO - Match found: a7f3b9c2e156
INFO - Annotating detection with hash_id
```

**Error Cases:**
```
ERROR - No objects detected in frame
ERROR - Frame extraction failed: {reason}
ERROR - Broadcasting failed: {reason}
ERROR - Hash registry lookup failed
```

---

## 4. Performance Metrics

### Expected Performance

**Per Registration:**
- Detection time: 100-200ms (YOLO)
- Feature extraction: 50-100ms (ORB + histogram)
- Hash generation: <1ms (SHA256)
- File I/O: 50-200ms (image save)
- Mesh broadcast: <50ms (UDP)
- **Total: 250-550ms end-to-end**

**Per Detection (Surveillance):**
- YOLO inference: 100-200ms
- Feature extraction: 50-100ms
- Similarity matching: 1-5ms (cosine distance)
- Registry lookup: <1ms (dict)
- Annotation rendering: 10-20ms
- **Total: 160-330ms per detection**

**Mesh Network:**
- Peer discovery: 5s interval
- Heartbeat: 5s interval
- Broadcast propagation: <100ms
- Hash registry sync: <500ms across 3 nodes

---

## 5. Expected Output Examples

### Successful Registration (Console)
```
2025-11-15 20:40:15,234 - INFO - Registration mode toggled: ON
2025-11-15 20:40:18,456 - INFO - Frame frozen for registration
2025-11-15 20:40:19,567 - INFO - Extracting embeddings from frame...
2025-11-15 20:40:19,789 - INFO - Person embedding: 512-dim vector
2025-11-15 20:40:19,801 - INFO - Bag embedding: 512-dim vector
2025-11-15 20:40:19,812 - INFO - Color histogram computed: HSV space
2025-11-15 20:40:19,823 - INFO - Hash ID generated: a7f3b9c2e156
2025-11-15 20:40:19,834 - INFO - Images saved:
  - registrations/a7f3b9c2e156/person.jpg
  - registrations/a7f3b9c2e156/bag.jpg
2025-11-15 20:40:19,845 - INFO - Metadata saved: registrations/a7f3b9c2e156/record.json
2025-11-15 20:40:19,856 - INFO - Broadcasting hash registration across mesh...
2025-11-15 20:40:19,867 - INFO - Mesh broadcast successful (3 peers notified)
2025-11-15 20:40:19,878 - INFO - ✓ Registration complete!
```

### Multi-Node Broadcast (Terminal 2)
```
2025-11-15 20:40:19,867 - INFO - Received hash registration from desk-cam
2025-11-15 20:40:19,868 - INFO - Processing registration: a7f3b9c2e156
2025-11-15 20:40:19,869 - INFO - Adding to hash registry
2025-11-15 20:40:19,870 - INFO - Registry updated: now 1 entry
2025-11-15 20:40:19,871 - INFO - Ready for detection matching
```

### Detection with Match (Surveillance Camera)
```
2025-11-15 20:40:25,234 - INFO - YOLO detected 2 objects
2025-11-15 20:40:25,245 - INFO - Object 1: person (confidence: 0.92)
2025-11-15 20:40:25,256 - INFO - Object 2: bag (confidence: 0.88)
2025-11-15 20:40:25,267 - INFO - Extracting embeddings for matching...
2025-11-15 20:40:25,278 - INFO - Checking hash registry (1 entries)
2025-11-15 20:40:25,289 - INFO - Similarity: 0.94 (threshold: 0.70)
2025-11-15 20:40:25,290 - INFO - ✓ MATCH FOUND: a7f3b9c2e156
2025-11-15 20:40:25,301 - INFO - Annotating detection with hash_id
```

---

## 6. File Structure After Testing

```
Kodikon/
├── registrations/                    # Created at runtime
│   ├── a7f3b9c2e156/               # Hash ID as directory
│   │   ├── person.jpg              # Person cropped image
│   │   ├── bag.jpg                 # Bag cropped image
│   │   └── record.json             # Full metadata
│   ├── b4e8f2a9d371/
│   │   ├── person.jpg
│   │   ├── bag.jpg
│   │   └── record.json
│   └── ...
├── baggage_linking.py              # Registration logic (NEW)
├── test_registration_implementation.py
├── mesh/
│   ├── mesh_protocol.py            # +broadcast_hash_registration()
│   ├── udp_setup_guide.py          # +wrapper methods
│   └── ...
├── integrated_runtime/
│   ├── integrated_system.py        # +registration mode
│   └── ...
└── ...
```

### record.json Format
```json
{
  "hash_id": "a7f3b9c2e156",
  "timestamp": 1731689415.234,
  "camera_id": "desk-cam",
  "person_image_path": "registrations/a7f3b9c2e156/person.jpg",
  "bag_image_path": "registrations/a7f3b9c2e156/bag.jpg",
  "person_embedding": [0.123, 0.456, ..., 0.789],  # 512 floats
  "bag_embedding": [0.234, 0.567, ..., 0.890],    # 512 floats
  "color_histogram": {
    "hue": [...],
    "saturation": [...],
    "value": [...]
  }
}
```

---

## 7. Troubleshooting Guide

| Issue | Cause | Solution |
|-------|-------|----------|
| No objects detected in frame | Poor lighting / wrong angle | Adjust camera, ensure person+bag visible |
| Registration fails silently | YOLO not detecting | Check detection with 's' key first |
| Hash ID not broadcast | Mesh disconnected | Check peer discovery (heartbeat logs) |
| Remote node not receiving | Network blocked | Verify UDP port (9999) is open |
| Files not saved | Permission denied | Check `registrations/` directory permissions |
| Countdown too fast | System lag | Normal behavior, happens in <1s total |
| Detection not matching | Embedding distance >0.7 | Register with better lighting/angle |
| Memory leak | Long operation | Check logs for error recovery |

---

## 8. Success Criteria

✅ **Registration Flow Complete When:**
1. Overlay displays "REGISTRATION MODE: ON"
2. Person + bag detected in frame
3. SPACE pressed → "FRAME FROZEN" appears
4. Countdown: "3... 2... 1..."
5. Success message: "✓ Registered: {hash_id[:8]}"
6. Files created: registrations/{hash_id}/ with person.jpg, bag.jpg, record.json
7. Mesh logs show: "Broadcasting hash registration"
8. Remote nodes log: "Received hash registration"

✅ **Multi-Node Broadcast Complete When:**
1. Registration successful on node 1
2. Node 2 receives broadcast
3. Node 3 receives broadcast
4. All nodes update hash_registry
5. `get_hash_registry()` returns 1+ entries on all nodes

✅ **Detection Matching Complete When:**
1. Registered person-bag appears on surveillance camera
2. YOLO detects both objects
3. Logs show: "MATCH FOUND: {hash_id}"
4. Bounding box annotated with hash_id
5. Confidence shown alongside

---

## 9. Next Steps

1. **Start Testing**: Run Phase 1 (single-node) first
2. **Verify Basics**: Ensure registration workflow completes
3. **Scale Testing**: Progress to Phase 2 (multi-node)
4. **Validate Broadcasting**: Confirm mesh synchronization
5. **Fine-tune Matching**: Adjust similarity threshold if needed (currently 0.7)
6. **Performance Test**: Measure registration/detection timing
7. **Error Testing**: Run Phase 4 (edge cases)
8. **Document Results**: Record performance metrics

---

## 10. Implementation Summary

| Component | Status | Files | LOC |
|-----------|--------|-------|-----|
| Baggage Linking | ✅ Complete | baggage_linking.py | 478 |
| Mesh Protocol | ✅ Complete | mesh_protocol.py | +65 |
| UDP Setup Guide | ✅ Complete | udp_setup_guide.py | +25 |
| Integrated System | ✅ Complete | integrated_system.py | +120 |
| Testing | ✅ Complete | test_registration_implementation.py | 250 |
| **TOTAL** | **✅ COMPLETE** | **5 files** | **~940 LOC** |

**All syntax verified ✅**
**All imports verified ✅**
**All methods verified ✅**
**All integrations verified ✅**

---

**Ready for end-to-end testing!** 🎉
