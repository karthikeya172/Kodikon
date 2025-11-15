# Baggage Ownership & Transfer Mitigation Design

**Hackathon Challenge**: Mitigate false baggage-mismatch alerts in dense zones (check-in, stores, transfers) using on-device mesh networking, knowledge-graph ownership tracking, and multi-stage alert verification.

---

## 1. DESIGN SUMMARY

**Approach**: Dense-zone & transfer mitigation via temporally-windowed ownership events stored in knowledge-graph, scored across embedding similarity, proximity, color histogram, camera role, and contextual signals.

Each node maintains an **OwnershipEvent ledger** tracking person-bag registration chains with timestamps and location context. Ownership transfers are **explicitly marked** (e.g., check-in drop-off, hand-off detected at checkout).

**Mesh-consensus** collects peer votes (alive nodes only) + weighted confidence fusion. **Multi-stage alert verification**: pending → N confirmations over M seconds, cross-camera corroboration, staff/checkout zone whitelisting.

**Hysteresis & backoff**: raise alert only if confidence drops below threshold AND no recent transfer event recorded AND cross-camera disagreement confirmed (avoids false positives).

---

## 2. KNOWLEDGE-GRAPH SCHEMA (JSON-LD)

### Person Node
```
{
  "person_id": "UUID string",
  "embeddings": ["array of ReID vectors (512-dim each)"],
  "first_seen": "ISO8601 timestamp",
  "last_seen": "ISO8601 timestamp",
  "confidence": "float [0,1]"
}
```

### Bag Node
```
{
  "bag_id": "UUID string",
  "hash_id": "16-char hex from SHA256(embedding+color)",
  "class": "enum: backpack|suitcase|handbag|...",
  "color_histogram": "quantized: hue(36)+sat(32)+val(32)+lab(32)=132 bytes",
  "embedding": "float[512]",
  "first_seen": "ISO8601",
  "last_seen": "ISO8601"
}
```

### OwnershipEvent Node
```
{
  "event_id": "UUID",
  "person_id": "string",
  "bag_id": "string",
  "timestamp": "ISO8601",
  "event_type": "REGISTER | HOLD | TRANSFER_OUT | TRANSFER_IN | CLEAR",
  "confidence": "float [0,1]",
  "source_node_id": "string (mesh node ID)",
  "location_signature": "hash of location context",
  "camera_role": "REGISTRATION | SURVEILLANCE | CHECKOUT | TRANSIT",
  "transfer_token": "optional UUID linking transfers",
  "reason": "optional text context"
}
```

### Example: Registration at Check-In
```json
{
  "event_id": "evt_2025_001_reg",
  "person_id": "p_alice_xyz",
  "bag_id": "b_red_backpack_001",
  "timestamp": "2025-11-15T08:30:00Z",
  "event_type": "REGISTER",
  "confidence": 0.98,
  "source_node_id": "device_phone_001",
  "location_signature": "loc_hash_checkin_gate_1",
  "camera_role": "REGISTRATION",
  "transfer_token": null,
  "reason": "Person-bag linked at check-in, confirmed by staff scan"
}
```

### Example: Check-In Transfer (Drop-Off)
```json
{
  "event_id": "evt_2025_002_xfer",
  "from_person_id": "p_alice_xyz",
  "to_person_id": "staff_gate_1",
  "bag_id": "b_red_backpack_001",
  "timestamp": "2025-11-15T08:35:00Z",
  "location": "loc_hash_checkin_gate_1",
  "transfer_type": "DROP_OFF",
  "confidence": 1.0,
  "verified": true,
  "reason": "Owner explicitly transferred bag at check-in counter"
}
```

### Example: Multi-Bag Scenario (HOLD)
```json
{
  "event_id": "evt_2025_003_hold",
  "person_id": "p_alice_xyz",
  "bag_id": "b_blue_suitcase_002",
  "timestamp": "2025-11-15T08:40:00Z",
  "event_type": "HOLD",
  "confidence": 0.87,
  "source_node_id": "device_phone_001",
  "location_signature": "loc_hash_store_zone_2",
  "camera_role": "SURVEILLANCE",
  "transfer_token": "tok_multibag_session_20251115_001",
  "reason": "Alice now carrying 2 bags in store (proximity + embedding + color match)"
}
```

---

## 3. MATCHING & TRANSFER ALGORITHM

### Ownership Score Computation

```
INPUT: 
  - current_person_detection
  - current_bag_detection
  - last_registered_owner
  - ownership_history_windowed (5 min)
  - peer_votes_from_mesh

STEP 1: SCORE COMPUTATION
  embedding_sim = cosine(person_emb_norm, bag_emb_registered_norm) in [0,1]
  proximity_score = 1.0 - min(spatial_distance_pixels / 300, 1.0)
  color_sim = histogram_distance(person_clothing_hist, bag_color_hist) in [0,1]
  
  camera_role_weight = map[
    REGISTRATION: 1.0,
    SURVEILLANCE: 0.8,
    CHECKOUT: 0.9,
    TRANSIT: 0.7
  ]
  
  context_score = is_transfer_zone(location) ? 0.5 : (is_staff_zone ? 0.3 : 1.0)
  time_decay = exp(-elapsed_seconds_since_register / 300)  # 5-min window
  
  WEIGHTED_SCORE = 
    0.35 * embedding_sim +
    0.25 * proximity_score +
    0.20 * color_sim +
    0.15 * camera_role_weight +
    0.05 * context_score
  
  FINAL_SCORE = WEIGHTED_SCORE * time_decay * camera_role_weight

STEP 2: TRANSFER EVENT CHECK (priority over score)
  IF recent_transfer_event_exists(person_id, bag_id, window=30sec):
    UPDATE ownership_event.event_type = TRANSFER_IN or TRANSFER_OUT
    IF transfer_type == DROP_OFF:
      CLEAR old_owner from ledger
    SUPPRESS alerts for ALERT_SUPPRESSION_WINDOW = 10 seconds
    RETURN TRANSFER (skip score comparison)

STEP 3: OWNERSHIP HYSTERESIS (main decision)
  IF FINAL_SCORE > 0.75 (THRESHOLD_MAINTAIN):
    MAINTAIN current ownership
    last_confident_score = FINAL_SCORE
    RETURN MAINTAIN
  
  ELSE IF FINAL_SCORE < 0.40 (THRESHOLD_CLEAR):
    CLEAR ownership; person != bag owner
    IF last_confident_score WAS > 0.75 AND (drop >= 0.25):
      FLAG_ALERT (potential mismatch)
    RETURN CLEAR
  
  ELSE (0.40 <= FINAL_SCORE <= 0.75, hysteresis zone):
    NO_ACTION; wait for more evidence
    RETURN UNCERTAIN

STEP 4: MESH CONSENSUS (async, non-blocking)
  BROADCAST OwnershipVote { person_id, bag_id, score, node_id, timestamp }
  COLLECT votes from alive_peers = mesh.discover_peers() with 2-sec timeout
  
  peer_scores = [extract scores from votes]
  consensus_score = mean(peer_scores) if len(peer_scores) > 0 else FINAL_SCORE
  consensus_confidence = stdev(peer_scores)
  
  IF consensus_confidence < 0.15 (tight agreement):
    USE consensus_score for final decision
  ELSE:
    USE FINAL_SCORE (peers disagree, trust local)

STEP 5: OUTPUT
  RETURN {
    decision: MAINTAIN | TRANSFER | CLEAR,
    confidence: final_score,
    reason: string
  }
```

### Transfer Detection
```python
def detect_transfer(person_id, bag_id, location_context, recent_events_window_sec=30):
    recent_events = kg_store.query_recent_events(bag_id, window_sec)
    for evt in recent_events:
        if evt.event_type in [TRANSFER_OUT, TRANSFER_IN, HAND_OFF]:
            if abs(evt.timestamp - now) < recent_events_window_sec:
                return True, evt  # Transfer detected
    return False, None
```

---

## 4. ALERT SUPPRESSION & CONFIRMATION

### Multi-Stage Verifier Logic

```
STAGE 1: WHITELIST CHECK
  IF location.zone_type in [CHECKOUT, STAFF_AREA, GATE]:
    SUPPRESS alert (expected transfer zone)
  IF current_person in staff_registry:
    SUPPRESS alert (staff member)

STAGE 2: PENDING ALERT TRACKING
  IF bag_id not in pending_alerts:
    pending_alerts[bag_id] = {
      detections: [(camera_id, prev_owner, current_person, score, timestamp)],
      first_time: now,
      confirmations: 1
    }
    log: pending alert 1/M confirmations
    RETURN False (not yet confirmed)

STAGE 3: CONFIRMATION WINDOW (M=5 seconds, N=2 confirmations)
  entry = pending_alerts[bag_id]
  elapsed = now - entry.first_time
  
  IF elapsed > CONFIRMATION_WINDOW_SEC (5 sec):
    IF entry.confirmations >= MIN_CONFIRMATIONS (2):
      ESCALATE_ALERT(entry)
    ELSE:
      DELETE pending_alerts[bag_id]
    RETURN result

STAGE 4: CROSS-CAMERA CORROBORATION
  cameras = set([d[0] for d in entry.detections])
  
  IF len(cameras) < MIN_CAMERA_AGREEMENT (2):
    entry.detections.append((camera_id, prev_owner, current_person, score, now))
    entry.confirmations += 1
    log: alert confirmation N/M
    
    IF entry.confirmations >= MIN_CONFIRMATIONS:
      ESCALATE_ALERT(entry)
      RETURN True
  
  RETURN False

STAGE 5: ESCALATION WITH BACKOFF
  IF bag_id in confirmed_alerts:
    last_escalation = confirmed_alerts[bag_id].last_escalation_time
    IF now - last_escalation < BACKOFF_WINDOW_SEC (30 sec):
      log: alert backoff, already escalated recently
      RETURN False
  
  LOCAL RETRY LOOP (attempt 1-2):
    mesh_consensus = collect_peer_votes(bag_id, votes_window_sec=5)
    
    IF consensus_score < ESCALATION_THRESHOLD (0.5):
      BROADCAST alert to mesh:
        {
          type: ownership_mismatch,
          bag_id, expected_owner, detected_person, confidence,
          detections: [(cam, owner, person, score), ...]
        }
      confirmed_alerts[bag_id] = {escalated_at: now}
      notify_staff(message)
      RETURN True (escalated)
    
    ELSE:
      sleep(0.5 sec)

THRESHOLDS:
  MIN_CONFIRMATIONS = 2
  CONFIRMATION_WINDOW_SEC = 5
  MIN_CAMERA_AGREEMENT = 2
  ALERT_SUPPRESSION_WINDOW_SEC = 10
  BACKOFF_WINDOW_SEC = 30
  LOCAL_RETRY_ATTEMPTS = 2
  LOCAL_RETRY_DELAY_SEC = 0.5
  ESCALATION_THRESHOLD = 0.5
```

---

## 5. DATA MODEL CHANGES

### OwnershipEvent Dataclass
```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class OwnershipEvent:
    event_id: str
    person_id: str
    bag_id: str
    timestamp: float
    event_type: str  # 'REGISTER', 'HOLD', 'TRANSFER_OUT', 'TRANSFER_IN', 'CLEAR'
    confidence: float
    source_node_id: str
    location_signature: str
    camera_role: str  # 'REGISTRATION', 'SURVEILLANCE', 'CHECKOUT', 'TRANSIT'
    transfer_token: Optional[str] = None
    reason: str = ''
    
    def to_dict(self):
        return {
            'event_id': self.event_id,
            'person_id': self.person_id,
            'bag_id': self.bag_id,
            'timestamp': self.timestamp,
            'event_type': self.event_type,
            'confidence': self.confidence,
            'source_node_id': self.source_node_id,
            'location_signature': self.location_signature,
            'camera_role': self.camera_role,
            'transfer_token': self.transfer_token,
            'reason': self.reason
        }
```

### HashRegistryEntry Extensions
**Existing Fields**: hash_value, node_id, timestamp, data_type, embedding, metadata

**New Fields to Add**:
- `ownership_history: List[OwnershipEvent]` (max 10, FIFO eviction)
- `transfer_token: Optional[str]` (UUID linking transfers)
- `confidence: float` (latest score, [0,1])
- `last_update: float` (unix timestamp)
- `source_node_id: str` (initiating mesh node)
- `location_signature: str` (location hash)

### KG Store Query Methods
```python
add_ownership_event(event: OwnershipEvent) -> bool
get_ownership_history(bag_id, window_sec=300) -> List[OwnershipEvent]
get_current_owner(bag_id) -> Optional[str]
get_person_bags(person_id, window_sec=300) -> List[str]
add_transfer_event(event: TransferEvent) -> bool
query_recent_events(bag_id, window_sec) -> List[OwnershipEvent]
```

---

## 6. MESH MESSAGE EXTENSIONS

### TRANSFER_EVENT (type=9)
```json
{
  "message_type": 9,
  "source_node_id": "device_phone_001",
  "timestamp": 1731658800.5,
  "sequence_number": 42,
  "payload": {
    "event_id": "xfer_001",
    "from_person_id": "p_alice",
    "to_person_id": "staff_1",
    "bag_id": "b_red_001",
    "timestamp": 1731658800.0,
    "location": "loc_gate_1",
    "transfer_type": "DROP_OFF",
    "confidence": 1.0,
    "verified": true,
    "source_node_id": "device_phone_001",
    "reason": "Check-in counter"
  }
}
```
**Max Size**: 400 bytes

### OWNERSHIP_VOTE (type=10)
```json
{
  "message_type": 10,
  "source_node_id": "device_laptop_003",
  "timestamp": 1731658805.2,
  "sequence_number": 127,
  "payload": {
    "person_id": "p_alice",
    "bag_id": "b_red_001",
    "score": 0.82,
    "node_id": "device_laptop_003",
    "timestamp": 1731658805.0,
    "embedding_similarity": 0.85,
    "proximity_score": 0.90,
    "color_similarity": 0.72,
    "camera_role": "SURVEILLANCE",
    "location_signature": "loc_store_2"
  }
}
```
**Max Size**: 250 bytes

### LOCATION_SIGNATURE (type=11)
```json
{
  "message_type": 11,
  "source_node_id": "device_phone_001",
  "timestamp": 1731658810.0,
  "sequence_number": 200,
  "payload": {
    "location_id": "loc_gate_1",
    "zone_type": "GATE",
    "is_staff_zone": false,
    "transfer_likely": true,
    "timestamp": 1731658810.0,
    "node_id": "device_phone_001"
  }
}
```
**Max Size**: 150 bytes

---

## 7. EMBEDDING & FUSION TIPS

### Step-by-Step Fusion (On-Device, Lightweight)

**1. Embedding Normalization**
```
L2_norm = sqrt(sum(e_i^2 for all dims))
normalized = e / (L2_norm + 1e-6)
Result: unit vector in [-1, 1], magnitude = 1
```

**2. Quantization for UDP**
```
Store as int8 or float16 instead of float32
float16(512 dims) = 1024 bytes vs float32 = 2048 bytes (50% savings)
Formula: quantized_val = round((normalized_val + 1.0) * 127.5)
Loss: ~1-2 percent, acceptable for scoring
```

**3. Compression (zlib + base64)**
```python
import zlib, base64, numpy as np

emb_bytes = np.float16(embedding).tobytes()
compressed = zlib.compress(emb_bytes, level=6)  # 60-70% smaller
encoded = base64.b64encode(compressed).decode('utf-8')
# Result: ~700 bytes for 512-dim float16 embedding
```

**4. Cosine Similarity**
```
emb1_norm = emb1 / norm(emb1)
emb2_norm = emb2 / norm(emb2)
similarity = dot(emb1_norm, emb2_norm)
Range: [0, 1] for normalized embeddings (single dot product, fast)
```

**5. Color Histogram Fusion**
```
Combine HSV + LAB: [h_hist(36), s_hist(32), v_hist(32), lab_hist(32)] = 132 values
Normalize each to [0, 1]
bhattacharyya_dist = -ln(sum(sqrt(h1_i * h2_i)))
color_similarity = exp(-bhattacharyya_dist)  # [0, 1]
```

**6. Weighted Score Fusion**
```
score = 0.35*embedding_sim + 0.25*proximity_score + 0.20*color_sim + 0.15*role_weight
All inputs normalized to [0, 1]
Result: single float in [0, 1] for decision threshold
```

**7. Time Decay (Exponential)**
```
elapsed_sec = current_time - registration_time
time_decay = exp(-elapsed_sec / TIME_WINDOW_SECS)  # e.g., 300 sec = 5 min

At 2.5 min: decay ≈ 0.71
At 5 min: decay ≈ 0.37
At 10 min: decay ≈ 0.14
```

### Compression Example (Python)
```python
import zlib, base64, numpy as np

def compress_embedding(embedding: np.ndarray) -> str:
    """Compress 512-dim float32 to ~700-byte base64 string"""
    # Convert to float16 (half precision)
    emb_f16 = np.float16(embedding)
    emb_bytes = emb_f16.tobytes()  # 1024 bytes
    # Compress with zlib
    compressed = zlib.compress(emb_bytes, level=6)  # ~60-70% smaller
    # Encode to base64 for JSON serialization
    encoded = base64.b64encode(compressed).decode('utf-8')
    return encoded  # ~1400 chars, fits in UDP message

def decompress_embedding(encoded: str) -> np.ndarray:
    """Decompress base64-encoded embedding back to float32"""
    compressed = base64.b64decode(encoded)
    emb_bytes = zlib.decompress(compressed)
    emb_f16 = np.frombuffer(emb_bytes, dtype=np.float16)
    emb_f32 = emb_f16.astype(np.float32)  # Back to float32
    return emb_f32
```

---

## 8. IMPLEMENTATION CHECKLIST (Prioritized)

| Priority | Task | Target File(s) | Function/Class to Modify |
|----------|------|-----------------|--------------------------|
| 1 | Add OwnershipEvent & TransferEvent dataclasses | `vision/baggage_linking.py` | Add after ObjectClass enum; include all 10 fields |
| 2 | Extend HashRegistryEntry with ownership fields | `mesh/mesh_protocol.py` | Modify HashRegistryEntry dataclass (~line 103); add 6 new fields |
| 3 | Create kg_store.py with ownership graph | `knowledge_graph/kg_store.py` | New file; KGStore class with 6 query methods; dict + JSON persistence |
| 4 | Extend MessageType enum (9,10,11) | `mesh/mesh_protocol.py` | Add TRANSFER_EVENT, OWNERSHIP_VOTE, LOCATION_SIGNATURE to IntEnum; add handlers in `_handle_message_type()` |
| 5 | Implement OwnershipMatcher class | `vision/baggage_linking.py` | New class; `match(person_det, bag_det, ownership_ledger, mesh, location_context)` method |
| 6 | Add mesh consensus voting | `mesh/mesh_protocol.py` | New methods: `broadcast_ownership_vote()`, `collect_peer_votes(bag_id, window_sec=5)` |
| 7 | Implement AlertVerifier class | `vision/baggage_linking.py` | New AlertVerifier class; `raise_alert()`, `escalate_alert()` with multi-stage logic |
| 8 | Integrate OwnershipMatcher & AlertVerifier | `vision/baggage_linking.py` | Modify `BaggageLinking.process_frame()`; call ownership_matcher on each person-bag pair; log to kg_store |
| 9 | Add mesh broadcast methods | `mesh/mesh_protocol.py` | New: `broadcast_transfer_event()`, `broadcast_ownership_vote()`; validate payload < 2KB |
| 10 | Integrate kg_store into integrated_system.py | `integrated_runtime/integrated_system.py` | Initialize KGStore in `__init__`; pass to BaggageLinking; update ownership after each frame |
| 11 | Add embedding compression utilities | `utils/embedding_utils.py` | New file; `compress_embedding()`, `decompress_embedding()` with float16 + zlib + base64 |
| 12 | Write unit & integration tests | `tests/test_ownership_transfer.py` | New file; test ownership score, transfer events, alert verification, mesh consensus |

---

## 9. TESTS & SIMULATION SNIPPETS

### Unit Test: Ownership Score Computation
```python
def test_ownership_score_computation():
    person = Detection(
        class_name=ObjectClass.PERSON,
        bbox=BoundingBox(100, 100, 200, 300),
        confidence=0.95,
        embedding=np.array([0.5] * 512)
    )
    bag = Detection(
        class_name=ObjectClass.BACKPACK,
        bbox=BoundingBox(150, 250, 180, 300),  # 30px distance
        confidence=0.9,
        embedding=np.array([0.51] * 512)  # Very similar
    )
    
    matcher = OwnershipMatcher()
    ownership_history = {'b_red_backpack_001': 'p_alice_xyz'}
    result = matcher.match(person, bag, ownership_history, location_context={'zone': 'SURVEILLANCE'})
    
    assert result['decision'] == 'MAINTAIN'
    assert result['confidence'] > 0.75
```

### Unit Test: Transfer Event Suppresses Alert
```python
def test_transfer_suppresses_alert():
    person = Detection(...)
    bag = Detection(...)
    
    transfer = OwnershipEvent(
        event_id='evt_002',
        person_id='p_alice',
        bag_id='b_red_001',
        timestamp=time.time() - 5,  # 5 seconds ago
        event_type='TRANSFER_OUT',
        confidence=1.0,
        source_node_id='device_1',
        location_signature='loc_gate_1',
        camera_role='REGISTRATION'
    )
    
    result = matcher.match_with_transfer_check(person, bag, 
                                               {'b_red_001': 'p_alice'}, 
                                               transfer)
    
    assert result['decision'] == 'TRANSFER'
```

### Simulation: Multi-Bag Check-In Scenario
```python
def simulate_multibag_checkin_scenario():
    """Simulate Alice with 1 bag -> check-in -> staff takes -> Alice picks new bag"""
    
    kg_store = KGStore()
    baggage_linking = BaggageLinking()
    mesh = MeshProtocol(node_id='device_001', port=9999)
    mesh.start()
    
    print('[FRAME 1] Alice arrives with red backpack at check-in')
    # Register ownership
    evt_register = OwnershipEvent(
        event_id='evt_001',
        person_id='alice_001',
        bag_id='red_backpack_1',
        timestamp=time.time(),
        event_type='REGISTER',
        confidence=0.98,
        source_node_id='device_001',
        location_signature='loc_checkin_gate',
        camera_role='REGISTRATION'
    )
    kg_store.add_ownership_event(evt_register)
    print('  ✓ Registered: alice owns red_backpack_1')
    
    print('\n[FRAME 2] Alice transfers bag to staff')
    evt_transfer_out = OwnershipEvent(
        event_id='evt_002',
        person_id='alice_001',
        bag_id='red_backpack_1',
        timestamp=time.time(),
        event_type='TRANSFER_OUT',
        confidence=1.0,
        source_node_id='device_001',
        location_signature='loc_checkin_gate',
        camera_role='REGISTRATION',
        transfer_token='xfer_tok_001'
    )
    kg_store.add_ownership_event(evt_transfer_out)
    print('  ✓ Alice dropped bag: TRANSFER_OUT event recorded')
    
    print('\n[FRAME 3] Alice picks up blue suitcase in store')
    ownership_history = kg_store.get_ownership_history('blue_suitcase_2', window_sec=300)
    if not ownership_history:
        evt_new_owner = OwnershipEvent(
            event_id='evt_003',
            person_id='alice_001',
            bag_id='blue_suitcase_2',
            timestamp=time.time(),
            event_type='HOLD',
            confidence=0.87,
            source_node_id='device_001',
            location_signature='loc_store_zone_2',
            camera_role='SURVEILLANCE',
            transfer_token='xfer_tok_001'
        )
        kg_store.add_ownership_event(evt_new_owner)
        print('  ✓ Multi-bag detected: Alice now carrying blue_suitcase_2 (confidence 0.87)')
    
    print('\n[FRAME 4] Surveillance sees bag without registered owner (ALERT TEST)')
    current_owner = kg_store.get_current_owner('blue_suitcase_2')
    if current_owner != 'unknown_002':
        alert = {
            'type': 'ownership_mismatch',
            'bag_id': 'blue_suitcase_2',
            'expected_owner': current_owner,
            'detected_person': 'unknown_002',
            'confidence': 0.15,
            'location': 'loc_store_zone_2'
        }
        mesh.broadcast_alert(alert, priority='high')
        print(f'  ✓ ALERT: {current_owner} expected, but unknown_002 detected')
    
    mesh.stop()
    print('\n[SIMULATION] Complete')
```

---

## Summary: 8-Hour Hackathon Roadmap

1. **Data Model** (1 hour): Add OwnershipEvent, TransferEvent, extend HashRegistryEntry
2. **Knowledge Graph** (1 hour): Create kg_store.py with persistence
3. **Mesh Extensions** (0.5 hours): Add message types 9, 10, 11; handlers
4. **Ownership Matching** (1.5 hours): OwnershipMatcher class with scoring & hysteresis
5. **Alert Verification** (1 hour): AlertVerifier with multi-stage logic
6. **Integration** (1.5 hours): Plug into BaggageLinking.process_frame() + integrated_system.py
7. **Compression & Utils** (0.5 hours): Embedding compression utilities
8. **Testing & Validation** (1 hour): Unit tests + simulation scenarios

**Total: ~8 hours**

All code changes fit within existing module architecture; no cloud dependency; UDP mesh-only; on-device JSON/JSON-LD storage.
