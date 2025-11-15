# Ownership & Transfer Mitigation: Complete Implementation Guide

**Document**: Hacka athon-ready implementation guide for dense-zone baggage ownership & transfer mitigation.
**Target**: 8 hours to full integration (data model + mesh protocol + alert logic + knowledge graph).

---

## Quick Reference: Files to Create/Modify

### Create (New Files)
1. `knowledge_graph/kg_store.py` — Ownership ledger with queries
2. `utils/embedding_utils.py` — Compression/quantization utilities
3. `tests/test_ownership_transfer.py` — Unit tests + simulation

### Modify (Existing)
1. `vision/baggage_linking.py` — Add OwnershipEvent, OwnershipMatcher, AlertVerifier classes
2. `mesh/mesh_protocol.py` — Extend MessageType, HashRegistryEntry, add consensus voting
3. `integrated_runtime/integrated_system.py` — Integrate KGStore, pass to BaggageLinking

---

## Phase 1: Data Model (1 hour)

### 1.1 Add to `vision/baggage_linking.py` (after ObjectClass enum, ~line 35)

```python
from uuid import uuid4
from typing import Optional

@dataclass
class OwnershipEvent:
    """Track person-bag ownership changes over time"""
    event_id: str = field(default_factory=lambda: str(uuid4()))
    person_id: str = ""
    bag_id: str = ""
    timestamp: float = field(default_factory=time.time)
    event_type: str = ""  # REGISTER, HOLD, TRANSFER_OUT, TRANSFER_IN, CLEAR
    confidence: float = 1.0  # [0, 1]
    source_node_id: str = ""  # Mesh node ID
    location_signature: str = ""  # Hash of location context
    camera_role: str = ""  # REGISTRATION, SURVEILLANCE, CHECKOUT, TRANSIT
    transfer_token: Optional[str] = None  # UUID linking related transfers
    reason: str = ""
    
    def to_dict(self) -> dict:
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
    
    @staticmethod
    def from_dict(d: dict) -> 'OwnershipEvent':
        return OwnershipEvent(**d)


@dataclass
class TransferEvent:
    """Explicit transfer of bag between persons (e.g., check-in drop-off)"""
    event_id: str = field(default_factory=lambda: str(uuid4()))
    from_person_id: str = ""
    to_person_id: str = ""
    bag_id: str = ""
    timestamp: float = field(default_factory=time.time)
    location: str = ""
    transfer_type: str = ""  # HAND_OFF, PICKUP, DROP_OFF, EXCHANGE
    confidence: float = 0.9
    verified: bool = False
    reason: str = ""
    
    def to_dict(self) -> dict:
        return {
            'event_id': self.event_id,
            'from_person_id': self.from_person_id,
            'to_person_id': self.to_person_id,
            'bag_id': self.bag_id,
            'timestamp': self.timestamp,
            'location': self.location,
            'transfer_type': self.transfer_type,
            'confidence': self.confidence,
            'verified': self.verified,
            'reason': self.reason
        }
```

### 1.2 Extend `mesh/mesh_protocol.py` HashRegistryEntry (~line 103)

**Find:**
```python
@dataclass
class HashRegistryEntry:
    """Entry in the distributed hash registry"""
    hash_value: str
    node_id: str
    timestamp: float
    data_type: str  # 'person', 'object', etc.
    embedding: Optional[List[float]] = None
    metadata: dict = field(default_factory=dict)
```

**Replace with:**
```python
@dataclass
class HashRegistryEntry:
    """Entry in the distributed hash registry"""
    hash_value: str
    node_id: str
    timestamp: float
    data_type: str  # 'person', 'object', etc.
    embedding: Optional[List[float]] = None
    metadata: dict = field(default_factory=dict)
    
    # NEW OWNERSHIP TRACKING FIELDS
    ownership_history: List[Dict] = field(default_factory=list)  # Last 10 OwnershipEvents
    transfer_token: Optional[str] = None  # UUID linking transfer events
    confidence: float = 1.0  # Latest ownership confidence [0,1]
    last_update: float = field(default_factory=time.time)  # Unix timestamp
    source_node_id: str = ""  # Initiating mesh node
    location_signature: str = ""  # Location context hash
```

---

## Phase 2: Knowledge Graph Store (1 hour)

### 2.1 Create `knowledge_graph/kg_store.py` (new file)

```python
"""
Knowledge Graph Store - Persistent ownership ledger for baggage tracking.
Stores OwnershipEvents and TransferEvents with time-windowed queries.
"""

import json
import time
import logging
from typing import Dict, List, Optional
from pathlib import Path
from collections import defaultdict
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)


class KGStore:
    """Knowledge Graph store for ownership and transfer events"""
    
    def __init__(self, persist_path: str = "kg_ownership_ledger.json"):
        self.persist_path = Path(persist_path)
        
        # In-memory indexes
        self.ownership_events: Dict[str, List[dict]] = defaultdict(list)  # bag_id -> events
        self.person_bags: Dict[str, List[str]] = defaultdict(list)  # person_id -> bag_ids
        self.transfer_events: Dict[str, dict] = {}  # event_id -> event
        
        self.lock = __import__('threading').Lock()
        
        # Load from disk if exists
        self._load_from_disk()
        
        logger.info(f"KGStore initialized with persist_path={persist_path}")
    
    def add_ownership_event(self, event: dict) -> bool:
        """
        Add ownership event to ledger.
        
        Args:
            event: dict with keys {event_id, person_id, bag_id, timestamp, event_type, ...}
        
        Returns:
            True if added, False if error
        """
        try:
            with self.lock:
                bag_id = event.get('bag_id')
                person_id = event.get('person_id')
                
                # Add to bag's event history
                self.ownership_events[bag_id].append(event)
                
                # Keep max 10 events per bag (FIFO eviction)
                if len(self.ownership_events[bag_id]) > 10:
                    self.ownership_events[bag_id].pop(0)
                
                # Track person-bag relationship
                if person_id and person_id not in self.person_bags[person_id]:
                    self.person_bags[person_id].append(bag_id)
                
            self._persist_to_disk()
            logger.debug(f"Added ownership event: {event.get('event_id')} for {bag_id}")
            return True
        except Exception as e:
            logger.error(f"Error adding ownership event: {e}")
            return False
    
    def get_ownership_history(self, bag_id: str, window_sec: int = 300) -> List[dict]:
        """
        Get ownership events for bag within time window.
        
        Args:
            bag_id: Bag identifier
            window_sec: Time window in seconds (default 5 min)
        
        Returns:
            List of OwnershipEvent dicts, sorted by timestamp (newest first)
        """
        with self.lock:
            events = self.ownership_events.get(bag_id, [])
        
        now = time.time()
        cutoff = now - window_sec
        
        # Filter by time window and sort newest first
        recent = [e for e in events if e.get('timestamp', 0) >= cutoff]
        recent.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        
        return recent
    
    def get_current_owner(self, bag_id: str) -> Optional[str]:
        """
        Get current owner of bag (last REGISTER or HOLD event).
        
        Args:
            bag_id: Bag identifier
        
        Returns:
            person_id or None if no owner
        """
        with self.lock:
            events = self.ownership_events.get(bag_id, [])
        
        # Scan from newest to oldest for ownership event
        for event in reversed(events):
            event_type = event.get('event_type', '')
            
            if event_type in ['REGISTER', 'HOLD']:
                return event.get('person_id')
            elif event_type == 'CLEAR':
                return None
            elif event_type in ['TRANSFER_OUT', 'TRANSFER_IN']:
                # Skip transfers, look for earlier HOLD/REGISTER
                continue
        
        return None
    
    def get_person_bags(self, person_id: str, window_sec: int = 300) -> List[str]:
        """
        Get all bags currently associated with person (within time window).
        
        Args:
            person_id: Person identifier
            window_sec: Time window in seconds
        
        Returns:
            List of bag_ids
        """
        result = []
        
        with self.lock:
            person_bag_ids = self.person_bags.get(person_id, [])
        
        for bag_id in person_bag_ids:
            owner = self.get_current_owner(bag_id)
            if owner == person_id:
                # Verify ownership is recent
                history = self.get_ownership_history(bag_id, window_sec)
                if history and any(e.get('event_type') in ['REGISTER', 'HOLD'] for e in history):
                    result.append(bag_id)
        
        return result
    
    def add_transfer_event(self, event: dict) -> bool:
        """Add explicit transfer event"""
        try:
            with self.lock:
                event_id = event.get('event_id')
                self.transfer_events[event_id] = event
            
            self._persist_to_disk()
            logger.debug(f"Added transfer event: {event_id}")
            return True
        except Exception as e:
            logger.error(f"Error adding transfer event: {e}")
            return False
    
    def query_recent_events(self, bag_id: str, window_sec: int = 30) -> List[dict]:
        """
        Query recent events for bag within short window (e.g., for transfer detection).
        
        Args:
            bag_id: Bag identifier
            window_sec: Time window (default 30 sec for transfer detection)
        
        Returns:
            List of recent events
        """
        return self.get_ownership_history(bag_id, window_sec=window_sec)
    
    def _persist_to_disk(self):
        """Save ownership events to JSON file"""
        try:
            data = {
                'ownership_events': {k: v for k, v in self.ownership_events.items()},
                'transfer_events': self.transfer_events,
                'timestamp': time.time()
            }
            with open(self.persist_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error persisting to disk: {e}")
    
    def _load_from_disk(self):
        """Load ownership events from JSON file"""
        try:
            if self.persist_path.exists():
                with open(self.persist_path, 'r') as f:
                    data = json.load(f)
                
                self.ownership_events = defaultdict(list, data.get('ownership_events', {}))
                self.transfer_events = data.get('transfer_events', {})
                logger.info(f"Loaded {len(self.ownership_events)} bags from disk")
        except Exception as e:
            logger.error(f"Error loading from disk: {e}")
```

---

## Phase 3: Mesh Protocol Extensions (1 hour)

### 3.1 Extend MessageType in `mesh/mesh_protocol.py` (~line 31)

**Find:**
```python
class MessageType(IntEnum):
    """Message types in the mesh network"""
    HEARTBEAT = 1
    PEER_DISCOVERY = 2
    NODE_STATE_SYNC = 3
    SEARCH_QUERY = 4
    ALERT = 5
    HASH_REGISTRY = 6
    ROUTE_BROADCAST = 7
    ACK = 8
```

**Replace with:**
```python
class MessageType(IntEnum):
    """Message types in the mesh network"""
    HEARTBEAT = 1
    PEER_DISCOVERY = 2
    NODE_STATE_SYNC = 3
    SEARCH_QUERY = 4
    ALERT = 5
    HASH_REGISTRY = 6
    ROUTE_BROADCAST = 7
    ACK = 8
    TRANSFER_EVENT = 9
    OWNERSHIP_VOTE = 10
    LOCATION_SIGNATURE = 11
```

### 3.2 Add consensus voting to MeshProtocol class (end of class, before _heartbeat_loop)

```python
def broadcast_ownership_vote(self, vote_data: dict) -> bool:
    """
    Broadcast ownership confidence vote to peers for consensus.
    
    Args:
        vote_data: { person_id, bag_id, score, embedding_similarity, 
                     proximity_score, color_similarity, camera_role, location_signature }
    
    Returns:
        Success status
    """
    message = MeshMessage(
        message_type=MessageType.OWNERSHIP_VOTE,
        source_node_id=self.node_id,
        payload={
            'person_id': vote_data.get('person_id'),
            'bag_id': vote_data.get('bag_id'),
            'score': vote_data.get('score', 0.5),
            'node_id': self.node_id,
            'timestamp': time.time(),
            'embedding_similarity': vote_data.get('embedding_similarity', 0),
            'proximity_score': vote_data.get('proximity_score', 0),
            'color_similarity': vote_data.get('color_similarity', 0),
            'camera_role': vote_data.get('camera_role', ''),
            'location_signature': vote_data.get('location_signature', '')
        }
    )
    return self.send_message(message)

def collect_peer_votes(self, bag_id: str, votes_window_sec: int = 5) -> dict:
    """
    Collect ownership votes from peers (blocking, with timeout).
    
    Args:
        bag_id: Bag to query
        votes_window_sec: Collection window in seconds
    
    Returns:
        { consensus_score: float, scores: [float], stdev: float }
    """
    import numpy as np
    
    # Register temporary handler
    collected_votes = []
    
    def vote_handler(message):
        if message.payload.get('bag_id') == bag_id:
            collected_votes.append(message.payload.get('score', 0.5))
    
    handler_id = len(self.message_handlers[MessageType.OWNERSHIP_VOTE])
    self.message_handlers[MessageType.OWNERSHIP_VOTE].append(vote_handler)
    
    # Wait for votes
    start = time.time()
    while time.time() - start < votes_window_sec:
        time.sleep(0.1)
    
    # Remove handler
    self.message_handlers[MessageType.OWNERSHIP_VOTE].pop(handler_id)
    
    # Compute consensus
    if collected_votes:
        scores = np.array(collected_votes)
        return {
            'consensus_score': float(np.mean(scores)),
            'scores': collected_votes,
            'stdev': float(np.std(scores))
        }
    else:
        return {'consensus_score': 0.5, 'scores': [], 'stdev': 1.0}

def broadcast_transfer_event(self, transfer_event: dict) -> bool:
    """
    Broadcast explicit transfer event to peers.
    
    Args:
        transfer_event: {event_id, from_person_id, to_person_id, bag_id, 
                        timestamp, location, transfer_type, confidence, verified}
    
    Returns:
        Success status
    """
    message = MeshMessage(
        message_type=MessageType.TRANSFER_EVENT,
        source_node_id=self.node_id,
        payload=transfer_event
    )
    return self.send_message(message)
```

### 3.3 Add handler for new message types in `_handle_message_type()` (~line 675)

**Add before the final `except`:**
```python
elif message.message_type == MessageType.TRANSFER_EVENT:
    logger.info(f"Transfer event from {message.source_node_id}: {message.payload}")
    # Handlers registered via register_message_handler()

elif message.message_type == MessageType.OWNERSHIP_VOTE:
    logger.debug(f"Ownership vote from {message.source_node_id}: score={message.payload.get('score')}")
    # Handlers registered via register_message_handler()

elif message.message_type == MessageType.LOCATION_SIGNATURE:
    logger.debug(f"Location signature: {message.payload.get('location_id')}")
    # Handlers registered via register_message_handler()
```

---

## Phase 4: Ownership Matching (1.5 hours)

### 4.1 Add OwnershipMatcher to `vision/baggage_linking.py`

```python
class OwnershipMatcher:
    """Match and maintain person-bag ownership with time-windowed scoring"""
    
    def __init__(self, 
                 threshold_maintain: float = 0.75,
                 threshold_clear: float = 0.40,
                 time_window_sec: float = 300,
                 max_spatial_distance: float = 300):
        """
        Initialize matcher.
        
        Args:
            threshold_maintain: Score threshold to maintain ownership
            threshold_clear: Score threshold to clear ownership
            time_window_sec: Time window for ownership decay (5 min default)
            max_spatial_distance: Max pixels for proximity scoring
        """
        self.logger = logging.getLogger(__name__)
        self.threshold_maintain = threshold_maintain
        self.threshold_clear = threshold_clear
        self.time_window_sec = time_window_sec
        self.max_spatial_distance = max_spatial_distance
        
        # Tracking
        self.last_confident_score: Dict[str, float] = {}  # bag_id -> last high score
        self.transfer_suppression: Dict[str, float] = {}  # bag_id -> suppression_until_time
    
    def match(self, person_det: Detection, bag_det: Detection,
              ownership_history: Dict[str, str], 
              location_context: dict,
              current_time: float = None) -> dict:
        """
        Match person to bag and determine ownership.
        
        Args:
            person_det: Person detection
            bag_det: Bag detection
            ownership_history: {bag_id: person_id} for registered owners
            location_context: {zone: str, is_staff_zone: bool, ...}
            current_time: Current timestamp (for testing)
        
        Returns:
            {decision: str, confidence: float, reason: str}
        """
        if current_time is None:
            current_time = time.time()
        
        bag_id = getattr(bag_det, 'bag_id', 'unknown_bag')
        
        # STEP 1: Get registered owner
        last_owner = ownership_history.get(bag_id)
        if not last_owner:
            return {'decision': 'UNLINKED', 'confidence': 0.0, 'reason': 'No registered owner'}
        
        # STEP 2: Check transfer suppression
        if bag_id in self.transfer_suppression:
            if current_time < self.transfer_suppression[bag_id]:
                return {'decision': 'UNCERTAIN', 'confidence': 0.5, 'reason': 'Alert suppressed (transfer window)'}
            else:
                del self.transfer_suppression[bag_id]
        
        # STEP 3: Compute score
        score = self._compute_score(person_det, bag_det, location_context, current_time, last_owner)
        
        # STEP 4: Hysteresis decision
        bag_key = bag_det.bag_id if hasattr(bag_det, 'bag_id') else bag_id
        
        if score > self.threshold_maintain:
            self.last_confident_score[bag_key] = score
            return {'decision': 'MAINTAIN', 'confidence': score, 'reason': 'High confidence match'}
        
        elif score < self.threshold_clear:
            # Check for significant drop
            prev_score = self.last_confident_score.get(bag_key, score)
            if prev_score > self.threshold_maintain and (prev_score - score) >= 0.25:
                return {'decision': 'CLEAR', 'confidence': score, 'reason': f'Confidence drop: {prev_score:.2f} -> {score:.2f}'}
            return {'decision': 'CLEAR', 'confidence': score, 'reason': 'Low confidence'}
        
        else:
            return {'decision': 'UNCERTAIN', 'confidence': score, 'reason': 'Hysteresis zone, awaiting confirmation'}
    
    def _compute_score(self, person_det: Detection, bag_det: Detection,
                      location_context: dict, current_time: float,
                      last_owner: str) -> float:
        """Compute weighted ownership score"""
        
        # Embedding similarity
        emb_sim = self._embedding_similarity(person_det, bag_det)
        
        # Proximity
        spatial_dist = person_det.bbox.distance_to(bag_det.bbox)
        prox_score = 1.0 - min(spatial_dist / self.max_spatial_distance, 1.0)
        
        # Color similarity
        color_sim = ColorDescriptor.histogram_distance(
            person_det.color_histogram,
            bag_det.color_histogram
        )
        
        # Camera role weight
        camera_role = location_context.get('camera_role', 'SURVEILLANCE')
        role_weight = {'REGISTRATION': 1.0, 'SURVEILLANCE': 0.8, 'CHECKOUT': 0.9, 'TRANSIT': 0.7}.get(camera_role, 0.8)
        
        # Context score
        zone = location_context.get('zone', 'GENERAL')
        context_score = 0.5 if zone in ['CHECKOUT', 'CHECK-IN'] else (0.3 if location_context.get('is_staff_zone') else 1.0)
        
        # Time decay
        registration_time = location_context.get('registration_time', current_time)
        elapsed = current_time - registration_time
        time_decay = np.exp(-elapsed / self.time_window_sec)
        
        # Weighted fusion
        weighted = 0.35 * emb_sim + 0.25 * prox_score + 0.20 * color_sim + 0.15 * role_weight + 0.05 * context_score
        final_score = weighted * time_decay * role_weight
        
        return float(np.clip(final_score, 0, 1))
    
    def _embedding_similarity(self, det1: Detection, det2: Detection) -> float:
        """Cosine similarity between embeddings"""
        try:
            emb1 = det1.embedding / (np.linalg.norm(det1.embedding) + 1e-6)
            emb2 = det2.embedding / (np.linalg.norm(det2.embedding) + 1e-6)
            return float(np.clip(np.dot(emb1, emb2), 0, 1))
        except:
            return 0.5
    
    def suppress_alerts(self, bag_id: str, duration_sec: float = 10):
        """Suppress alerts for bag for specified duration"""
        self.transfer_suppression[bag_id] = time.time() + duration_sec
```

---

## Phase 5: Alert Verification (1 hour)

### 5.1 Add AlertVerifier to `vision/baggage_linking.py`

```python
class AlertVerifier:
    """Multi-stage alert verification with suppression and confirmation logic"""
    
    def __init__(self, min_confirmations: int = 2, 
                 confirmation_window_sec: int = 5,
                 min_camera_agreement: int = 2,
                 backoff_window_sec: int = 30):
        self.logger = logging.getLogger(__name__)
        self.min_confirmations = min_confirmations
        self.confirmation_window_sec = confirmation_window_sec
        self.min_camera_agreement = min_camera_agreement
        self.backoff_window_sec = backoff_window_sec
        
        # Whitelist zones and staff
        self.whitelist_zones = ['CHECKOUT', 'STAFF_AREA', 'GATE']
        self.staff_registry = set()
        
        # Alert tracking
        self.pending_alerts: Dict[str, dict] = {}  # bag_id -> {detections, first_time, confirmations}
        self.confirmed_alerts: Dict[str, dict] = {}  # bag_id -> {escalated_at}
        self.lock = __import__('threading').Lock()
    
    def raise_alert(self, bag_id: str, prev_owner: str, current_person: str,
                   score: float, camera_id: str, location: dict) -> bool:
        """
        Raise and verify alert through multi-stage confirmation.
        
        Returns: True if alert escalated, False otherwise
        """
        # STAGE 1: Whitelist check
        zone = location.get('zone_type', '')
        if zone in self.whitelist_zones:
            self.logger.debug(f"Alert suppressed: bag {bag_id} in whitelist zone {zone}")
            return False
        
        if current_person in self.staff_registry:
            self.logger.debug(f"Alert suppressed: {current_person} is staff")
            return False
        
        # STAGE 2 & 3: Pending alert tracking + confirmation window
        with self.lock:
            if bag_id not in self.pending_alerts:
                # New pending alert
                self.pending_alerts[bag_id] = {
                    'detections': [(camera_id, prev_owner, current_person, score, time.time())],
                    'first_time': time.time(),
                    'confirmations': 1
                }
                self.logger.info(f"Alert pending: {bag_id} (1/{self.min_confirmations} confirmations)")
                return False
            
            entry = self.pending_alerts[bag_id]
            elapsed = time.time() - entry['first_time']
            
            # Check confirmation window
            if elapsed > self.confirmation_window_sec:
                # Window closed
                if entry['confirmations'] >= self.min_confirmations:
                    # Try to escalate
                    result = self._escalate_alert(bag_id, entry)
                    del self.pending_alerts[bag_id]
                    return result
                else:
                    del self.pending_alerts[bag_id]
                    return False
            
            # STAGE 4: Cross-camera corroboration
            cameras = set(d[0] for d in entry['detections'])
            if len(cameras) < self.min_camera_agreement:
                entry['detections'].append((camera_id, prev_owner, current_person, score, time.time()))
                entry['confirmations'] += 1
                self.logger.info(f"Alert confirmation: {bag_id} ({entry['confirmations']}/{self.min_confirmations}, cameras: {cameras})")
                
                if entry['confirmations'] >= self.min_confirmations:
                    result = self._escalate_alert(bag_id, entry)
                    del self.pending_alerts[bag_id]
                    return result
            
            return False
    
    def _escalate_alert(self, bag_id: str, entry: dict) -> bool:
        """Escalate alert with backoff and retry logic"""
        
        # STAGE 5: Backoff check
        if bag_id in self.confirmed_alerts:
            last_escalation = self.confirmed_alerts[bag_id].get('last_escalation_time', 0)
            if time.time() - last_escalation < self.backoff_window_sec:
                self.logger.debug(f"Alert backoff: {bag_id} already escalated recently")
                return False
        
        # Local retry (would integrate with mesh consensus in real implementation)
        self.logger.warning(f"ESCALATING ALERT: {bag_id}")
        self.logger.warning(f"  Detections: {entry['detections']}")
        
        self.confirmed_alerts[bag_id] = {
            'escalated_at': time.time(),
            'last_escalation_time': time.time(),
            'detections': entry['detections']
        }
        
        return True
    
    def add_staff_member(self, person_id: str):
        """Register person as staff (alerts suppressed)"""
        self.staff_registry.add(person_id)
```

---

## Phase 6: Integration (1.5 hours)

### 6.1 Integrate into BaggageLinking.process_frame() in `vision/baggage_linking.py`

Modify the `BaggageLinking.__init__()` to add:

```python
def __init__(self, config: Optional[Dict[str, Any]] = None, kg_store = None):
    # ... existing init code ...
    
    # NEW: Add ownership matching and alert verification
    self.ownership_matcher = OwnershipMatcher()
    self.alert_verifier = AlertVerifier()
    self.kg_store = kg_store  # Knowledge graph store (passed in)
    
    # Dictionary to track current ownership
    self.current_ownership: Dict[str, str] = {}  # bag_id -> person_id
```

Modify `process_frame()` after linking step to add ownership matching:

```python
# Step 5: OWNERSHIP MATCHING (NEW)
if self.kg_store:
    for link in links:
        if link.person_detection and link.bag_detection:
            match_result = self.ownership_matcher.match(
                link.person_detection,
                link.bag_detection,
                self.current_ownership,
                location_context={'camera_role': 'REGISTRATION', 'zone': camera_id}
            )
            
            if match_result['decision'] == 'MAINTAIN':
                # Create ownership event
                from vision.baggage_linking import OwnershipEvent
                evt = {
                    'event_id': f"evt_{frame_id}_{link.bag_id}",
                    'person_id': link.person_id,
                    'bag_id': link.bag_id,
                    'timestamp': time.time(),
                    'event_type': 'HOLD',
                    'confidence': match_result['confidence'],
                    'source_node_id': 'device_local',
                    'location_signature': camera_id,
                    'camera_role': 'REGISTRATION'
                }
                self.kg_store.add_ownership_event(evt)
                self.current_ownership[link.bag_id] = link.person_id
```

Add alert verification step:

```python
# Step 6: ALERT VERIFICATION (NEW)
mismatches_verified = []
for mismatch in mismatches:
    escalated = self.alert_verifier.raise_alert(
        bag_id=mismatch['current_bag'],
        prev_owner=mismatch['expected_bag'],
        current_person=mismatch.get('person_id', 'unknown'),
        score=0.2,  # Low score for mismatch
        camera_id=camera_id,
        location={'zone_type': 'SURVEILLANCE'}
    )
    if escalated:
        mismatches_verified.append(mismatch)

with self.lock:
    self.mismatches.extend(mismatches_verified)
```

### 6.2 Integrate KGStore into `integrated_system.py`

Add to IntegratedSystem.__init__():

```python
from knowledge_graph.kg_store import KGStore

# Initialize knowledge graph store
self.kg_store = KGStore(persist_path=Path(self.workspace_dir) / "kg_ownership.json")

# Pass to baggage linking
self.baggage_linking = BaggageLinking(config=config, kg_store=self.kg_store)
```

---

## Phase 7: Utilities (0.5 hours)

### 7.1 Create `utils/embedding_utils.py`

```python
"""Embedding compression and quantization utilities for UDP transmission"""

import zlib
import base64
import numpy as np
from typing import Optional


def compress_embedding(embedding: np.ndarray) -> str:
    """
    Compress float32 embedding to ~700-byte base64 string.
    
    Args:
        embedding: 512-dim float32 array
    
    Returns:
        Base64-encoded compressed string
    """
    # Convert to float16 (half precision)
    emb_f16 = np.float16(embedding)
    emb_bytes = emb_f16.tobytes()  # 1024 bytes
    
    # Compress with zlib
    compressed = zlib.compress(emb_bytes, level=6)  # ~60-70% smaller
    
    # Encode to base64 for JSON serialization
    encoded = base64.b64encode(compressed).decode('utf-8')
    return encoded


def decompress_embedding(encoded: str) -> np.ndarray:
    """
    Decompress base64-encoded embedding back to float32.
    
    Args:
        encoded: Base64-encoded compressed embedding
    
    Returns:
        512-dim float32 array
    """
    compressed = base64.b64decode(encoded)
    emb_bytes = zlib.decompress(compressed)
    emb_f16 = np.frombuffer(emb_bytes, dtype=np.float16)
    emb_f32 = emb_f16.astype(np.float32)
    return emb_f32


def quantize_embedding(embedding: np.ndarray, bits: int = 8) -> np.ndarray:
    """Quantize embedding to int8 or int16"""
    if bits == 8:
        return ((embedding + 1.0) * 127.5).astype(np.int8)
    elif bits == 16:
        return ((embedding + 1.0) * 32767.5).astype(np.int16)
    return embedding


def dequantize_embedding(quantized: np.ndarray, bits: int = 8) -> np.ndarray:
    """Dequantize back to float32"""
    if bits == 8:
        return (quantized.astype(np.float32) / 127.5) - 1.0
    elif bits == 16:
        return (quantized.astype(np.float32) / 32767.5) - 1.0
    return quantized.astype(np.float32)
```

---

## Phase 8: Testing (1 hour)

### 8.1 Create `tests/test_ownership_transfer.py`

```python
"""Unit and integration tests for ownership and transfer logic"""

import unittest
import time
import uuid
import numpy as np
from collections import namedtuple

# Mock imports (adjust based on actual project structure)
from vision.baggage_linking import (
    Detection, BoundingBox, ObjectClass, ColorHistogram,
    OwnershipEvent, OwnershipMatcher, AlertVerifier
)
from knowledge_graph.kg_store import KGStore


class TestOwnershipMatching(unittest.TestCase):
    
    def setUp(self):
        self.matcher = OwnershipMatcher()
        self.kg_store = KGStore(persist_path=":memory:")  # In-memory for testing
    
    def test_high_confidence_maintain(self):
        """Test maintaining ownership with high similarity"""
        person = Detection(
            class_name=ObjectClass.PERSON,
            bbox=BoundingBox(100, 100, 200, 300),
            confidence=0.95,
            embedding=np.array([0.5] * 512),
            bag_id=None
        )
        person.bag_id = None
        
        bag = Detection(
            class_name=ObjectClass.BACKPACK,
            bbox=BoundingBox(150, 250, 180, 300),  # 30px distance
            confidence=0.9,
            embedding=np.array([0.51] * 512),  # Very similar
            bag_id='b_red_001'
        )
        
        ownership_history = {'b_red_001': 'p_alice'}
        result = self.matcher.match(person, bag, ownership_history, 
                                   location_context={'camera_role': 'SURVEILLANCE'})
        
        self.assertEqual(result['decision'], 'MAINTAIN')
        self.assertGreater(result['confidence'], 0.75)
    
    def test_low_confidence_clear(self):
        """Test clearing ownership with low similarity"""
        person = Detection(
            class_name=ObjectClass.PERSON,
            bbox=BoundingBox(100, 100, 200, 300),
            confidence=0.95,
            embedding=np.array([0.1] * 512),  # Very different
            bag_id=None
        )
        
        bag = Detection(
            class_name=ObjectClass.BACKPACK,
            bbox=BoundingBox(800, 800, 900, 900),  # Far away
            confidence=0.9,
            embedding=np.array([0.9] * 512),
            bag_id='b_red_001'
        )
        
        ownership_history = {'b_red_001': 'p_alice'}
        result = self.matcher.match(person, bag, ownership_history, 
                                   location_context={'camera_role': 'SURVEILLANCE'})
        
        self.assertEqual(result['decision'], 'CLEAR')


class TestAlertVerification(unittest.TestCase):
    
    def setUp(self):
        self.verifier = AlertVerifier()
    
    def test_pending_alert_confirmation(self):
        """Test pending alert requires multiple confirmations"""
        result1 = self.verifier.raise_alert(
            bag_id='b_red_001',
            prev_owner='p_alice',
            current_person='unknown_1',
            score=0.2,
            camera_id='cam_1',
            location={'zone_type': 'SURVEILLANCE'}
        )
        # First detection: pending
        self.assertFalse(result1)
        
        # Second detection from different camera: should escalate
        result2 = self.verifier.raise_alert(
            bag_id='b_red_001',
            prev_owner='p_alice',
            current_person='unknown_1',
            score=0.2,
            camera_id='cam_2',
            location={'zone_type': 'SURVEILLANCE'}
        )
        # Second camera agreement: escalated
        self.assertTrue(result2)


class TestKGStore(unittest.TestCase):
    
    def setUp(self):
        self.kg_store = KGStore(persist_path=":memory:")
    
    def test_ownership_event_storage(self):
        """Test storing and retrieving ownership events"""
        evt = {
            'event_id': 'evt_001',
            'person_id': 'p_alice',
            'bag_id': 'b_red_001',
            'timestamp': time.time(),
            'event_type': 'REGISTER',
            'confidence': 0.98,
            'source_node_id': 'device_1',
            'location_signature': 'loc_gate_1',
            'camera_role': 'REGISTRATION'
        }
        
        self.kg_store.add_ownership_event(evt)
        history = self.kg_store.get_ownership_history('b_red_001')
        
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['event_type'], 'REGISTER')
    
    def test_current_owner_query(self):
        """Test getting current owner"""
        evt = {
            'event_id': 'evt_001',
            'person_id': 'p_alice',
            'bag_id': 'b_red_001',
            'timestamp': time.time(),
            'event_type': 'REGISTER',
            'confidence': 0.98,
            'source_node_id': 'device_1',
            'location_signature': 'loc_gate_1',
            'camera_role': 'REGISTRATION'
        }
        
        self.kg_store.add_ownership_event(evt)
        owner = self.kg_store.get_current_owner('b_red_001')
        
        self.assertEqual(owner, 'p_alice')


if __name__ == '__main__':
    unittest.main()
```

---

## Summary: 8-Hour Implementation Sequence

| Hour | Phase | Deliverable |
|------|-------|-------------|
| 1 | Data Model | Add OwnershipEvent, TransferEvent; extend HashRegistryEntry |
| 1 | KG Store | Create kg_store.py with 6 query methods |
| 1 | Mesh Extensions | Add MessageTypes 9/10/11; consensus voting |
| 1.5 | Ownership Matching | OwnershipMatcher class with scoring & hysteresis |
| 1 | Alert Verification | AlertVerifier with 5-stage logic |
| 1.5 | Integration | Wire into BaggageLinking, IntegratedSystem |
| 0.5 | Utilities | Embedding compression tools |
| 1 | Testing | Unit tests + simulation scenario |

**Total: 8 hours to full implementation and testing.**

All changes fit within existing architecture; no cloud, UDP mesh only, on-device JSON persistence.
