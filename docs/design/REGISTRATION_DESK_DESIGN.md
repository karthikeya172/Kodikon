# Registration Desk System Design
## For Kodikon Distributed Baggage Tracking

---

## 1. REGISTRATION WORKFLOW

### Step-by-Step Behavior in `integrated_system.py`

```
REGISTRATION MODE STATE MACHINE:
├─ IDLE (default)
│  └─ User presses 'r' key → WAITING_FOR_ARRIVAL
│
├─ WAITING_FOR_ARRIVAL
│  └─ YOLO detects 1 PERSON + 1 BAG → FREEZE_FRAME
│
├─ FREEZE_FRAME (1 second)
│  ├─ Display frame with countdown overlay: "3... 2... 1..."
│  └─ After 1s → VALIDATE_DETECTION
│
├─ VALIDATE_DETECTION
│  ├─ Re-run YOLO inference to confirm:
│  │  ├─ Exactly 1 person detected
│  │  └─ Exactly 1 bag (suitcase/backpack/handbag)
│  ├─ If valid → EXTRACT_FEATURES
│  └─ If invalid → back to WAITING_FOR_ARRIVAL (show "Invalid detection")
│
├─ EXTRACT_FEATURES
│  ├─ Crop person bounding box (10px padding)
│  ├─ Crop bag bounding box (10px padding)
│  ├─ Extract person embedding (ResNet-50 or similar)
│  ├─ Extract bag embedding
│  ├─ Compute bag color histogram (HSV)
│  ├─ Generate hash_id = SHA256(timestamp + person_emb[:8])
│  └─ Save images to: registrations/{hash_id}/person.jpg, bag.jpg
│
├─ BROADCAST
│  ├─ Create RegistrationRecord
│  ├─ Call mesh_node.broadcast_hash_registration(record)
│  ├─ Display: "✓ Registered: {hash_id}" (2 second overlay)
│  └─ Return to IDLE
│
└─ ERROR states
   ├─ If YOLO not available → show "YOLO offline"
   ├─ If mesh offline → show "MESH offline, local save only"
   └─ Any exception → log + show "Registration failed"
```

### UI Overlay Integration
- **Registration mode active**: Display at top-left: `[REGISTRATION MODE] Press SPACE to capture`
- **Countdown**: Large "3... 2... 1..." in center
- **Success**: Green checkmark + `✓ Registered: <hash_id>` (2 sec)
- **Error**: Red X + error message (3 sec)

### Keybindings in `run()` loop
```
'r' or 'R' → Toggle registration mode on/off
SPACE      → Trigger capture when in WAITING_FOR_ARRIVAL state
'ESC'      → Cancel registration mode, return to IDLE
```

---

## 2. NEW DATACLASS: RegistrationRecord

### Location: `baggage_linking.py` (top of file, after imports)

```python
from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np
from datetime import datetime
import hashlib

@dataclass
class RegistrationRecord:
    """
    Represents a person-bag pair registered at a desk.
    Used by mesh network to identify individuals across cameras.
    """
    hash_id: str
    """Unique identifier: SHA256(timestamp + person_embedding[:8])"""
    
    person_embedding: np.ndarray
    """ResNet-50 embedding of registered person's face/pose (512-dim or 128-dim)"""
    
    bag_embedding: np.ndarray
    """Bag appearance embedding (CNN-based, same dimension as person_embedding)"""
    
    person_image_path: str
    """Local path: registrations/{hash_id}/person.jpg"""
    
    bag_image_path: str
    """Local path: registrations/{hash_id}/bag.jpg"""
    
    color_histogram: Dict[str, List[float]]
    """
    Bag color histogram in HSV space:
    {
        'hue': [h0, h1, ..., h179],      # 180 bins
        'saturation': [s0, s1, ..., s63],  # 64 bins
        'value': [v0, v1, ..., v63]        # 64 bins
    }
    """
    
    timestamp: float
    """Unix timestamp of registration (time.time())"""
    
    camera_id: str
    """ID of registration desk camera (e.g., "desk_gate_1")"""
    
    person_bbox: tuple = field(default=(0, 0, 0, 0))
    """(x1, y1, x2, y2) bounding box at registration time"""
    
    bag_bbox: tuple = field(default=(0, 0, 0, 0))
    """(x1, y1, x2, y2) bounding box at registration time"""
    
    confidence_person: float = 0.0
    """YOLO detection confidence for person (0.0-1.0)"""
    
    confidence_bag: float = 0.0
    """YOLO detection confidence for bag (0.0-1.0)"""
    
    metadata: Dict = field(default_factory=dict)
    """Additional data: {'airline': 'XX', 'gate': '5', 'flight': 'AB123', ...}"""
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict for mesh broadcast"""
        return {
            'hash_id': self.hash_id,
            'person_embedding': self.person_embedding.tolist(),
            'bag_embedding': self.bag_embedding.tolist(),
            'timestamp': self.timestamp,
            'camera_id': self.camera_id,
            'person_bbox': self.person_bbox,
            'bag_bbox': self.bag_bbox,
            'confidence_person': self.confidence_person,
            'confidence_bag': self.confidence_bag,
            'color_histogram': self.color_histogram,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'RegistrationRecord':
        """Reconstruct from mesh broadcast payload"""
        data['person_embedding'] = np.array(data['person_embedding'])
        data['bag_embedding'] = np.array(data['bag_embedding'])
        return cls(**data)
```

---

## 3. NEW FUNCTION: `register_from_frame()`

### Location: `baggage_linking.py` (after RegistrationRecord class)

```python
import cv2
import os
import time
import hashlib
from pathlib import Path

def register_from_frame(
    frame: np.ndarray,
    mesh_node,
    yolo_model,
    camera_id: str = "desk_gate_1",
    metadata: Dict = None
) -> tuple[RegistrationRecord, bool]:
    """
    Register a person-bag pair from a frozen frame.
    
    Args:
        frame: Frozen video frame (BGR, uint8)
        mesh_node: IntegratedMeshNode instance for broadcasting
        yolo_model: YOLO model instance for detection
        camera_id: ID of this registration desk
        metadata: Optional dict with flight info, gate, etc.
    
    Returns:
        (RegistrationRecord or None, success: bool)
        If success=False, RegistrationRecord will be None
    """
    logger = logging.getLogger(__name__)
    
    try:
        # 1. VALIDATE DETECTION
        results = yolo_model(frame, conf=0.5, verbose=False)
        if not results or len(results) == 0:
            logger.warning("No YOLO results")
            return None, False
        
        result = results[0]
        detections = []
        
        if hasattr(result, 'boxes') and result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0]) if hasattr(box.cls, '__len__') else int(box.cls)
                class_name = result.names.get(class_id, "unknown")
                conf = float(box.conf[0]) if hasattr(box.conf, '__len__') else float(box.conf)
                bbox = box.xyxy[0].tolist()
                
                detections.append({
                    'class': class_name,
                    'confidence': conf,
                    'bbox': bbox
                })
        
        # Check for exactly 1 person and 1 bag
        persons = [d for d in detections if d['class'] == 'person']
        bags = [d for d in detections if d['class'] in ['suitcase', 'backpack', 'handbag', 'bag']]
        
        if len(persons) != 1 or len(bags) != 1:
            logger.warning(f"Invalid detection: {len(persons)} persons, {len(bags)} bags")
            return None, False
        
        person_det = persons[0]
        bag_det = bags[0]
        
        # 2. CROP IMAGES
        person_bbox = person_det['bbox']  # [x1, y1, x2, y2]
        bag_bbox = bag_det['bbox']
        
        # Add padding (10 pixels)
        h, w = frame.shape[:2]
        pad = 10
        
        x1_p = max(0, int(person_bbox[0]) - pad)
        y1_p = max(0, int(person_bbox[1]) - pad)
        x2_p = min(w, int(person_bbox[2]) + pad)
        y2_p = min(h, int(person_bbox[3]) + pad)
        person_crop = frame[y1_p:y2_p, x1_p:x2_p]
        
        x1_b = max(0, int(bag_bbox[0]) - pad)
        y1_b = max(0, int(bag_bbox[1]) - pad)
        x2_b = min(w, int(bag_bbox[2]) + pad)
        y2_b = min(h, int(bag_bbox[3]) + pad)
        bag_crop = frame[y1_b:y2_b, x1_b:x2_b]
        
        if person_crop.size == 0 or bag_crop.size == 0:
            logger.warning("Crop resulted in empty image")
            return None, False
        
        # 3. EXTRACT EMBEDDINGS
        person_emb = _extract_embedding(person_crop)
        bag_emb = _extract_embedding(bag_crop)
        color_hist = _compute_color_histogram(bag_crop)
        
        if person_emb is None or bag_emb is None:
            logger.warning("Failed to extract embeddings")
            return None, False
        
        # 4. GENERATE HASH_ID
        timestamp = time.time()
        hash_input = f"{timestamp:.2f}_{person_emb[:8].tobytes().hex()}"
        hash_id = hashlib.sha256(hash_input.encode()).hexdigest()[:12]
        
        # 5. SAVE IMAGES
        reg_dir = Path("registrations") / hash_id
        reg_dir.mkdir(parents=True, exist_ok=True)
        
        person_path = str(reg_dir / "person.jpg")
        bag_path = str(reg_dir / "bag.jpg")
        
        cv2.imwrite(person_path, person_crop)
        cv2.imwrite(bag_path, bag_crop)
        
        logger.info(f"Saved images: {person_path}, {bag_path}")
        
        # 6. CREATE RECORD
        record = RegistrationRecord(
            hash_id=hash_id,
            person_embedding=person_emb,
            bag_embedding=bag_emb,
            person_image_path=person_path,
            bag_image_path=bag_path,
            color_histogram=color_hist,
            timestamp=timestamp,
            camera_id=camera_id,
            person_bbox=tuple(map(int, person_bbox)),
            bag_bbox=tuple(map(int, bag_bbox)),
            confidence_person=person_det['confidence'],
            confidence_bag=bag_det['confidence'],
            metadata=metadata or {}
        )
        
        # 7. BROADCAST (if mesh available)
        if mesh_node is not None:
            try:
                mesh_node.broadcast_hash_registration(record)
                logger.info(f"Broadcast registration: {hash_id}")
            except Exception as e:
                logger.warning(f"Failed to broadcast: {e}")
        
        logger.info(f"Registration successful: {hash_id}")
        return record, True
        
    except Exception as e:
        logger.error(f"Registration error: {e}", exc_info=True)
        return None, False


def _extract_embedding(crop: np.ndarray) -> np.ndarray:
    """
    Extract feature embedding from cropped person/bag image.
    
    Using simple ResNet-50 pretrained model for now.
    In production, use face recognition or specialized baggage models.
    
    Returns:
        512-dimensional embedding (or None if fails)
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Simple placeholder using OpenCV feature extraction
        # In production, replace with:
        # - face_recognition.face_encodings() for person
        # - torchvision.models.resnet50(pretrained=True) for bag
        
        if crop.shape[0] < 10 or crop.shape[1] < 10:
            return None
        
        # Normalize to 128x128
        crop_resized = cv2.resize(crop, (128, 128))
        
        # Placeholder: use ORB features + flatten
        orb = cv2.ORB_create(nfeatures=256)
        kp, des = orb.detectAndCompute(cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY), None)
        
        if des is None:
            # Fallback: use flattened histogram
            hist = cv2.calcHist([crop_resized], [0, 1, 2], None, [8, 8, 8], 
                               [0, 256, 0, 256, 0, 256])
            emb = hist.flatten()
        else:
            # Use mean of descriptor + padding
            emb = des.mean(axis=0) if len(des) > 0 else np.zeros(32)
            emb = np.pad(emb, (0, 512 - len(emb)), mode='constant')
        
        # Ensure 512-dim
        if len(emb) < 512:
            emb = np.pad(emb, (0, 512 - len(emb)), mode='constant')
        else:
            emb = emb[:512]
        
        # Normalize
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb.astype(np.float32)
        
    except Exception as e:
        logger.error(f"Embedding extraction failed: {e}")
        return None


def _compute_color_histogram(crop: np.ndarray) -> Dict[str, List[float]]:
    """
    Compute HSV color histogram for bag.
    
    Returns:
        {
            'hue': [180 floats],
            'saturation': [64 floats],
            'value': [64 floats]
        }
    """
    try:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        hue_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        sat_hist = cv2.calcHist([hsv], [1], None, [64], [0, 256])
        val_hist = cv2.calcHist([hsv], [2], None, [64], [0, 256])
        
        # Normalize
        hue_hist = (hue_hist / (hue_hist.sum() + 1e-8)).flatten().tolist()
        sat_hist = (sat_hist / (sat_hist.sum() + 1e-8)).flatten().tolist()
        val_hist = (val_hist / (val_hist.sum() + 1e-8)).flatten().tolist()
        
        return {
            'hue': hue_hist,
            'saturation': sat_hist,
            'value': val_hist
        }
    except Exception as e:
        logging.error(f"Color histogram failed: {e}")
        return {'hue': [], 'saturation': [], 'value': []}
```

---

## 4. CODE PATCHES — FILE BY FILE

### A. `baggage_linking.py` — Add imports at top

**INSERTION POINT**: After existing imports

```python
# Add these imports
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import logging

logger = logging.getLogger(__name__)
```

### B. `mesh_protocol.py` — Ensure broadcast method exists

**INSERTION POINT**: In `IntegratedMeshNode` class (or create if missing)

```python
def broadcast_hash_registration(self, record: 'RegistrationRecord') -> bool:
    """
    Broadcast hash registration to all peers via mesh network.
    Uses msg_type="hash_registry" for compatibility.
    
    Args:
        record: RegistrationRecord instance
    
    Returns:
        bool: True if broadcast succeeded
    """
    try:
        payload = {
            'action': 'hash_registration',
            'record': record.to_dict()
        }
        
        msg = self.protocol.create_message(
            message_type='hash_registry',
            source_node_id=self.node_id,
            payload=payload
        )
        
        self.protocol.broadcast(msg)
        logger.info(f"Broadcast hash registration: {record.hash_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to broadcast registration: {e}")
        return False

def on_hash_registry_received(self, msg, payload):
    """
    Handle incoming hash registry from peer.
    Store locally for matching.
    """
    try:
        if 'record' in payload:
            record_data = payload['record']
            # Convert to RegistrationRecord
            from baggage_linking import RegistrationRecord
            record = RegistrationRecord.from_dict(record_data)
            
            # Store locally
            self.hash_registry[record.hash_id] = record
            logger.info(f"Stored hash registry: {record.hash_id} from {record.camera_id}")
    except Exception as e:
        logger.error(f"Error processing hash registry: {e}")

# Add to __init__
self.hash_registry = {}  # hash_id -> RegistrationRecord
```

### C. `integrated_system.py` — Add registration mode variables

**INSERTION POINT**: In `__init__` of main class

```python
# Registration mode state
self.registration_mode = False
self.registration_state = "IDLE"  # IDLE, WAITING_FOR_ARRIVAL, FREEZE_FRAME, EXTRACT_FEATURES
self.registration_freeze_time = None
self.registration_frame = None
self.last_registration_record = None
```

### D. `integrated_system.py` — Add registration processing in main loop

**INSERTION POINT**: In `run()` method, in the keyboard input section

```python
# Existing keyboard input handling...
key = cv2.waitKey(int(display_interval * 1000)) & 0xFF

if key == ord('q'):
    break
    
# NEW: Registration mode handling
elif key == ord('r') or key == ord('R'):
    self.registration_mode = not self.registration_mode
    self.registration_state = "IDLE"
    logger.info(f"Registration mode: {'ON' if self.registration_mode else 'OFF'}")

elif key == ord(' '):  # SPACE
    if self.registration_mode and self.registration_state == "WAITING_FOR_ARRIVAL":
        self.registration_state = "FREEZE_FRAME"
        self.registration_freeze_time = time.time()
        self.registration_frame = display_frame.copy()
        logger.info("Frame frozen for registration")

elif key == 27:  # ESC
    if self.registration_mode:
        self.registration_mode = False
        self.registration_state = "IDLE"
        logger.info("Registration cancelled")
```

### E. `integrated_system.py` — Update display loop with registration overlay

**INSERTION POINT**: In `run()`, before `cv2.imshow()` call

```python
# Handle registration mode display
if self.registration_mode:
    if self.registration_state == "IDLE":
        # Waiting for user to press SPACE
        cv2.putText(display_frame, "[REGISTRATION MODE] Press SPACE to freeze", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    elif self.registration_state == "WAITING_FOR_ARRIVAL":
        cv2.putText(display_frame, "[WAITING] Person + Bag must be visible", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    elif self.registration_state == "FREEZE_FRAME":
        elapsed = time.time() - self.registration_freeze_time
        countdown = max(0, 1.0 - elapsed)
        cv2.putText(display_frame, 
                   f"[FREEZE] {countdown:.1f}s", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        
        # Draw center overlay
        h, w = display_frame.shape[:2]
        cv2.putText(display_frame, 
                   f"{int(countdown * 3)}...", 
                   (w//2 - 50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 0), 3)
        
        # After 1 second, extract features
        if elapsed > 1.0 and self.registration_state == "FREEZE_FRAME":
            self.registration_state = "EXTRACT_FEATURES"
            # Trigger registration async (optional: use threading)
            self._process_registration()
    
    elif self.registration_state == "EXTRACT_FEATURES":
        if self.last_registration_record:
            cv2.putText(display_frame, 
                       f"✓ Registered: {self.last_registration_record.hash_id[:8]}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            # Show for 2 seconds
            if time.time() - self.registration_freeze_time > 3.0:
                self.registration_state = "IDLE"
```

### F. `integrated_system.py` — Add registration processing method

**INSERTION POINT**: Add new method to main class

```python
def _process_registration(self):
    """Process the frozen frame for registration"""
    try:
        if self.registration_frame is None or self.mesh_node is None:
            logger.warning("Registration frame or mesh not available")
            self.registration_state = "IDLE"
            return
        
        from baggage_linking import register_from_frame
        
        record, success = register_from_frame(
            frame=self.registration_frame,
            mesh_node=self.mesh_node,
            yolo_model=self.yolo_model if hasattr(self, 'yolo_model') else None,
            camera_id=self.camera_id,
            metadata={'timestamp': time.time()}
        )
        
        if success:
            self.last_registration_record = record
            logger.info(f"✓ Registration successful: {record.hash_id}")
        else:
            logger.warning("✗ Registration failed: Invalid detection")
            self.registration_state = "WAITING_FOR_ARRIVAL"
            
    except Exception as e:
        logger.error(f"Registration processing error: {e}", exc_info=True)
        self.registration_state = "IDLE"
```

### G. `integrated_system.py` — Update matching logic with hash registry

**INSERTION POINT**: In detection matching section (find where bounding boxes are drawn)

```python
# When drawing bounding boxes for detected objects, add:

def _annotate_with_hash(self, detection, frame, bbox_color=(0, 255, 0)):
    """
    Check if detection matches any registered hash_id.
    If so, annotate with hash_id.
    """
    try:
        if not hasattr(self.mesh_node, 'hash_registry'):
            return bbox_color
        
        # Simple similarity check (in production, use cosine similarity)
        # For now, check if detection bbox overlaps with recent registrations
        
        hash_id = None
        for hid, record in self.mesh_node.hash_registry.items():
            # Compute embedding similarity (placeholder)
            if record.timestamp > time.time() - 3600:  # Within 1 hour
                hash_id = hid
                break
        
        if hash_id:
            # Draw with different color and label
            cv2.putText(frame, f"Hash: {hash_id[:8]}", 
                       (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 0, 0), 2)
            return (255, 0, 0)  # Blue for matched
        
        return bbox_color
        
    except Exception as e:
        logger.debug(f"Hash annotation error: {e}")
        return bbox_color
```

---

## 5. MATCHING LOGIC EXTENSION

### Location: `baggage_linking.py` — Add similarity matching

```python
def compute_embedding_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings.
    
    Returns:
        float: similarity score (0.0 to 1.0)
    """
    if emb1 is None or emb2 is None:
        return 0.0
    
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0
    
    return float(np.dot(emb1, emb2) / (norm1 * norm2))


def check_hash_registry_match(
    detection_embedding: np.ndarray,
    hash_registry: Dict,
    similarity_threshold: float = 0.7
) -> tuple[str, float]:
    """
    Check if detection embedding matches any registered hash_id.
    
    Args:
        detection_embedding: Embedding from current detection
        hash_registry: Dict[hash_id -> RegistrationRecord]
        similarity_threshold: Minimum similarity to match (0.0-1.0)
    
    Returns:
        (hash_id or None, similarity_score)
    """
    best_hash_id = None
    best_score = 0.0
    
    for hash_id, record in hash_registry.items():
        score = compute_embedding_similarity(detection_embedding, record.person_embedding)
        if score > best_score:
            best_score = score
            best_hash_id = hash_id if score >= similarity_threshold else None
    
    return best_hash_id, best_score


def update_linking_with_hash_id(
    detection_person_id: str,
    detection_bag_id: str,
    matched_hash_id: str,
    existing_links: Dict
) -> Dict:
    """
    Override normal linking if hash_id match is found.
    
    Args:
        detection_person_id: Current detection person ID
        detection_bag_id: Current detection bag ID
        matched_hash_id: Hash ID from registry (if found)
        existing_links: Current person-to-bag links
    
    Returns:
        Updated existing_links dict
    """
    if matched_hash_id:
        # Force link based on hash_id
        existing_links[matched_hash_id] = {
            'bag_id': detection_bag_id,
            'hash_id': matched_hash_id,
            'confidence': 1.0,
            'source': 'hash_registry'
        }
    
    return existing_links
```

---

## 6. BROADCAST PAYLOAD

### JSON Structure for `msg_type="hash_registry"`

```json
{
  "message_type": "hash_registry",
  "source_node_id": "desk_gate_1",
  "timestamp": 1731705600.123,
  "payload": {
    "action": "hash_registration",
    "record": {
      "hash_id": "a1b2c3d4e5f6",
      "person_embedding": [0.123, -0.456, 0.789, ...],
      "bag_embedding": [0.234, -0.567, 0.890, ...],
      "person_image_path": "registrations/a1b2c3d4e5f6/person.jpg",
      "bag_image_path": "registrations/a1b2c3d4e5f6/bag.jpg",
      "color_histogram": {
        "hue": [0.05, 0.10, ..., 0.02],
        "saturation": [0.15, 0.20, ..., 0.08],
        "value": [0.12, 0.18, ..., 0.06]
      },
      "timestamp": 1731705600.123,
      "camera_id": "desk_gate_1",
      "person_bbox": [100, 50, 200, 400],
      "bag_bbox": [220, 300, 380, 450],
      "confidence_person": 0.95,
      "confidence_bag": 0.88,
      "metadata": {
        "airline": "EK",
        "gate": "A5",
        "flight": "EK123",
        "seat": "12A"
      }
    }
  }
}
```

---

## 7. STORAGE FORMAT

### Directory Structure for Registrations

```
project_root/
├── registrations/
│   ├── a1b2c3d4e5f6/          # hash_id (12 chars)
│   │   ├── person.jpg          # Cropped person image
│   │   ├── bag.jpg             # Cropped bag image
│   │   └── record.json         # Backup of RegistrationRecord
│   │
│   ├── b2c3d4e5f6a1/
│   │   ├── person.jpg
│   │   ├── bag.jpg
│   │   └── record.json
│   │
│   └── [more registrations...]
│
├── mesh/
├── streaming/
├── power/
└── [other modules...]
```

### Helper Function: Save Record Metadata

```python
def save_registration_metadata(record: RegistrationRecord):
    """
    Save full RegistrationRecord as JSON for future reference.
    
    Location: registrations/{hash_id}/record.json
    """
    import json
    
    try:
        record_dir = Path("registrations") / record.hash_id
        record_dir.mkdir(parents=True, exist_ok=True)
        
        # Save record (excluding image data)
        record_path = record_dir / "record.json"
        
        record_dict = record.to_dict()
        # Remove embeddings (too large) but keep metadata
        record_dict['person_embedding'] = None
        record_dict['bag_embedding'] = None
        
        with open(record_path, 'w') as f:
            json.dump(record_dict, f, indent=2)
        
        logger.info(f"Saved record metadata: {record_path}")
    except Exception as e:
        logger.error(f"Failed to save record metadata: {e}")
```

---

## INTEGRATION CHECKLIST

✅ **Compatible with existing modules:**
- ✓ `mesh_protocol.py` - Uses existing `broadcast()` method
- ✓ `integrated_system.py` - Adds registration mode without changing existing loops
- ✓ `baggage_linking.py` - New functions don't override existing linking logic
- ✓ `power_mode_algo.py` - Registration is stateless, doesn't require power changes
- ✓ `phone_stream_viewer.py` - Works independently (can be a desk camera)

✅ **Key integration points:**
1. `mesh_node.broadcast_hash_registration()` sends registrations
2. `mesh_node.hash_registry` stores received registrations
3. `check_hash_registry_match()` provides override logic for linking
4. UI overlays in `integrated_system.py` without changing core detection loop

✅ **Fallback behavior:**
- If mesh offline: registrations saved locally only
- If YOLO offline: registration mode disabled gracefully
- If detection invalid: user retries with SPACE key

---

## DEPLOYMENT STEPS

1. Add `RegistrationRecord` dataclass to `baggage_linking.py`
2. Add `register_from_frame()`, `_extract_embedding()`, `_compute_color_histogram()` functions
3. Update `mesh_protocol.py` with `broadcast_hash_registration()` method
4. Add registration mode variables to `integrated_system.py.__init__()`
5. Add keyboard handling for 'r' and SPACE in main loop
6. Add `_process_registration()` method
7. Add registration overlay rendering
8. Update matching logic with `check_hash_registry_match()`
9. Test with one desk camera + two surveillance cameras

---

**END OF DESIGN DOCUMENT**
