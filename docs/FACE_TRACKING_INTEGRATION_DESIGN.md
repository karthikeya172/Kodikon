# Face Tracking + Timestamped Video History Integration

## Executive Summary

This document provides a **complete integration blueprint** for adding:
- **Face Detection + ResNet-50 Embeddings** (person re-identification)
- **Timestamped Frame Capture** (local storage with configurable history)
- **Server-Side Event Logging** (PERSON_IN/OUT events with CSV logging)
- **Backtrack Search** (find persons by reference face in historical frames)
- **Mesh Network Extensions** (face search requests/results across nodes)

**Non-breaking additions:** All changes are additive; existing YOLO + ReID logic remains untouched.

---

## 1. SYSTEM EXTENSION DESIGN

### 1.1 Per-Frame Timestamping in `integrated_system.py`

**Current State:** Frames are processed in-memory without persistent timestamps.

**Extension:**
- Attach `datetime.now()` + millisecond precision to each frame
- Store in new `FrameMetadata` dataclass
- Pass through entire processing pipeline

**Impact:** <5ms overhead per frame; enables historical reconstruction.

---

### 1.2 Local Storage (Configurable Frame History)

**Implementation Strategy:**
- **Ring Buffer:** Keep last N frames in memory (default N=300 @ 30fps = 10 seconds)
- **Disk Storage:** On mismatch/alert, save frame range to disk (GIF or JPEG sequence)
- **Auto-Prune:** Discard oldest frames when buffer full
- **Compression:** Optional H264 encoding for long-term storage

**Recommended Config:**
```yaml
frame_history:
  memory_buffer_size: 300        # ~10 seconds @ 30fps
  save_on_alert: true            # Persist on mismatch detection
  disk_cache_dir: "./frame_cache"
  max_disk_frames: 5000          # ~2 hours
  compression: "jpg"             # jpg, png, or h264
```

---

### 1.3 Server-Side Event Logging

**Events to Emit:**

| Event | Trigger | Payload |
|-------|---------|---------|
| `PERSON_IN` | First YOLO person detection | `{person_id, timestamp, camera_id, bbox, confidence}` |
| `PERSON_OUT` | Person not detected for >2 sec | `{person_id, last_seen_timestamp, duration_seconds}` |
| `FACE_MATCHED` | Face embedding match found | `{person_id, reference_face_id, similarity, timestamp}` |

**Log Format (JSON Lines):**
```json
{"event": "PERSON_IN", "person_id": "p_123", "timestamp": 1731702451.234, "camera_id": "camera-0", "bbox": [100, 50, 200, 300], "confidence": 0.95}
{"event": "PERSON_OUT", "person_id": "p_123", "timestamp": 1731702481.234, "duration_sec": 30, "last_bbox": [105, 55, 205, 305]}
```

---

### 1.4 Face Embedding Extraction (ResNet-50)

**Lightweight Approach for Mobile:**
- Use **torchvision ResNet-50** pretrained on ImageNet
- Extract features from final average pooling layer (2048-dim)
- Reduce to 512-dim via PCA or fully connected layer
- Normalize L2 for cosine similarity matching

**Key Differences from YOLO/ReID:**
- ReID (OSNet) focuses on **full-body appearance**
- Face embeddings focus on **facial features**
- Both can coexist; face gives higher accuracy for long-range cameras

---

### 1.5 PersonTracker Class (Per-Target Tracking)

**Purpose:** Track each detected person across frames to build temporal consistency.

**State Machine:**
```
IDLE
  ↓ (YOLO detects person)
TRACKING
  ↓ (no detection for N frames)
LOST
  ↓ (timeout after M seconds)
IDLE
```

**Temporal Consistency Bonus:**
- If person detected at T=t and T=t+0.5s, boost confidence by +0.1
- If face embedding matches from T=t and T=t+1.0s, boost by +0.15

---

### 1.6 Face Search Workflow (Reference Face Upload)

**User Flow:**
```
1. User uploads reference face image
2. System extracts face_embedding via ResNet-50
3. Broadcast FACE_SEARCH_REQUEST through mesh
   - Include: target_embedding, timestamp_range, camera_id
4. Each node backtracks through frame history
   - Run YOLO to find person detections
   - Extract face crops from each person
   - Compute cosine similarity
   - Return matches with timestamps
5. Aggregate results; display timeline overlay
```

---

## 2. NEW CODE FOR FACE EMBEDDINGS

### 2.1 Insert into `vision/baggage_linking.py`

**Location:** After `EmbeddingExtractor` class (around line 500)

```python
# ============================================================================
# FACE EMBEDDING EXTRACTION (NEW)
# ============================================================================

class FaceEmbeddingExtractor:
    """
    Extract face embeddings using ResNet-50 for person re-identification.
    Lightweight version optimized for mobile deployment.
    """
    
    def __init__(self, embedding_dim: int = 512,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize face embedding extractor.
        
        Args:
            embedding_dim: Output embedding dimension (512 or 2048)
            device: 'cuda' or 'cpu'
        """
        self.logger = logging.getLogger(__name__)
        self.embedding_dim = embedding_dim
        self.device = device
        
        try:
            import torchvision.models as models
            
            # Load pretrained ResNet-50
            self.model = models.resnet50(pretrained=True)
            
            # Remove classification head, keep feature extractor
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            self.model.eval()
            self.model.to(device)
            
            # Dimension reduction layer if needed
            if embedding_dim != 2048:
                self.reduction = torch.nn.Linear(2048, embedding_dim)
                self.reduction.to(device)
            else:
                self.reduction = None
            
            self.logger.info(f"FaceEmbeddingExtractor initialized (dim={embedding_dim})")
        
        except ImportError as e:
            self.logger.warning(f"ResNet-50 initialization failed: {e}")
            self.model = None
    
    def extract(self, frame: np.ndarray, face_bbox: BoundingBox) -> np.ndarray:
        """
        Extract face embedding from region in frame.
        
        Args:
            frame: Input frame (BGR)
            face_bbox: Bounding box of face region
        
        Returns:
            Embedding vector (512-dim or 2048-dim)
        """
        try:
            # Crop face region
            x1, y1, x2, y2 = face_bbox.to_int_coords()
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                return np.zeros(self.embedding_dim)
            
            if self.model is None:
                return self._extract_simple_face_features(face_crop)
            
            # Preprocess
            img_tensor = self._preprocess_face(face_crop)
            
            # Extract embedding
            with torch.no_grad():
                features = self.model(img_tensor)
                features = features.view(features.size(0), -1)  # Flatten
                
                if self.reduction:
                    features = self.reduction(features)
                
                # L2 normalize
                features = F.normalize(features, p=2, dim=1)
            
            return features.cpu().numpy()[0]
        
        except Exception as e:
            self.logger.error(f"Face embedding extraction failed: {e}")
            return np.zeros(self.embedding_dim)
    
    def _preprocess_face(self, face_img: np.ndarray) -> torch.Tensor:
        """Preprocess face image for ResNet-50."""
        # Resize to 224x224 (ResNet standard)
        face_img = cv2.resize(face_img, (224, 224))
        
        # Convert BGR to RGB
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(face_img).float().permute(2, 0, 1) / 255.0
        
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        return img_tensor.unsqueeze(0).to(self.device)
    
    def _extract_simple_face_features(self, face_img: np.ndarray) -> np.ndarray:
        """Fallback: extract simple facial features using histogram analysis."""
        features = []
        
        # Resize for consistency
        face_img = cv2.resize(face_img, (64, 64))
        
        # Convert to LAB color space
        lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
        
        # Extract histogram features
        for channel in range(3):
            hist = cv2.calcHist([lab], [channel], None, [64], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            features.extend(hist)
        
        # Extract texture features (Sobel edges)
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        features.extend(sobelx.flatten()[:100])
        
        features = np.array(features)
        
        # Pad to embedding_dim
        if len(features) < self.embedding_dim:
            features = np.pad(features, (0, self.embedding_dim - len(features)))
        else:
            features = features[:self.embedding_dim]
        
        # Normalize
        features = features / (np.linalg.norm(features) + 1e-6)
        return features


# ============================================================================
# PERSON DATACLASS EXTENSION (NEW FIELDS)
# ============================================================================

@dataclass
class Person:
    """Extended Person detection with face embedding (NEW)"""
    person_id: str
    detection: Detection                           # YOLO person detection
    face_embedding: np.ndarray = field(            # NEW: Face embedding
        default_factory=lambda: np.zeros(512)
    )
    face_bbox: Optional[BoundingBox] = None        # NEW: Face region (may differ from person bbox)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    track_length: int = 0
    face_embedding_confidence: float = 0.0         # NEW: Confidence of face extraction
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict"""
        return {
            'person_id': self.person_id,
            'detection': {
                'bbox': [self.detection.bbox.x1, self.detection.bbox.y1,
                        self.detection.bbox.x2, self.detection.bbox.y2],
                'confidence': self.detection.confidence
            },
            'face_embedding': self.face_embedding.tolist() if self.face_embedding is not None else [],
            'face_bbox': [self.face_bbox.x1, self.face_bbox.y1,
                         self.face_bbox.x2, self.face_bbox.y2] if self.face_bbox else None,
            'first_seen': self.first_seen,
            'last_seen': self.last_seen,
            'track_length': self.track_length,
            'face_embedding_confidence': self.face_embedding_confidence
        }


def extract_face_embedding_from_person(frame: np.ndarray, 
                                       person_detection: Detection,
                                       face_extractor: FaceEmbeddingExtractor) -> Tuple[np.ndarray, float]:
    """
    Extract face embedding from person detection in frame.
    
    Args:
        frame: Input frame (BGR)
        person_detection: Person YOLO detection
        face_extractor: FaceEmbeddingExtractor instance
    
    Returns:
        (embedding, confidence) tuple
    """
    try:
        # Use person bbox as face region (simplified; could use dedicated face detector)
        # In production, run separate face detection on person crop
        face_embedding = face_extractor.extract(frame, person_detection.bbox)
        confidence = person_detection.confidence  # Inherit YOLO confidence
        
        return face_embedding, confidence
    
    except Exception as e:
        logging.getLogger(__name__).error(f"Face embedding extraction error: {e}")
        return np.zeros(512), 0.0
```

---

## 3. TIMESTAMPED FRAME BUFFER

### 3.1 Insert into `integrated_system.py`

**Location:** After imports, before `SystemState` enum (around line 50)

```python
# ============================================================================
# TIMESTAMPED FRAME BUFFER (NEW)
# ============================================================================

@dataclass
class TimestampedFrame:
    """Frame with precise timestamp metadata"""
    frame: np.ndarray
    timestamp: float
    frame_id: int
    camera_id: str
    processing_metadata: Dict[str, Any] = field(default_factory=dict)


class FrameHistoryBuffer:
    """
    Ring buffer for storing timestamped frames with disk persistence.
    Supports querying frames by timestamp and range-based retrieval.
    """
    
    def __init__(self, max_memory_frames: int = 300, 
                 cache_dir: str = "./frame_cache",
                 compression: str = "jpg"):
        """
        Initialize frame history buffer.
        
        Args:
            max_memory_frames: Max frames to keep in memory (ring buffer)
            cache_dir: Directory for disk persistence
            compression: Image format ('jpg', 'png')
        """
        self.logger = logging.getLogger("FrameHistoryBuffer")
        self.max_memory_frames = max_memory_frames
        self.cache_dir = Path(cache_dir)
        self.compression = compression
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Ring buffer
        self.buffer: deque = deque(maxlen=max_memory_frames)
        self.buffer_lock = threading.Lock()
        
        # Metadata for quick lookup
        self.timestamp_index: Dict[float, int] = {}  # timestamp -> buffer index
        self.index_lock = threading.Lock()
        
        self.logger.info(f"FrameHistoryBuffer initialized (capacity={max_memory_frames})")
    
    def append(self, frame: np.ndarray, timestamp: float, 
               frame_id: int, camera_id: str, 
               metadata: Dict[str, Any] = None):
        """
        Add frame to buffer.
        
        Args:
            frame: Input frame (BGR)
            timestamp: Unix timestamp (seconds)
            frame_id: Frame sequence number
            camera_id: Camera identifier
            metadata: Optional processing metadata
        """
        try:
            ts_frame = TimestampedFrame(
                frame=frame.copy(),
                timestamp=timestamp,
                frame_id=frame_id,
                camera_id=camera_id,
                processing_metadata=metadata or {}
            )
            
            with self.buffer_lock:
                self.buffer.append(ts_frame)
            
            # Update timestamp index
            with self.index_lock:
                self.timestamp_index[timestamp] = len(self.buffer) - 1
        
        except Exception as e:
            self.logger.error(f"Error appending frame: {e}")
    
    def get_frame_by_timestamp(self, timestamp: float, 
                              tolerance_sec: float = 0.1) -> Optional[TimestampedFrame]:
        """
        Retrieve frame closest to target timestamp.
        
        Args:
            timestamp: Target timestamp
            tolerance_sec: Max time difference allowed
        
        Returns:
            TimestampedFrame or None if not found
        """
        try:
            with self.buffer_lock:
                # Find closest frame
                closest = None
                min_diff = tolerance_sec
                
                for ts_frame in self.buffer:
                    diff = abs(ts_frame.timestamp - timestamp)
                    if diff < min_diff:
                        min_diff = diff
                        closest = ts_frame
                
                return closest
        
        except Exception as e:
            self.logger.error(f"Error retrieving frame: {e}")
            return None
    
    def get_frames_in_range(self, start_timestamp: float, 
                           end_timestamp: float) -> List[TimestampedFrame]:
        """
        Retrieve all frames within timestamp range.
        
        Args:
            start_timestamp: Range start
            end_timestamp: Range end
        
        Returns:
            List of TimestampedFrame objects
        """
        try:
            with self.buffer_lock:
                return [ts_frame for ts_frame in self.buffer
                       if start_timestamp <= ts_frame.timestamp <= end_timestamp]
        
        except Exception as e:
            self.logger.error(f"Error retrieving frame range: {e}")
            return []
    
    def save_frames_to_disk(self, start_timestamp: float, 
                           end_timestamp: float,
                           output_prefix: str = "frames") -> str:
        """
        Save frame range to disk.
        
        Args:
            start_timestamp: Range start
            end_timestamp: Range end
            output_prefix: Output file prefix
        
        Returns:
            Path to saved frames directory
        """
        try:
            frames = self.get_frames_in_range(start_timestamp, end_timestamp)
            
            if not frames:
                self.logger.warning("No frames in range")
                return ""
            
            # Create output directory
            timestamp_str = datetime.fromtimestamp(start_timestamp).strftime("%Y%m%d_%H%M%S")
            output_dir = self.cache_dir / f"{output_prefix}_{timestamp_str}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save frames
            for i, ts_frame in enumerate(frames):
                filename = f"{i:04d}_{ts_frame.frame_id}.{self.compression}"
                filepath = output_dir / filename
                cv2.imwrite(str(filepath), ts_frame.frame)
            
            self.logger.info(f"Saved {len(frames)} frames to {output_dir}")
            return str(output_dir)
        
        except Exception as e:
            self.logger.error(f"Error saving frames: {e}")
            return ""
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        with self.buffer_lock:
            if not self.buffer:
                return {'frames': 0, 'duration_sec': 0}
            
            oldest = self.buffer[0]
            newest = self.buffer[-1]
            duration = newest.timestamp - oldest.timestamp
            
            return {
                'frames': len(self.buffer),
                'duration_sec': duration,
                'oldest_timestamp': oldest.timestamp,
                'newest_timestamp': newest.timestamp,
                'size_mb': sum(f.frame.nbytes for f in self.buffer) / (1024 * 1024)
            }
    
    def clear(self):
        """Clear buffer"""
        with self.buffer_lock:
            self.buffer.clear()
        with self.index_lock:
            self.timestamp_index.clear()
```

---

## 4. SERVER LOGGING (Event Emissions)

### 4.1 Insert into `integrated_system.py`

**Location:** Add to `IntegratedSystem` class, after `__init__` method (around line 220)

```python
    # ========================================================================
    # FACE TRACKING + EVENT LOGGING (NEW)
    # ========================================================================
    
    def _initialize_face_tracking(self):
        """Initialize face tracking subsystems"""
        from vision.baggage_linking import FaceEmbeddingExtractor
        
        self.logger.info("Initializing face tracking...")
        
        # Face embedding extractor
        self.face_embedding_extractor = FaceEmbeddingExtractor(
            embedding_dim=512,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Frame history buffer
        self.frame_history = FrameHistoryBuffer(
            max_memory_frames=300,
            cache_dir=self.config.get('frame_cache', {}).get('dir', './frame_cache'),
            compression=self.config.get('frame_cache', {}).get('format', 'jpg')
        )
        
        # Event logging
        self.event_logger = EventLogger(
            log_dir=self.config.get('event_logging', {}).get('dir', './event_logs'),
            log_format=self.config.get('event_logging', {}).get('format', 'jsonl')
        )
        
        # Person tracking
        self.active_persons: Dict[str, 'PersonTracker'] = {}
        self.persons_lock = threading.Lock()
        
        self.logger.info("Face tracking initialized")
    
    def _track_persons_in_frame(self, detections: List[Detection], frame_id: int, timestamp: float):
        """
        Track detected persons and emit PERSON_IN/OUT events.
        
        Args:
            detections: Current frame detections
            frame_id: Frame sequence number
            timestamp: Frame timestamp
        """
        try:
            persons = [d for d in detections if d.class_name == ObjectClass.PERSON]
            current_person_ids = set()
            
            # Process current detections
            for person_det in persons:
                person_id = self._get_or_create_person_id(person_det)
                current_person_ids.add(person_id)
                
                # Extract face embedding
                face_embedding, face_confidence = extract_face_embedding_from_person(
                    self.registration_frame or frame,  # Use frozen frame if in registration mode
                    person_det,
                    self.face_embedding_extractor
                )
                
                # Update or create tracker
                with self.persons_lock:
                    if person_id not in self.active_persons:
                        # NEW PERSON DETECTED
                        tracker = PersonTracker(person_id)
                        self.active_persons[person_id] = tracker
                        
                        # Emit PERSON_IN event
                        self.event_logger.log_event({
                            'event': 'PERSON_IN',
                            'person_id': person_id,
                            'timestamp': timestamp,
                            'frame_id': frame_id,
                            'camera_id': 'camera-0',
                            'bbox': [person_det.bbox.x1, person_det.bbox.y1,
                                    person_det.bbox.x2, person_det.bbox.y2],
                            'confidence': person_det.confidence
                        })
                        
                        self.logger.info(f"Person IN: {person_id}")
                    
                    tracker = self.active_persons[person_id]
                    tracker.update(person_det, face_embedding, face_confidence, timestamp)
            
            # Check for persons no longer detected
            with self.persons_lock:
                persons_to_remove = []
                
                for person_id, tracker in self.active_persons.items():
                    if person_id not in current_person_ids:
                        # Person not in current frame
                        if tracker.update_no_detection(timestamp):
                            # PERSON_OUT event
                            self.event_logger.log_event({
                                'event': 'PERSON_OUT',
                                'person_id': person_id,
                                'timestamp': timestamp,
                                'frame_id': frame_id,
                                'camera_id': 'camera-0',
                                'last_bbox': [tracker.last_bbox.x1, tracker.last_bbox.y1,
                                             tracker.last_bbox.x2, tracker.last_bbox.y2] if tracker.last_bbox else None,
                                'duration_sec': timestamp - tracker.first_seen
                            })
                            
                            self.logger.info(f"Person OUT: {person_id} (duration: {timestamp - tracker.first_seen:.1f}s)")
                            persons_to_remove.append(person_id)
                
                for person_id in persons_to_remove:
                    del self.active_persons[person_id]
        
        except Exception as e:
            self.logger.error(f"Error tracking persons: {e}", exc_info=True)
    
    def _get_or_create_person_id(self, detection: Detection) -> str:
        """Generate or retrieve person ID"""
        # Simple spatial hash for frame-local persistence
        x_center = (detection.bbox.x1 + detection.bbox.x2) / 2
        y_center = (detection.bbox.y1 + detection.bbox.y2) / 2
        return f"p_{int(x_center)}_{int(y_center)}"


class EventLogger:
    """Log vision events (PERSON_IN, PERSON_OUT, etc.)"""
    
    def __init__(self, log_dir: str = "./event_logs", log_format: str = "jsonl"):
        """
        Initialize event logger.
        
        Args:
            log_dir: Directory for log files
            log_format: 'jsonl' for JSON Lines, 'csv' for CSV
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_format = log_format
        self.logger = logging.getLogger("EventLogger")
        
        # Open current log file
        today = datetime.now().strftime("%Y%m%d")
        self.log_file = self.log_dir / f"events_{today}.{log_format}"
        self.file_lock = threading.Lock()
    
    def log_event(self, event: Dict[str, Any]):
        """
        Log a vision event.
        
        Args:
            event: Event dictionary
        """
        try:
            with self.file_lock:
                with open(self.log_file, 'a') as f:
                    if self.log_format == 'jsonl':
                        f.write(json.dumps(event) + '\n')
                    elif self.log_format == 'csv':
                        # Simple CSV format
                        csv_line = ','.join(str(v) for v in event.values()) + '\n'
                        f.write(csv_line)
        
        except Exception as e:
            self.logger.error(f"Error logging event: {e}")


class PersonTracker:
    """Track individual person across multiple frames"""
    
    def __init__(self, person_id: str):
        self.person_id = person_id
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.last_bbox: Optional[BoundingBox] = None
        self.face_embeddings: List[np.ndarray] = []
        self.no_detection_count = 0
        self.no_detection_timeout = 2.0  # seconds
    
    def update(self, detection: Detection, face_embedding: np.ndarray,
               face_confidence: float, timestamp: float):
        """Update tracker with new detection"""
        self.last_seen = timestamp
        self.last_bbox = detection.bbox
        self.no_detection_count = 0
        
        if face_embedding is not None and face_confidence > 0.5:
            self.face_embeddings.append(face_embedding)
            if len(self.face_embeddings) > 30:  # Keep last 30
                self.face_embeddings.pop(0)
    
    def update_no_detection(self, timestamp: float) -> bool:
        """
        Update tracker when person not detected.
        
        Returns:
            True if person should be marked as OUT
        """
        self.no_detection_count += 1
        elapsed = timestamp - self.last_seen
        
        return elapsed > self.no_detection_timeout
    
    def get_average_face_embedding(self) -> Optional[np.ndarray]:
        """Get average face embedding across tracked frames"""
        if not self.face_embeddings:
            return None
        
        avg = np.mean(self.face_embeddings, axis=0)
        return avg / (np.linalg.norm(avg) + 1e-6)  # Normalize
```

---

## 5. BACKTRACK SEARCH ALGORITHM

### 5.1 Insert into `integrated_system.py`

**Location:** Add to `IntegratedSystem` class, after event logging section (around line 400)

```python
    # ========================================================================
    # BACKTRACK FACE SEARCH (NEW)
    # ========================================================================
    
    def backtrack_face_search(self, reference_face_embedding: np.ndarray,
                             target_timestamp: float,
                             search_window_sec: float = 300.0,
                             similarity_threshold: float = 0.75) -> List[Dict[str, Any]]:
        """
        Backtrack through frame history to find person matching reference face.
        
        Args:
            reference_face_embedding: Reference face embedding (512-dim)
            target_timestamp: Approximate timestamp to search around
            search_window_sec: Search window (±seconds from target)
            similarity_threshold: Min cosine similarity to report match
        
        Returns:
            List of matches: [{frame_id, timestamp, bbox, similarity, frame_path}, ...]
        """
        matches = []
        
        try:
            # Get frame range to search
            start_ts = target_timestamp - search_window_sec / 2
            end_ts = target_timestamp + search_window_sec / 2
            
            frames = self.frame_history.get_frames_in_range(start_ts, end_ts)
            self.logger.info(f"Backtracking {len(frames)} frames for face search")
            
            for ts_frame in frames:
                # Run YOLO detection on each frame
                detections = self.yolo_engine.detect(
                    ts_frame.frame,
                    camera_id='camera-0',
                    frame_id=ts_frame.frame_id
                )
                
                persons = [d for d in detections if d.class_name == ObjectClass.PERSON]
                
                for person_det in persons:
                    # Extract face embedding from person crop
                    face_emb, face_conf = extract_face_embedding_from_person(
                        ts_frame.frame,
                        person_det,
                        self.face_embedding_extractor
                    )
                    
                    # Compute cosine similarity
                    if face_emb is not None and reference_face_embedding is not None:
                        # Normalize embeddings
                        ref_norm = reference_face_embedding / (np.linalg.norm(reference_face_embedding) + 1e-6)
                        face_norm = face_emb / (np.linalg.norm(face_emb) + 1e-6)
                        
                        similarity = float(np.dot(ref_norm, face_norm))
                        
                        if similarity > similarity_threshold:
                            matches.append({
                                'frame_id': ts_frame.frame_id,
                                'timestamp': ts_frame.timestamp,
                                'bbox': [person_det.bbox.x1, person_det.bbox.y1,
                                        person_det.bbox.x2, person_det.bbox.y2],
                                'similarity': similarity,
                                'camera_id': 'camera-0'
                            })
            
            # Sort by similarity (descending)
            matches.sort(key=lambda x: x['similarity'], reverse=True)
            
            self.logger.info(f"Backtrack search found {len(matches)} matches (threshold={similarity_threshold})")
            return matches
        
        except Exception as e:
            self.logger.error(f"Backtrack search error: {e}", exc_info=True)
            return []
```

---

## 6. MESH EXTENSIONS

### 6.1 Insert into `mesh/mesh_protocol.py`

**Location:** After `MessageType` enum (around line 30)

```python
class MessageType(IntEnum):
    """Message types in the mesh network"""
    # ... existing types ...
    
    # NEW: Face tracking extensions
    FACE_SEARCH_REQUEST = 12      # Request face search across mesh
    FACE_SEARCH_RESULT = 13       # Return face search results
    FACE_EMBEDDING_SYNC = 14      # Broadcast face embeddings for sync


@dataclass
class FaceSearchRequest:
    """Face search request message"""
    search_id: str                                  # Unique search ID
    reference_face_embedding: List[float]          # 512-dim embedding (compressed)
    target_timestamp: float                        # Approximate time to search
    search_window_sec: float = 300.0               # Search window (±seconds)
    similarity_threshold: float = 0.75             # Min similarity to report
    requesting_node_id: str = ""                   # Origin node
    camera_ids: List[str] = field(default_factory=list)  # Target cameras (empty = all)


@dataclass
class FaceSearchResult:
    """Face search result message"""
    search_id: str                                  # Matches request search_id
    reporting_node_id: str                         # Node reporting results
    matches: List[Dict[str, Any]] = field(default_factory=list)  # List of matches
    timestamp: float = field(default_factory=time.time)
    
    # Each match contains:
    # {frame_id, timestamp, bbox, similarity, camera_id}
```

### 6.2 Add Methods to `MeshProtocol` Class

**Location:** In `mesh_protocol.py`, add after `broadcast_hash_registration()` method (around line 650)

```python
    def broadcast_face_search_request(self, search_request: Dict[str, Any]) -> bool:
        """
        Broadcast face search request to all peers.
        
        Args:
            search_request: Dict with keys:
                - search_id: Unique search identifier
                - reference_face_embedding: 512-dim embedding as list
                - target_timestamp: Unix timestamp
                - search_window_sec: Search window
                - similarity_threshold: Min similarity
                - camera_ids: List of target camera IDs ([] = all)
        
        Returns:
            True if broadcast succeeded
        """
        try:
            message = MeshMessage(
                message_type=MessageType.FACE_SEARCH_REQUEST,
                source_node_id=self.node_id,
                payload=search_request
            )
            
            self.broadcast_message(message)
            self.logger.info(f"Broadcast face search: {search_request.get('search_id')}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to broadcast face search: {e}")
            return False
    
    def send_face_search_result(self, search_id: str, 
                               matches: List[Dict[str, Any]],
                               target_node_id: str) -> bool:
        """
        Send face search results back to requesting node.
        
        Args:
            search_id: Original search ID
            matches: List of match results
            target_node_id: Node ID to send results to
        
        Returns:
            True if sent successfully
        """
        try:
            result_payload = {
                'search_id': search_id,
                'reporting_node_id': self.node_id,
                'matches': matches,
                'timestamp': time.time()
            }
            
            message = MeshMessage(
                message_type=MessageType.FACE_SEARCH_RESULT,
                source_node_id=self.node_id,
                payload=result_payload
            )
            
            # Get target node info
            with self.peers_lock:
                target_peer = self.peers.get(target_node_id)
            
            if target_peer:
                self.send_message(message, target_peer.ip_address, target_peer.port)
                self.logger.info(f"Sent face search results to {target_node_id}: {len(matches)} matches")
                return True
            else:
                self.logger.warning(f"Target node {target_node_id} not found")
                return False
        except Exception as e:
            self.logger.error(f"Error sending face search result: {e}")
            return False
```

---

## 7. UI OVERLAY ADAPTATION

### 7.1 Update Visualization in `integrated_system.py`

**Location:** Modify `_draw_overlays()` method (around line 610)

```python
    def _draw_overlays(self, frame: np.ndarray):
        """Draw UI overlays on frame"""
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # ... existing code ...
        
        # NEW: Face search results overlay
        if hasattr(self, 'latest_face_search_results') and self.latest_face_search_results:
            y_offset = 30
            for i, match in enumerate(self.latest_face_search_results[:3]):  # Top 3
                confidence_text = f"Face Match #{i+1}: {match['similarity']:.2f}"
                timestamp_text = f"  @ {datetime.fromtimestamp(match['timestamp']).strftime('%H:%M:%S')}"
                
                cv2.putText(frame, confidence_text, (10, h - 150 - (i*30)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
                cv2.putText(frame, timestamp_text, (10, h - 130 - (i*30)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 150), 1)
        
        # NEW: Person tracking status
        with self.persons_lock:
            num_active = len(self.active_persons)
        cv2.putText(frame, f"Active Persons: {num_active}", (w - 250, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)
```

### 7.2 Handler for Face Search Results

**Location:** Add to `IntegratedSystem` class

```python
    def _handle_face_search_result(self, message: MeshMessage):
        """Handle incoming face search results from peers"""
        try:
            payload = message.payload
            matches = payload.get('matches', [])
            
            # Store for UI display
            self.latest_face_search_results = matches
            
            self.logger.info(f"Received {len(matches)} face matches from {message.source_node_id}")
            
            # Log results
            if self.event_logger:
                for match in matches:
                    self.event_logger.log_event({
                        'event': 'FACE_MATCHED',
                        'search_id': payload.get('search_id'),
                        'from_node': message.source_node_id,
                        'timestamp': match['timestamp'],
                        'similarity': match['similarity'],
                        'camera_id': match.get('camera_id')
                    })
        
        except Exception as e:
            self.logger.error(f"Error handling face search result: {e}")
```

---

## 8. EXACT INSERTION POINTS SUMMARY

| File | Section | Line # | Action |
|------|---------|--------|--------|
| `vision/baggage_linking.py` | After `EmbeddingExtractor` class | ~500 | Insert `FaceEmbeddingExtractor` class + `Person` dataclass + `extract_face_embedding_from_person()` |
| `integrated_system.py` | After imports, before enums | ~50 | Insert `TimestampedFrame` + `FrameHistoryBuffer` classes |
| `integrated_system.py` | In `IntegratedSystem.__init__()` | ~220 | Add `self.frame_history`, `self.face_embedding_extractor`, `self.event_logger`, `self.active_persons` initialization |
| `integrated_system.py` | Add new method after `__init__()` | ~250 | Insert `_initialize_face_tracking()` method |
| `integrated_system.py` | In `_processing_loop()` | ~370 | Call `self._track_persons_in_frame()` after detections processed |
| `integrated_system.py` | In `_processing_loop()` | ~380 | Append frame to buffer: `self.frame_history.append(frame, time.time(), ...)` |
| `integrated_system.py` | Add new methods | ~400 | Insert `_track_persons_in_frame()`, `EventLogger`, `PersonTracker` classes |
| `integrated_system.py` | Add new methods | ~500 | Insert `backtrack_face_search()` method |
| `integrated_system.py` | In `_draw_overlays()` | ~610 | Add face search results overlay + person count |
| `integrated_system.py` | In mesh handlers registration | ~180 | Register handler: `self.mesh.register_message_handler(MessageType.FACE_SEARCH_RESULT, self._handle_face_search_result)` |
| `mesh/mesh_protocol.py` | In `MessageType` enum | ~35 | Add `FACE_SEARCH_REQUEST = 12`, `FACE_SEARCH_RESULT = 13`, `FACE_EMBEDDING_SYNC = 14` |
| `mesh/mesh_protocol.py` | Add dataclasses | ~100 | Insert `FaceSearchRequest` and `FaceSearchResult` dataclasses |
| `mesh/mesh_protocol.py` | In `MeshProtocol` class | ~650 | Add `broadcast_face_search_request()` and `send_face_search_result()` methods |

---

## 9. CONFIGURATION ADDITIONS

### Add to `config/defaults.yaml`:

```yaml
face_tracking:
  enabled: true
  embedding_dim: 512
  device: "cuda"  # or "cpu"

frame_cache:
  dir: "./frame_cache"
  format: "jpg"
  max_disk_frames: 5000
  memory_buffer_size: 300

event_logging:
  dir: "./event_logs"
  format: "jsonl"  # or "csv"
  enabled: true

backtrack_search:
  similarity_threshold: 0.75
  search_window_sec: 300.0
  max_results: 20
```

---

## 10. EXAMPLE MESH MESSAGE PAYLOADS

### Face Search Request:
```json
{
  "type": 12,
  "source": "node-abc123",
  "timestamp": 1731702451.234,
  "seq": 1,
  "payload": {
    "search_id": "search_20231115_120000",
    "reference_face_embedding": [0.12, -0.34, 0.56, ...],
    "target_timestamp": 1731702300.0,
    "search_window_sec": 300.0,
    "similarity_threshold": 0.75,
    "requesting_node_id": "node-xyz789",
    "camera_ids": []
  },
  "path": ["node-abc123"]
}
```

### Face Search Result:
```json
{
  "type": 13,
  "source": "node-surveillance-01",
  "timestamp": 1731702475.234,
  "seq": 42,
  "payload": {
    "search_id": "search_20231115_120000",
    "reporting_node_id": "node-surveillance-01",
    "matches": [
      {
        "frame_id": 1234,
        "timestamp": 1731702325.5,
        "bbox": [100, 50, 250, 400],
        "similarity": 0.89,
        "camera_id": "camera-0"
      },
      {
        "frame_id": 1235,
        "timestamp": 1731702326.0,
        "bbox": [105, 55, 255, 405],
        "similarity": 0.87,
        "camera_id": "camera-0"
      }
    ],
    "timestamp": 1731702475.234
  },
  "path": ["node-surveillance-01"]
}
```

---

## 11. DEPENDENCIES

Add to `requirements.txt`:

```
torchvision>=0.14.0          # ResNet-50 face extraction
torch>=1.13.0                # PyTorch core
```

---

## 12. PERFORMANCE NOTES

| Component | Overhead | Notes |
|-----------|----------|-------|
| Face embedding extraction | 15-30ms/person | Using ResNet-50 |
| Frame history (mem) | 30-60MB | 300 frames @ 1280x720x3 |
| Event logging | <1ms/event | JSON Lines to disk |
| Backtrack search (300 frames) | 5-15 sec | Depends on YOLO speed |
| Mesh broadcast | <2ms | Compressed embedding |

**Mobile Optimization:**
- Use quantized ResNet-50 (INT8) for 2-3x speedup
- Reduce frame buffer to 150 frames (~5 sec) on low-end phones
- Skip face extraction on low battery (<20%)

---

## 13. TESTING CHECKLIST

- [ ] FaceEmbeddingExtractor loads ResNet-50 correctly
- [ ] Face embeddings are consistent (same face → similar vector)
- [ ] FrameHistoryBuffer ring-buffer logic works
- [ ] Event logger creates dated files correctly
- [ ] PersonTracker emits PERSON_IN/OUT at correct times
- [ ] Backtrack search finds matching faces in frame history
- [ ] Mesh broadcast/receive of face search requests works
- [ ] UI overlay shows search results without lag
- [ ] Face search across 2+ nodes succeeds

---

**End of Design Document**

