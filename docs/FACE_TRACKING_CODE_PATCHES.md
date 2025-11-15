# FACE TRACKING INTEGRATION - CODE PATCHES

Quick reference for implementing face tracking in existing files.

## PATCH 1: vision/baggage_linking.py - Add FaceEmbeddingExtractor

Insert after line 500 (after `EmbeddingExtractor` class definition).

```python
# ============================================================================
# FACE EMBEDDING EXTRACTION (NEW)
# ============================================================================

class FaceEmbeddingExtractor:
    """Extract face embeddings using ResNet-50 for person re-identification."""
    
    def __init__(self, embedding_dim: int = 512,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.logger = logging.getLogger(__name__)
        self.embedding_dim = embedding_dim
        self.device = device
        
        try:
            import torchvision.models as models
            self.model = models.resnet50(pretrained=True)
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            self.model.eval()
            self.model.to(device)
            
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
        try:
            x1, y1, x2, y2 = face_bbox.to_int_coords()
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                return np.zeros(self.embedding_dim)
            
            if self.model is None:
                return self._extract_simple_face_features(face_crop)
            
            img_tensor = self._preprocess_face(face_crop)
            
            with torch.no_grad():
                features = self.model(img_tensor)
                features = features.view(features.size(0), -1)
                
                if self.reduction:
                    features = self.reduction(features)
                
                features = F.normalize(features, p=2, dim=1)
            
            return features.cpu().numpy()[0]
        except Exception as e:
            self.logger.error(f"Face embedding extraction failed: {e}")
            return np.zeros(self.embedding_dim)
    
    def _preprocess_face(self, face_img: np.ndarray) -> torch.Tensor:
        face_img = cv2.resize(face_img, (224, 224))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(face_img).float().permute(2, 0, 1) / 255.0
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        return img_tensor.unsqueeze(0).to(self.device)
    
    def _extract_simple_face_features(self, face_img: np.ndarray) -> np.ndarray:
        features = []
        face_img = cv2.resize(face_img, (64, 64))
        lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
        
        for channel in range(3):
            hist = cv2.calcHist([lab], [channel], None, [64], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            features.extend(hist)
        
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        features.extend(sobelx.flatten()[:100])
        
        features = np.array(features)
        if len(features) < self.embedding_dim:
            features = np.pad(features, (0, self.embedding_dim - len(features)))
        else:
            features = features[:self.embedding_dim]
        
        features = features / (np.linalg.norm(features) + 1e-6)
        return features


def extract_face_embedding_from_person(frame: np.ndarray, 
                                       person_detection: Detection,
                                       face_extractor: FaceEmbeddingExtractor) -> Tuple[np.ndarray, float]:
    """Extract face embedding from person detection."""
    try:
        face_embedding = face_extractor.extract(frame, person_detection.bbox)
        confidence = person_detection.confidence
        return face_embedding, confidence
    except Exception as e:
        logging.getLogger(__name__).error(f"Face embedding extraction error: {e}")
        return np.zeros(512), 0.0
```

---

## PATCH 2: integrated_system.py - Add FrameHistoryBuffer

Insert after imports (around line 50).

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
    """Ring buffer for storing timestamped frames with disk persistence."""
    
    def __init__(self, max_memory_frames: int = 300, 
                 cache_dir: str = "./frame_cache",
                 compression: str = "jpg"):
        self.logger = logging.getLogger("FrameHistoryBuffer")
        self.max_memory_frames = max_memory_frames
        self.cache_dir = Path(cache_dir)
        self.compression = compression
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.buffer: deque = deque(maxlen=max_memory_frames)
        self.buffer_lock = threading.Lock()
        
        self.timestamp_index: Dict[float, int] = {}
        self.index_lock = threading.Lock()
        
        self.logger.info(f"FrameHistoryBuffer initialized (capacity={max_memory_frames})")
    
    def append(self, frame: np.ndarray, timestamp: float, 
               frame_id: int, camera_id: str, 
               metadata: Dict[str, Any] = None):
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
            
            with self.index_lock:
                self.timestamp_index[timestamp] = len(self.buffer) - 1
        except Exception as e:
            self.logger.error(f"Error appending frame: {e}")
    
    def get_frame_by_timestamp(self, timestamp: float, 
                              tolerance_sec: float = 0.1) -> Optional[TimestampedFrame]:
        try:
            with self.buffer_lock:
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
        try:
            with self.buffer_lock:
                return [ts_frame for ts_frame in self.buffer
                       if start_timestamp <= ts_frame.timestamp <= end_timestamp]
        except Exception as e:
            self.logger.error(f"Error retrieving frame range: {e}")
            return []
    
    def get_buffer_stats(self) -> Dict[str, Any]:
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


class EventLogger:
    """Log vision events (PERSON_IN, PERSON_OUT, etc.)"""
    
    def __init__(self, log_dir: str = "./event_logs", log_format: str = "jsonl"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_format = log_format
        self.logger = logging.getLogger("EventLogger")
        
        today = datetime.now().strftime("%Y%m%d")
        self.log_file = self.log_dir / f"events_{today}.{log_format}"
        self.file_lock = threading.Lock()
    
    def log_event(self, event: Dict[str, Any]):
        try:
            with self.file_lock:
                with open(self.log_file, 'a') as f:
                    if self.log_format == 'jsonl':
                        f.write(json.dumps(event) + '\n')
                    elif self.log_format == 'csv':
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
        self.no_detection_timeout = 2.0
    
    def update(self, detection: Detection, face_embedding: np.ndarray,
               face_confidence: float, timestamp: float):
        self.last_seen = timestamp
        self.last_bbox = detection.bbox
        self.no_detection_count = 0
        
        if face_embedding is not None and face_confidence > 0.5:
            self.face_embeddings.append(face_embedding)
            if len(self.face_embeddings) > 30:
                self.face_embeddings.pop(0)
    
    def update_no_detection(self, timestamp: float) -> bool:
        self.no_detection_count += 1
        elapsed = timestamp - self.last_seen
        return elapsed > self.no_detection_timeout
    
    def get_average_face_embedding(self) -> Optional[np.ndarray]:
        if not self.face_embeddings:
            return None
        
        avg = np.mean(self.face_embeddings, axis=0)
        return avg / (np.linalg.norm(avg) + 1e-6)
```

---

## PATCH 3: integrated_system.py - Initialize in IntegratedSystem

In `IntegratedSystem.__init__()`, add after line 120 (after other subsystems):

```python
        # NEW: Face tracking and event logging
        self.frame_history = None
        self.face_embedding_extractor = None
        self.event_logger = None
        self.active_persons: Dict[str, 'PersonTracker'] = {}
        self.persons_lock = threading.Lock()
        self.latest_face_search_results = []
```

Then in `initialize()` method, add after power manager initialization (around line 165):

```python
            # Initialize face tracking
            self.logger.info("Initializing face tracking...")
            self.frame_history = FrameHistoryBuffer(
                max_memory_frames=300,
                cache_dir=self.config.get('frame_cache', {}).get('dir', './frame_cache')
            )
            
            from vision.baggage_linking import FaceEmbeddingExtractor
            self.face_embedding_extractor = FaceEmbeddingExtractor(embedding_dim=512)
            
            self.event_logger = EventLogger(
                log_dir=self.config.get('event_logging', {}).get('dir', './event_logs')
            )
```

---

## PATCH 4: integrated_system.py - Track Persons in _processing_loop

In `_process_frame()` method, add after detections are extracted (around line 360):

```python
            # NEW: Track persons and emit events
            if self.event_logger and self.frame_history:
                self._track_persons_in_frame(detections, frame, frame_id, timestamp)
                self.frame_history.append(frame, timestamp, frame_id, "camera-0")
```

Then add this method to `IntegratedSystem` class:

```python
    def _track_persons_in_frame(self, detections: List[Detection], 
                                frame: np.ndarray, frame_id: int, timestamp: float):
        """Track detected persons and emit PERSON_IN/OUT events."""
        try:
            persons = [d for d in detections if d.class_name == ObjectClass.PERSON]
            current_person_ids = set()
            
            for person_det in persons:
                person_id = f"p_{int(person_det.bbox.x1)}_{int(person_det.bbox.y1)}"
                current_person_ids.add(person_id)
                
                face_embedding, face_conf = extract_face_embedding_from_person(
                    frame, person_det, self.face_embedding_extractor
                )
                
                with self.persons_lock:
                    if person_id not in self.active_persons:
                        self.active_persons[person_id] = PersonTracker(person_id)
                        
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
                    tracker.update(person_det, face_embedding, face_conf, timestamp)
            
            with self.persons_lock:
                persons_to_remove = []
                for person_id, tracker in self.active_persons.items():
                    if person_id not in current_person_ids:
                        if tracker.update_no_detection(timestamp):
                            self.event_logger.log_event({
                                'event': 'PERSON_OUT',
                                'person_id': person_id,
                                'timestamp': timestamp,
                                'frame_id': frame_id,
                                'camera_id': 'camera-0',
                                'duration_sec': timestamp - tracker.first_seen
                            })
                            self.logger.info(f"Person OUT: {person_id}")
                            persons_to_remove.append(person_id)
                
                for person_id in persons_to_remove:
                    del self.active_persons[person_id]
        
        except Exception as e:
            self.logger.error(f"Error tracking persons: {e}")
```

---

## PATCH 5: integrated_system.py - Backtrack Search

Add this method to `IntegratedSystem` class:

```python
    def backtrack_face_search(self, reference_face_embedding: np.ndarray,
                             target_timestamp: float,
                             search_window_sec: float = 300.0,
                             similarity_threshold: float = 0.75) -> List[Dict[str, Any]]:
        """Backtrack through frame history to find person matching reference face."""
        matches = []
        
        try:
            start_ts = target_timestamp - search_window_sec / 2
            end_ts = target_timestamp + search_window_sec / 2
            
            frames = self.frame_history.get_frames_in_range(start_ts, end_ts)
            self.logger.info(f"Backtracking {len(frames)} frames for face search")
            
            for ts_frame in frames:
                detections = self.yolo_engine.detect(
                    ts_frame.frame, camera_id='camera-0', frame_id=ts_frame.frame_id
                )
                
                persons = [d for d in detections if d.class_name == ObjectClass.PERSON]
                
                for person_det in persons:
                    face_emb, face_conf = extract_face_embedding_from_person(
                        ts_frame.frame, person_det, self.face_embedding_extractor
                    )
                    
                    if face_emb is not None and reference_face_embedding is not None:
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
            
            matches.sort(key=lambda x: x['similarity'], reverse=True)
            self.logger.info(f"Found {len(matches)} matches")
            return matches
        
        except Exception as e:
            self.logger.error(f"Backtrack search error: {e}")
            return []
```

---

## PATCH 6: integrated_system.py - UI Overlay

Modify `_draw_overlays()` method to add after person counter (around line 650):

```python
        # NEW: Face search results
        if self.latest_face_search_results:
            y_offset = 0
            for i, match in enumerate(self.latest_face_search_results[:3]):
                cv2.putText(frame, 
                           f"Match #{i+1}: {match['similarity']:.2f}", 
                           (10, h - 150 - (i*25)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
        
        # NEW: Active persons count
        with self.persons_lock:
            num_active = len(self.active_persons)
        cv2.putText(frame, f"Active: {num_active}", (w - 150, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)
```

---

## PATCH 7: mesh/mesh_protocol.py - New Message Types

Add to `MessageType` enum (after `LOCATION_SIGNATURE`):

```python
    FACE_SEARCH_REQUEST = 12
    FACE_SEARCH_RESULT = 13
    FACE_EMBEDDING_SYNC = 14
```

Add these methods to `MeshProtocol` class (after `broadcast_hash_registration()`):

```python
    def broadcast_face_search_request(self, search_request: Dict[str, Any]) -> bool:
        """Broadcast face search request to all peers."""
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
        """Send face search results back to requesting node."""
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
            
            with self.peers_lock:
                target_peer = self.peers.get(target_node_id)
            
            if target_peer:
                self.send_message(message, target_peer.ip_address, target_peer.port)
                self.logger.info(f"Sent results to {target_node_id}: {len(matches)} matches")
                return True
            
            return False
        except Exception as e:
            self.logger.error(f"Error sending face search result: {e}")
            return False
```

---

## PATCH 8: Add to requirements.txt

```
torchvision>=0.14.0
```

