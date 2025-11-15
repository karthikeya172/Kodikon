"""
Integrated System Runtime
Orchestrates YOLO loading, camera capture, processing pipeline, and mesh updates.
Implements full baggage tracking orchestrator with UI overlays, power management,
mesh networking, and system lifecycle management.

Features:
- Multi-camera capture with threading
- YOLO detection and embedding extraction
- Person-bag linking and mismatch detection
- Adaptive power management (ECO/BALANCED/PERFORMANCE)
- Mesh network synchronization
- Real-time UI overlays (FPS, alerts, peer count, mode)
- Search interface for baggage queries
- Graceful start/stop lifecycle
"""

import cv2
import numpy as np
import threading
import time
import logging
import queue
import yaml
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque
import uuid
import signal
import sys

from mesh.mesh_protocol import MeshProtocol, MessageType
from power.power_mode_controller import PowerModeController, PowerMode, PowerConfig
from vision.baggage_linking import (
    BaggageLinking, YOLODetectionEngine, EmbeddingExtractor,
    ColorDescriptor, BaggageProfile, PersonBagLink, LinkingStatus,
    Detection, ObjectClass, BoundingBox
)
from knowledge_graph.kg_store import KGStore  # Phase 6: Ownership tracking
import base64
from collections import deque


# ============================================================================
# ENUMS AND DATACLASSES
# ============================================================================

class SystemState(Enum):
    """System operational states"""
    STOPPED = 0
    INITIALIZING = 1
    RUNNING = 2
    PROCESSING = 3
    ERROR = 4
    SHUTTING_DOWN = 5


class CameraState(Enum):
    """Camera thread states"""
    IDLE = 0
    CAPTURING = 1
    PROCESSING = 2
    ERROR = 3


@dataclass
class FrameMetadata:
    """Metadata for processed frames"""
    timestamp: float = field(default_factory=time.time)
    camera_id: str = ""
    frame_id: int = 0
    fps: float = 0.0
    processing_time_ms: float = 0.0
    detections_count: int = 0
    links_found: int = 0
    mismatches_detected: int = 0


@dataclass
class SystemMetrics:
    """System-wide metrics"""
    total_frames_processed: int = 0
    total_detections: int = 0
    total_links_found: int = 0
    total_mismatches: int = 0
    avg_processing_time_ms: float = 0.0
    active_peers: int = 0
    current_power_mode: PowerMode = PowerMode.BALANCED
    battery_level: float = 100.0
    alerts_count: int = 0


class FrameHistoryBuffer:
    """Buffer for storing recent frames with timestamps"""
    
    def __init__(self, max_size: int = 300):
        self.max_size = max_size
        self.frames = deque(maxlen=max_size)
        self.lock = threading.Lock()
    
    def append_frame(self, frame: np.ndarray, timestamp: float):
        """Add frame to history"""
        with self.lock:
            self.frames.append((frame.copy(), timestamp))
    
    def get_frame_history(self, start_time: float = None, end_time: float = None) -> List[Tuple[np.ndarray, float]]:
        """Get frames within time range"""
        with self.lock:
            if start_time is None and end_time is None:
                return list(self.frames)
            
            result = []
            for frame, ts in self.frames:
                if start_time and ts < start_time:
                    continue
                if end_time and ts > end_time:
                    continue
                result.append((frame, ts))
            return result
    
    def get_latest_frame(self) -> Optional[Tuple[np.ndarray, float]]:
        """Get most recent frame"""
        with self.lock:
            if self.frames:
                return self.frames[-1]
            return None


# ============================================================================
# CAMERA CAPTURE WORKER
# ============================================================================

class CameraWorker(threading.Thread):
    """Dedicated camera capture thread"""
    
    def __init__(self, camera_id: str, source: int = 0, fps_config: tuple = (30, 1)):
        """
        Initialize camera worker.
        
        Args:
            camera_id: Unique camera identifier
            source: Video source (0 for default camera, or video file path)
            fps_config: Tuple of (target_fps, frame_skip)
        """
        super().__init__(daemon=True, name=f"CameraWorker-{camera_id}")
        self.camera_id = camera_id
        self.source = source
        self.target_fps, self.frame_skip = fps_config
        
        self.frame_queue = queue.Queue(maxsize=5)
        self.running = False
        self.state = CameraState.IDLE
        self.logger = logging.getLogger(f"Camera-{camera_id}")
        
        # Performance tracking
        self.frame_count = 0
        self.last_fps_update = time.time()
        self.current_fps = 0.0
        
    def run(self):
        """Capture loop running in background thread"""
        self.state = CameraState.CAPTURING
        self.running = True
        
        cap = None
        try:
            cap = cv2.VideoCapture(self.source)
            cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            frame_skip_counter = 0
            start_time = time.time()
            
            self.logger.info(f"Camera {self.camera_id} opened: {cap.isOpened()}")
            
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning(f"Failed to read frame from {self.camera_id}")
                    self.state = CameraState.ERROR
                    time.sleep(0.1)
                    continue
                
                frame_skip_counter += 1
                self.frame_count += 1
                
                # Frame skip logic
                if frame_skip_counter % self.frame_skip != 0:
                    continue
                
                # Try to queue frame (non-blocking, drop if queue full)
                try:
                    self.frame_queue.put_nowait((self.frame_count, frame.copy(), time.time()))
                    self.state = CameraState.CAPTURING
                except queue.Full:
                    self.logger.debug(f"Frame queue full for {self.camera_id}, dropping frame")
                
                # Update FPS
                current_time = time.time()
                if current_time - self.last_fps_update >= 1.0:
                    self.current_fps = self.frame_count / (current_time - start_time)
                    self.last_fps_update = current_time
                
                # Control frame rate
                time.sleep(1.0 / self.target_fps)
        
        except Exception as e:
            self.logger.error(f"Camera error: {e}")
            self.state = CameraState.ERROR
        
        finally:
            if cap:
                cap.release()
            self.state = CameraState.IDLE
            self.running = False
            self.logger.info(f"Camera {self.camera_id} closed")
    
    def get_frame(self, timeout: float = 0.1) -> Optional[Tuple[int, np.ndarray, float]]:
        """Get next frame from queue (non-blocking)"""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def stop(self):
        """Stop camera capture"""
        self.running = False


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

class IntegratedSystem:
    """
    Main orchestrator for the baggage tracking system.
    Manages lifecycle, camera capture, processing, visualization, and mesh sync.
    """
    
    def __init__(self, config_path: str = None, node_id: str = None):
        """
        Initialize integrated system.
        
        Args:
            config_path: Path to YAML config file
            node_id: Unique node identifier (auto-generated if None)
        """
        self.logger = logging.getLogger("IntegratedSystem")
        
        # Node identity
        self.node_id = node_id or f"node-{uuid.uuid4().hex[:8]}"
        self.logger.info(f"Initializing IntegratedSystem: {self.node_id}")
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # System state
        self.state = SystemState.STOPPED
        self.running = False
        self.metrics = SystemMetrics()
        
        # Subsystems (initialized but not started yet)
        self.mesh = None
        self.power = None
        self.vision = None
        self.yolo_engine = None
        self.embedding_extractor = None
        self.color_descriptor = ColorDescriptor()
        self.kg_store = None  # Phase 6: Knowledge graph store for ownership tracking
        
        # Cameras
        self.cameras: Dict[str, CameraWorker] = {}
        self.frame_queue_lock = threading.Lock()
        
        # Baggage profiles (in-memory registry)
        self.baggage_profiles: Dict[str, BaggageProfile] = {}
        self.profiles_lock = threading.Lock()
        
        # Search queries
        self.search_queue = queue.Queue()
        self.search_results = {}
        
        # Alerts
        self.alerts = deque(maxlen=100)
        self.alerts_lock = threading.Lock()
        
        # Performance tracking
        self.frame_times = deque(maxlen=30)
        self.last_mesh_sync = time.time()
        
        # Frame history for backtracking
        self.frame_history = FrameHistoryBuffer(max_size=300)
        
        # Person event log
        self.person_events = deque(maxlen=500)
        self.person_events_lock = threading.Lock()
        
        # Current frame for Command Centre
        self.current_frame = None
        self.current_frame_lock = threading.Lock()
        self.current_camera_id = None
        
        # WebSocket manager reference (set by command centre)
        self.ws_manager = None
        
        # Registration mode state
        self.registration_mode = False
        self.registration_state = "IDLE"  # IDLE, WAITING_FOR_ARRIVAL, FREEZE_FRAME, EXTRACT_FEATURES
        self.registration_freeze_time = None
        self.registration_frame = None
        self.last_registration_record = None
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _load_config(self, config_path: str = None) -> dict:
        """Load configuration from YAML file"""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "defaults.yaml"
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"Loaded config from {config_path}")
            return config
        except Exception as e:
            self.logger.warning(f"Failed to load config: {e}, using defaults")
            return {
                'camera': {'fps': 30, 'width': 1280, 'height': 720},
                'yolo': {'model': 'yolov8n', 'confidence_threshold': 0.5},
                'reid': {'model': 'osnet_x1_0', 'embedding_dim': 512},
                'power': {'mode': 'balanced', 'min_fps': 10, 'max_fps': 30},
                'mesh': {'udp_port': 9999, 'heartbeat_interval': 5}
            }
    
    def _load_camera_config(self) -> dict:
        """Load camera configuration from YAML file"""
        camera_config_path = Path(__file__).parent.parent / "config" / "cameras.yaml"
        
        try:
            with open(camera_config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"Loaded camera config from {camera_config_path}")
            return config
        except Exception as e:
            self.logger.warning(f"Failed to load camera config: {e}, using defaults")
            return {
                'cameras': [
                    {'id': 'cam1', 'name': 'Camera 1', 'url': 'http://10.7.74.56:8080/video', 'enabled': True},
                    {'id': 'cam2', 'name': 'Camera 2', 'url': 'http://10.7.74.165:8080/video', 'enabled': True},
                    {'id': 'cam3', 'name': 'Camera 3', 'url': 'http://10.7.74.168:8080/video', 'enabled': True}
                ],
                'fallback_to_local': True,
                'local_camera_id': 0,
                'fps': 30,
                'frame_skip': 1,
                'timeout': 5.0
            }
    
    def initialize(self):
        """Initialize all subsystems"""
        try:
            self.state = SystemState.INITIALIZING
            self.logger.info("Initializing subsystems...")
            
            # Initialize vision subsystem
            self.logger.info("Loading YOLO model...")
            yolo_cfg = self.config.get('yolo', {})
            self.yolo_engine = YOLODetectionEngine(
                model_name=yolo_cfg.get('model', 'yolov8n'),
                confidence_threshold=yolo_cfg.get('confidence_threshold', 0.5)
            )
            
            # Initialize embedding extractor
            self.logger.info("Loading ReID model...")
            reid_cfg = self.config.get('reid', {})
            self.embedding_extractor = EmbeddingExtractor(
                model_type=reid_cfg.get('model', 'osnet_x1_0'),
                embedding_dim=reid_cfg.get('embedding_dim', 512)
            )
            
            # Initialize power manager
            self.logger.info("Initializing power management...")
            power_cfg = self.config.get('power', {})
            # Create power config with desired mode
            from power.power_mode_controller import PowerConfig
            power_config = PowerConfig()
            mode_str = power_cfg.get('mode', 'balanced').upper()
            if mode_str in ['ECO', 'BALANCED', 'PERFORMANCE']:
                power_config.current_mode = PowerMode[mode_str]
            self.power = PowerModeController(config=power_config)
            
            # Phase 6: Initialize knowledge graph store
            self.logger.info("Initializing knowledge graph store...")
            self.kg_store = KGStore(persist_path="kg_store.json")
            
            # Initialize mesh networking
            self.logger.info("Initializing mesh network...")
            mesh_cfg = self.config.get('mesh', {})
            self.mesh = MeshProtocol(
                node_id=self.node_id,
                port=mesh_cfg.get('udp_port', 9999),
                heartbeat_interval=mesh_cfg.get('heartbeat_interval', 5),
                heartbeat_timeout=mesh_cfg.get('heartbeat_timeout', 30)
            )
            
            # Initialize vision subsystem with KGStore and MeshProtocol (Phase 6)
            self.logger.info("Initializing baggage linking...")
            vision_cfg = self.config.get('vision', {})
            self.vision = BaggageLinking(
                config=vision_cfg,
                kg_store=self.kg_store,
                mesh_protocol=self.mesh
            )
            
            # Initialize cameras
            self.logger.info("Initializing cameras...")
            camera_cfg = self.config.get('camera', {})
            fps_config = (camera_cfg.get('fps', 30), 1)
            
            # Load camera configuration
            cameras_config = self._load_camera_config()
            
            cameras_started = 0
            for cam_config in cameras_config.get('cameras', []):
                if not cam_config.get('enabled', True):
                    continue
                
                cam_id = cam_config['id']
                source = cam_config['url']
                
                try:
                    self.logger.info(f"Attempting to start {cam_id} from {source}...")
                    camera = CameraWorker(cam_id, source=source, fps_config=fps_config)
                    self.cameras[cam_id] = camera
                    camera.start()
                    
                    # Wait a bit to see if camera opens
                    time.sleep(0.5)
                    if camera.state == CameraState.CAPTURING:
                        self.logger.info(f"✓ Started camera: {cam_id}")
                        cameras_started += 1
                    else:
                        self.logger.warning(f"✗ Camera {cam_id} failed to start")
                        camera.stop()
                        del self.cameras[cam_id]
                except Exception as e:
                    self.logger.error(f"✗ Failed to start camera {cam_id}: {e}")
            
            # Fallback to local camera if no IP cameras started
            if cameras_started == 0 and cameras_config.get('fallback_to_local', True):
                self.logger.info("No IP cameras available, falling back to local camera...")
                try:
                    local_id = cameras_config.get('local_camera_id', 0)
                    camera = CameraWorker("local-camera", source=local_id, fps_config=fps_config)
                    self.cameras["local-camera"] = camera
                    camera.start()
                    time.sleep(0.5)
                    if camera.state == CameraState.CAPTURING:
                        self.logger.info("✓ Started local camera")
                        cameras_started += 1
                    else:
                        self.logger.warning("✗ Local camera failed to start")
                except Exception as e:
                    self.logger.error(f"✗ Failed to start local camera: {e}")
            
            if cameras_started == 0:
                self.logger.warning("⚠ No cameras available - system will run without video input")
            else:
                self.logger.info(f"✓ {cameras_started} camera(s) initialized")
            
            # Register mesh handlers
            self._register_mesh_handlers()
            
            self.logger.info("Subsystems initialized successfully")
            self.state = SystemState.RUNNING
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}", exc_info=True)
            self.state = SystemState.ERROR
            raise
    
    def start(self):
        """Start the integrated system"""
        try:
            self.initialize()
            self.running = True
            
            # Start mesh protocol
            self.mesh.start(local_state={
                'node_id': self.node_id,
                'power_mode': self.power.config.current_mode.name,
                'status': 'active'
            })
            
            # Start processing threads
            threading.Thread(target=self._processing_loop, daemon=True).start()
            threading.Thread(target=self._visualization_loop, daemon=True).start()
            threading.Thread(target=self._mesh_sync_loop, daemon=True).start()
            threading.Thread(target=self._search_handler_loop, daemon=True).start()
            
            self.logger.info(f"System started: {self.node_id}")
        
        except Exception as e:
            self.logger.error(f"Start failed: {e}", exc_info=True)
            self.shutdown()
            raise
    
    # ========================================================================
    # PROCESSING PIPELINE
    # ========================================================================
    
    def _processing_loop(self):
        """Main processing loop for frame handling"""
        self.logger.info("Processing loop started")
        frame_counter = 0
        
        try:
            while self.running:
                # Process frames from all cameras
                frames_processed = False
                
                for cam_id, camera in self.cameras.items():
                    result = camera.get_frame(timeout=0.01)
                    if result is None:
                        continue
                    
                    frame_id, frame, timestamp = result
                    frame_counter += 1
                    frames_processed = True
                    process_start = time.time()
                    
                    # Store frame in history
                    self.frame_history.append_frame(frame, timestamp)
                    
                    # Update current frame for Command Centre (use first available)
                    with self.current_frame_lock:
                        if self.current_frame is None or cam_id == "cam1":
                            self.current_frame = frame.copy()
                            self.current_camera_id = cam_id
                    
                    # Adaptive processing based on power mode
                    power_config = self.power.config
                    should_process_yolo = frame_counter % self._get_yolo_interval() == 0
                    
                    # Process frame
                    metadata = self._process_frame(frame, timestamp, frame_id, should_process_yolo)
                    
                    # Update metrics
                    process_time = (time.time() - process_start) * 1000
                    self.frame_times.append(process_time)
                    metadata.processing_time_ms = process_time
                    self.metrics.total_frames_processed += 1
                    self.metrics.avg_processing_time_ms = np.mean(list(self.frame_times))
                
                if not frames_processed:
                    time.sleep(0.01)
                
                # Log occasionally
                if frame_counter % 30 == 0:
                    fps = 1000.0 / self.metrics.avg_processing_time_ms if self.metrics.avg_processing_time_ms > 0 else 0
                    self.logger.debug(
                        f"Processed {frame_counter} frames | "
                        f"Avg FPS: {fps:.1f} | "
                        f"Detections: {self.metrics.total_detections} | "
                        f"Links: {self.metrics.total_links_found}"
                    )
        
        except Exception as e:
            self.logger.error(f"Processing loop error: {e}", exc_info=True)
            self.state = SystemState.ERROR
        
        finally:
            self.logger.info(f"Processing loop stopped after {frame_counter} frames")
    
    def _process_frame(self, frame: np.ndarray, timestamp: float, 
                       frame_id: int, run_yolo: bool) -> FrameMetadata:
        """
        Process single frame through entire pipeline.
        
        Args:
            frame: Input frame (BGR)
            timestamp: Frame timestamp
            frame_id: Frame sequence number
            run_yolo: Whether to run YOLO detection
        
        Returns:
            FrameMetadata with processing results
        """
        metadata = FrameMetadata(
            timestamp=timestamp,
            camera_id="camera-0",
            frame_id=frame_id
        )
        
        try:
            # Motion analysis for power management
            motion_metrics = self.power.motion_analyzer.analyze_frame(frame)
            
            detections = []
            
            # Run YOLO detection if interval reached
            if run_yolo:
                detections = self.yolo_engine.detect(
                    frame, 
                    camera_id="camera-0",
                    frame_id=frame_id
                )
                metadata.detections_count = len(detections)
                self.metrics.total_detections += len(detections)
            
            # Extract embeddings and color histograms
            for detection in detections:
                try:
                    # Extract embedding
                    detection.embedding = self.embedding_extractor.extract(
                        frame, 
                        detection.bbox
                    )
                    
                    # Extract color histogram
                    detection.color_histogram = self.color_descriptor.extract_histogram(
                        frame, 
                        detection.bbox
                    )
                except Exception as e:
                    self.logger.debug(f"Feature extraction error: {e}")
            
            # Person-bag linking
            links = self._link_persons_and_bags(detections, frame)
            metadata.links_found = len(links)
            self.metrics.total_links_found += len(links)
            
            # Mismatch detection
            mismatches = self._detect_mismatches(detections, links)
            metadata.mismatches_detected = len(mismatches)
            self.metrics.total_mismatches += len(mismatches)
            
            # Handle alerts
            if mismatches:
                for mismatch in mismatches:
                    self._create_alert(
                        f"MISMATCH: {mismatch.person_id} vs {mismatch.bag_id}",
                        priority="high",
                        data=mismatch
                    )
            
            # Emit person events
            for detection in detections:
                if detection.class_name == ObjectClass.PERSON:
                    self._emit_person_event("person_in", f"p_{detection.bbox.to_int_coords()}", timestamp)
            
            # Update power mode adaptively
            activity_density = self.power.analyze_frame(frame, detections)
            self.power.update_tracking(len(links))
            self.power.update_power_mode()
            self.metrics.current_power_mode = self.power.config.current_mode
        
        except Exception as e:
            self.logger.error(f"Frame processing error: {e}", exc_info=True)
        
        return metadata
    
    def _link_persons_and_bags(self, detections: List[Detection], 
                               frame: np.ndarray) -> List[PersonBagLink]:
        """
        Link detected persons with their baggage.
        
        Args:
            detections: List of detections
            frame: Frame for visualization
        
        Returns:
            List of PersonBagLink objects
        """
        persons = [d for d in detections if d.class_name == ObjectClass.PERSON]
        bags = [d for d in detections 
                if d.class_name in [ObjectClass.BAG, ObjectClass.BACKPACK, 
                                   ObjectClass.SUITCASE, ObjectClass.HANDBAG]]
        
        links = []
        
        for person in persons:
            best_bag = None
            best_score = 0.0
            
            for bag in bags:
                # Calculate linking metrics
                spatial_dist = person.bbox.distance_to(bag.bbox)
                feature_sim = float(np.dot(
                    person.embedding / (np.linalg.norm(person.embedding) + 1e-6),
                    bag.embedding / (np.linalg.norm(bag.embedding) + 1e-6)
                ))
                
                # Color similarity
                color_sim = self._compute_color_similarity(
                    person.color_histogram,
                    bag.color_histogram
                )
                
                # Weighted score
                score = 0.3 * feature_sim + \
                       0.4 * (1.0 - min(spatial_dist / 500.0, 1.0)) + \
                       0.3 * color_sim
                
                if score > best_score and score > 0.5:
                    best_score = score
                    best_bag = bag
            
            if best_bag:
                link = PersonBagLink(
                    person_id=f"p_{person.bbox.to_int_coords()}",
                    bag_id=f"b_{best_bag.bbox.to_int_coords()}",
                    person_detection=person,
                    bag_detection=best_bag,
                    confidence=best_score,
                    status=LinkingStatus.LINKED,
                    spatial_distance=spatial_dist,
                    feature_similarity=feature_sim,
                    color_similarity=color_sim,
                    camera_id="camera-0",
                    timestamp=datetime.now()
                )
                links.append(link)
                
                # Update baggage profiles
                self._update_baggage_profile(link)
        
        return links
    
    def _detect_mismatches(self, detections: List[Detection], 
                          links: List[PersonBagLink]) -> List[PersonBagLink]:
        """
        Detect baggage mismatches (bags without linked persons).
        
        Args:
            detections: Current frame detections
            links: Current frame links
        
        Returns:
            List of mismatch links
        """
        bags = [d for d in detections 
                if d.class_name in [ObjectClass.BAG, ObjectClass.BACKPACK,
                                   ObjectClass.SUITCASE, ObjectClass.HANDBAG]]
        linked_bag_ids = {link.bag_id for link in links}
        
        mismatches = []
        for bag in bags:
            bag_id = f"b_{bag.bbox.to_int_coords()}"
            if bag_id not in linked_bag_ids:
                mismatch = PersonBagLink(
                    person_id="UNKNOWN",
                    bag_id=bag_id,
                    bag_detection=bag,
                    confidence=0.0,
                    status=LinkingStatus.SUSPICIOUS,
                    camera_id="camera-0",
                    timestamp=datetime.now()
                )
                mismatches.append(mismatch)
        
        return mismatches
    
    def _compute_color_similarity(self, hist1, hist2) -> float:
        """Compute color histogram similarity"""
        try:
            # Use Bhattacharyya distance
            sim_h = cv2.compareHist(hist1.h_hist, hist2.h_hist, cv2.HISTCMP_BHATTACHARYYA)
            sim_s = cv2.compareHist(hist1.s_hist, hist2.s_hist, cv2.HISTCMP_BHATTACHARYYA)
            sim_v = cv2.compareHist(hist1.v_hist, hist2.v_hist, cv2.HISTCMP_BHATTACHARYYA)
            
            # Average and convert to similarity (1 - distance)
            avg_distance = (sim_h + sim_s + sim_v) / 3.0
            return 1.0 - min(1.0, avg_distance)
        except:
            return 0.0
    
    def _update_baggage_profile(self, link: PersonBagLink):
        """Update baggage profile from link"""
        with self.profiles_lock:
            bag_id = link.bag_id
            if bag_id not in self.baggage_profiles:
                hash_id = self._generate_hash_id(link.bag_detection)
                profile = BaggageProfile(
                    bag_id=bag_id,
                    hash_id=hash_id,
                    class_name=link.bag_detection.class_name,
                    color_histogram=link.bag_detection.color_histogram,
                    embedding=link.bag_detection.embedding,
                    person_id=link.person_id
                )
                self.baggage_profiles[bag_id] = profile
            else:
                profile = self.baggage_profiles[bag_id]
                profile.last_seen = datetime.now()
                profile.detections.append(link.bag_detection)
                profile.camera_ids.append(link.camera_id)
    
    def _generate_hash_id(self, detection: Detection) -> str:
        """Generate unique hash for baggage item"""
        import hashlib
        data = str(detection.embedding.tobytes()).encode()
        return hashlib.sha256(data).hexdigest()[:16]
    
    def _get_yolo_interval(self) -> int:
        """Get YOLO processing interval based on power mode"""
        mode = self.power.config.current_mode
        if mode == PowerMode.ECO:
            return self.power.config.yolo_interval_eco
        elif mode == PowerMode.BALANCED:
            return self.power.config.yolo_interval_balanced
        else:  # PERFORMANCE
            return self.power.config.yolo_interval_performance
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    
    def _visualization_loop(self):
        """Visualization loop for UI rendering"""
        self.logger.info("Visualization loop started")
        
        try:
            while self.running:
                # Get frame from first available camera
                frame = None
                for cam_id, camera in self.cameras.items():
                    result = camera.get_frame(timeout=0.1)
                    if result is not None:
                        _, frame, _ = result
                        break
                
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                # Draw overlays
                frame_overlay = frame.copy()
                self._draw_overlays(frame_overlay)
                
                # Display
                cv2.imshow("Kodikon Baggage Tracker", frame_overlay)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                elif key == ord('s'):
                    self._trigger_search_ui()
                
                # NEW: Registration mode handling
                elif key == ord('r') or key == ord('R'):
                    self.registration_mode = not self.registration_mode
                    self.registration_state = "IDLE"
                    self.logger.info(f"Registration mode: {'ON' if self.registration_mode else 'OFF'}")
                
                elif key == ord(' '):  # SPACE
                    if self.registration_mode and self.registration_state == "WAITING_FOR_ARRIVAL":
                        self.registration_state = "FREEZE_FRAME"
                        self.registration_freeze_time = time.time()
                        self.registration_frame = frame_overlay.copy()
                        self.logger.info("Frame frozen for registration")
                
                elif key == 27:  # ESC
                    if self.registration_mode:
                        self.registration_mode = False
                        self.registration_state = "IDLE"
                        self.logger.info("Registration cancelled")
        
        except Exception as e:
            self.logger.error(f"Visualization error: {e}")
        
        finally:
            cv2.destroyAllWindows()
    
    def _draw_overlays(self, frame: np.ndarray):
        """Draw UI overlays on frame"""
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # FPS overlay
        avg_time = self.metrics.avg_processing_time_ms
        fps = 1000.0 / avg_time if avg_time > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), font, 1, (0, 255, 0), 2)
        
        # Power mode
        mode_name = self.metrics.current_power_mode.name
        cv2.putText(frame, f"Mode: {mode_name}", (10, 70), font, 0.8, (0, 255, 255), 2)
        
        # Peer count
        with self.mesh.peers_lock if self.mesh else threading.Lock():
            peer_count = len(self.mesh.peers) if self.mesh else 0
        cv2.putText(frame, f"Peers: {peer_count}", (10, 110), font, 0.8, (255, 0, 0), 2)
        
        # Metrics
        cv2.putText(
            frame,
            f"Detections: {self.metrics.total_detections} | Links: {self.metrics.total_links_found}",
            (10, 150), font, 0.7, (200, 200, 0), 1
        )
        
        # Alerts
        with self.alerts_lock:
            if self.alerts:
                latest_alert = self.alerts[-1]
                cv2.putText(
                    frame,
                    f"Alert: {latest_alert[:50]}...",
                    (10, h - 20), font, 0.7, (0, 0, 255), 2
                )
        
        # NEW: Registration mode overlay
        if self.registration_mode:
            if self.registration_state == "IDLE":
                # Waiting for user to press SPACE
                cv2.putText(frame, "[REGISTRATION MODE] Press SPACE to freeze", 
                           (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            elif self.registration_state == "WAITING_FOR_ARRIVAL":
                cv2.putText(frame, "[WAITING] Person + Bag must be visible", 
                           (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            elif self.registration_state == "FREEZE_FRAME":
                elapsed = time.time() - self.registration_freeze_time
                countdown = max(0, 1.0 - elapsed)
                cv2.putText(frame, 
                           f"[FREEZE] {countdown:.1f}s", 
                           (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                
                # Draw center overlay
                cv2.putText(frame, 
                           f"{int(countdown * 3)}...", 
                           (w//2 - 50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 0), 3)
                
                # After 1 second, extract features
                if elapsed > 1.0 and self.registration_state == "FREEZE_FRAME":
                    self.registration_state = "EXTRACT_FEATURES"
                    # Trigger registration async
                    threading.Thread(target=self._process_registration, daemon=True).start()
            
            elif self.registration_state == "EXTRACT_FEATURES":
                if self.last_registration_record:
                    cv2.putText(frame, 
                               f"✓ Registered: {self.last_registration_record.hash_id[:8]}", 
                               (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    # Show for 2 seconds
                    if time.time() - self.registration_freeze_time > 3.0:
                        self.registration_state = "IDLE"
    
    # ========================================================================
    # MESH SYNCHRONIZATION
    # ========================================================================
    
    def _register_mesh_handlers(self):
        """Register handlers for mesh messages"""
        if not self.mesh:
            return
        
        self.mesh.register_message_handler(
            MessageType.SEARCH_QUERY,
            self._handle_search_query
        )
        self.mesh.register_message_handler(
            MessageType.ALERT,
            self._handle_alert_message
        )
        self.mesh.register_message_handler(
            MessageType.NODE_STATE_SYNC,
            self._handle_state_sync
        )
    
    def _handle_search_query(self, message):
        """Handle incoming search query"""
        query = message.payload.get('query', {})
        self.logger.info(f"Received search query: {query}")
        
        # Search local profiles
        results = self._search_baggage(query)
        
        # Send results back
        if self.mesh and message.source_node_id:
            response = {
                'query_id': query.get('id'),
                'results': [p.to_dict() for p in results]
            }
            self.mesh.send_message(
                message=self.mesh.MeshMessage(
                    message_type=MessageType.HASH_REGISTRY,
                    source_node_id=self.node_id,
                    payload=response
                ),
                target_ip=message.source_node_id
            )
    
    def _handle_alert_message(self, message):
        """Handle incoming alert"""
        alert = message.payload.get('alert')
        self._create_alert(alert, priority="network")
    
    def _handle_state_sync(self, message):
        """Handle node state sync"""
        remote_state = message.payload.get('state', {})
        if self.mesh:
            self.mesh.node_state.merge_remote_state(remote_state)
    
    def _mesh_sync_loop(self):
        """Periodic mesh synchronization"""
        self.logger.info("Mesh sync loop started")
        
        try:
            while self.running:
                if self.mesh and time.time() - self.last_mesh_sync > 5.0:
                    # Broadcast current state
                    state = {
                        'node_id': self.node_id,
                        'baggage_count': len(self.baggage_profiles),
                        'total_detections': self.metrics.total_detections,
                        'power_mode': self.metrics.current_power_mode.name,
                        'timestamp': time.time()
                    }
                    self.mesh.node_state.update_local_state(state)
                    self.last_mesh_sync = time.time()
                
                time.sleep(1.0)
        
        except Exception as e:
            self.logger.error(f"Mesh sync error: {e}")
    
    # ========================================================================
    # SEARCH INTERFACE
    # ========================================================================
    
    def _search_handler_loop(self):
        """Process search queries from queue"""
        self.logger.info("Search handler loop started")
        
        try:
            while self.running:
                try:
                    query = self.search_queue.get(timeout=1.0)
                    results = self._search_baggage(query)
                    self.search_results[query.get('id')] = results
                except queue.Empty:
                    continue
        
        except Exception as e:
            self.logger.error(f"Search handler error: {e}")
    
    def _process_registration(self):
        """Process the frozen frame for registration"""
        try:
            if self.registration_frame is None or self.yolo_engine is None:
                self.logger.warning("Registration frame or YOLO not available")
                self.registration_state = "IDLE"
                return
            
            # Import registration functions
            from baggage_linking import register_from_frame
            
            # Create mesh node wrapper if needed
            record, success = register_from_frame(
                frame=self.registration_frame,
                mesh_node=self.mesh,
                yolo_model=self.yolo_engine.model if hasattr(self.yolo_engine, 'model') else None,
                camera_id=self.node_id,
                metadata={'timestamp': time.time(), 'mode': 'integrated_system'}
            )
            
            if success:
                self.last_registration_record = record
                self.logger.info(f"✓ Registration successful: {record.hash_id}")
                # Add alert
                with self.alerts_lock:
                    self.alerts.append(f"✓ Registered person-bag: {record.hash_id[:8]}")
            else:
                self.logger.warning("✗ Registration failed: Invalid detection")
                self.registration_state = "WAITING_FOR_ARRIVAL"
                with self.alerts_lock:
                    self.alerts.append("✗ Registration failed: Person + bag must be visible")
                
        except Exception as e:
            self.logger.error(f"Registration processing error: {e}", exc_info=True)
            self.registration_state = "IDLE"
            with self.alerts_lock:
                self.alerts.append(f"✗ Registration error: {str(e)[:40]}")
    
    def _search_baggage(self, query: Dict) -> List[BaggageProfile]:
        """
        Search baggage profiles by criteria.
        
        Args:
            query: Search query dict with 'description', 'color', 'embedding', etc.
        
        Returns:
            List of matching BaggageProfile objects
        """
        results = []
        
        with self.profiles_lock:
            for profile in self.baggage_profiles.values():
                score = 0.0
                
                # Description match
                if 'description' in query:
                    query_desc = query['description'].lower()
                    if query_desc in (profile.description or "").lower():
                        score += 0.5
                
                # Color match
                if 'color' in query and profile.color_histogram:
                    color_sim = self._compute_color_similarity(
                        profile.color_histogram,
                        query['color']
                    )
                    score += color_sim * 0.3
                
                # Embedding similarity
                if 'embedding' in query and profile.embedding is not None:
                    query_emb = np.array(query['embedding'])
                    emb_sim = float(np.dot(
                        profile.embedding / (np.linalg.norm(profile.embedding) + 1e-6),
                        query_emb / (np.linalg.norm(query_emb) + 1e-6)
                    ))
                    score += max(0, emb_sim) * 0.2
                
                if score > 0.3:
                    results.append((profile, score))
        
        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in results[:10]]  # Top 10
    
    def search_by_description(self, description: str) -> List[Dict]:
        """Public search API by description"""
        results = self._search_baggage({'description': description})
        return [r.to_dict() for r in results]
    
    def _trigger_search_ui(self):
        """Interactive search interface"""
        try:
            description = input("Enter baggage description: ")
            results = self.search_by_description(description)
            print(f"Found {len(results)} results:")
            for r in results:
                print(f"  - {r['hash_id']}: {r['description']}")
        except Exception as e:
            self.logger.error(f"Search UI error: {e}")
    
    # ========================================================================
    # ALERTS
    # ========================================================================
    
    def _create_alert(self, message: str, priority: str = "normal", data=None):
        """Create system alert"""
        with self.alerts_lock:
            self.alerts.append(message)
            self.metrics.alerts_count += 1
        
        self.logger.warning(f"[{priority.upper()}] {message}")
        
        # Broadcast to mesh
        if self.mesh and priority != "network":
            self.mesh.broadcast_alert({
                'alert': message,
                'priority': priority,
                'timestamp': time.time()
            })
        
        # Emit to WebSocket
        if self.ws_manager:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self.ws_manager.send_alert(priority, {
                        'message': message,
                        'timestamp': time.time()
                    }))
            except:
                pass
    
    def _emit_person_event(self, event_type: str, person_id: str, timestamp: float):
        """Emit person in/out event"""
        event = {
            'type': event_type,
            'person_id': person_id,
            'timestamp': timestamp
        }
        
        with self.person_events_lock:
            self.person_events.append(event)
        
        # Emit to WebSocket
        if self.ws_manager:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self.ws_manager.send_person_event(event_type, person_id, timestamp))
            except:
                pass
    
    # ========================================================================
    # COMMAND CENTRE API METHODS
    # ========================================================================
    
    def get_status_snapshot(self) -> Dict:
        """Get current system status for Command Centre"""
        with self.mesh.peers_lock if self.mesh else threading.Lock():
            peer_count = len(self.mesh.peers) if self.mesh else 0
        
        avg_time = self.metrics.avg_processing_time_ms
        fps = 1000.0 / avg_time if avg_time > 0 else 0
        
        # Get camera statuses
        cameras = []
        for cam_id, camera in self.cameras.items():
            cameras.append({
                'camera_id': cam_id,
                'fps': camera.current_fps,
                'state': camera.state.name,
                'frame_count': camera.frame_count
            })
        
        return {
            'node_id': self.node_id,
            'power_mode': self.metrics.current_power_mode.name,
            'fps': round(fps, 1),
            'activity': self.metrics.total_detections,
            'peers': list(self.mesh.peers.keys()) if self.mesh else [],
            'cameras': cameras,
            'timestamp': time.time(),
            'total_frames': self.metrics.total_frames_processed,
            'total_detections': self.metrics.total_detections,
            'total_links': self.metrics.total_links_found,
            'alerts_count': self.metrics.alerts_count
        }
    
    def get_current_frame_jpeg(self) -> Optional[bytes]:
        """Get current frame as JPEG bytes"""
        with self.current_frame_lock:
            if self.current_frame is None:
                return None
            
            # Encode to JPEG
            success, buffer = cv2.imencode('.jpg', self.current_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if success:
                return buffer.tobytes()
            return None
    
    def get_person_event_log(self, limit: int = 100) -> List[Dict]:
        """Get recent person events"""
        with self.person_events_lock:
            events = list(self.person_events)[-limit:]
            return events
    
    def extract_face_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract face embedding from image"""
        try:
            # Use embedding extractor
            if self.embedding_extractor is None:
                return None
            
            # Detect face region (simplified - use full image)
            h, w = image.shape[:2]
            bbox = BoundingBox(0, 0, w, h)
            
            # Extract embedding
            embedding = self.embedding_extractor.extract(image, bbox)
            return embedding
        except Exception as e:
            self.logger.error(f"Error extracting face embedding: {e}")
            return None
    
    def run_face_backtrack(self, embedding: np.ndarray, timestamp: float = None) -> List[Dict]:
        """Run face backtracking search"""
        try:
            results = []
            
            # Get frame history
            if timestamp:
                # Search around timestamp (±5 minutes)
                start_time = timestamp - 300
                end_time = timestamp + 300
                frames = self.frame_history.get_frame_history(start_time, end_time)
            else:
                # Search all frames
                frames = self.frame_history.get_frame_history()
            
            # Search for matching faces
            for frame, ts in frames:
                try:
                    # Extract embedding from frame
                    frame_embedding = self.extract_face_embedding(frame)
                    if frame_embedding is None:
                        continue
                    
                    # Compute similarity
                    similarity = float(np.dot(
                        embedding / (np.linalg.norm(embedding) + 1e-6),
                        frame_embedding / (np.linalg.norm(frame_embedding) + 1e-6)
                    ))
                    
                    if similarity > 0.7:  # Threshold
                        # Encode frame to base64
                        success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        if success:
                            frame_b64 = base64.b64encode(buffer).decode('utf-8')
                            
                            results.append({
                                'match_timestamp': ts,
                                'confidence': similarity,
                                'frame': frame_b64,
                                'hash_id': None  # TODO: Link to baggage hash_id
                            })
                except Exception as e:
                    self.logger.debug(f"Error processing frame: {e}")
                    continue
            
            # Sort by confidence
            results.sort(key=lambda x: x['confidence'], reverse=True)
            return results[:10]  # Top 10 matches
        
        except Exception as e:
            self.logger.error(f"Error in face backtrack: {e}")
            return []
    
    def set_ws_manager(self, ws_manager):
        """Set WebSocket manager for Command Centre"""
        self.ws_manager = ws_manager
    
    # ========================================================================
    # LIFECYCLE
    # ========================================================================
    
    def _signal_handler(self, sig, frame):
        """Handle shutdown signals"""
        self.logger.info("Shutdown signal received")
        self.shutdown()
    
    def run(self):
        """Run the system (blocking)"""
        try:
            self.start()
            self.logger.info("System is running. Press Ctrl+C to stop.")
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.logger.info("Interrupted")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Clean shutdown of all subsystems"""
        if self.state == SystemState.SHUTTING_DOWN:
            return
        
        self.state = SystemState.SHUTTING_DOWN
        self.logger.info("Shutting down...")
        
        self.running = False
        
        # Stop cameras
        for camera in self.cameras.values():
            camera.stop()
        
        # Stop mesh
        if self.mesh:
            self.mesh.stop()
        
        # Close CV windows
        cv2.destroyAllWindows()
        
        self.state = SystemState.STOPPED
        self.logger.info("Shutdown complete")




# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    system = IntegratedSystem()
    system.run()
