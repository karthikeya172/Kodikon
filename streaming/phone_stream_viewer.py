import cv2
import numpy as np
from threading import Thread, Lock
import time
import logging
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from queue import Queue
import argparse
import sys
import os

# Add parent directory to path for mesh imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optional mesh network support
try:
    from mesh.udp_setup_guide import IntegratedMeshNode
    from mesh.event_broadcaster import CameraRole, EventType
    MESH_AVAILABLE = True
except ImportError:
    MESH_AVAILABLE = False
    logger.warning("Mesh network not available. Install mesh components.")

# Optional YOLO support
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("YOLO not available. Install with: pip install ultralytics")

@dataclass
class StreamConfig:
    """Configuration for stream viewing"""
    url: str
    name: str
    max_retries: int = 3
    retry_delay: int = 2
    enable_yolo: bool = True
    confidence_threshold: float = 0.5
    
    def __post_init__(self):
        if self.enable_yolo and not YOLO_AVAILABLE:
            logger.warning(f"{self.name}: YOLO disabled (not installed)")
            self.enable_yolo = False

class WebcamStream:
    """Threaded IP Webcam stream reader with auto-reconnection and optional YOLO inference"""
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.stream = None
        self.ret = False
        self.frame = None
        self.frame_lock = Lock()
        self.running = True
        self.connected = False
        self.frame_count = 0
        self.fps_counter = {"frames": 0, "start_time": time.time()}
        
        # YOLO inference
        self.yolo_model = None
        self.detections = []
        self.detection_lock = Lock()
        if config.enable_yolo and YOLO_AVAILABLE:
            self._initialize_yolo()
        
        self._initialize_stream()
        
    def _initialize_yolo(self):
        """Initialize YOLO model for inference"""
        try:
            self.yolo_model = YOLO("yolov8n.pt")  # Nano model for lightweight inference
            logger.info(f"{self.config.name}: YOLO model loaded")
        except Exception as e:
            logger.error(f"{self.config.name}: YOLO initialization failed - {e}")
            self.yolo_model = None
        
    def _initialize_stream(self):
        """Attempt to initialize video stream with retries"""
        for attempt in range(self.config.max_retries):
            try:
                self.stream = cv2.VideoCapture(self.config.url)
                # Set buffer size to minimize latency
                self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.ret, self.frame = self.stream.read()
                if self.ret and self.frame is not None:
                    self.connected = True
                    logger.info(f"{self.config.name}: Connected successfully (resolution: {self.frame.shape[1]}x{self.frame.shape[0]})")
                    return
                self.stream.release()
            except Exception as e:
                logger.warning(f"{self.config.name}: Attempt {attempt + 1} failed - {e}")
            
            if attempt < self.config.max_retries - 1:
                time.sleep(self.config.retry_delay)
        
        logger.warning(f"{self.config.name}: Failed to connect after {self.config.max_retries} attempts")
        
    def start(self):
        """Start frame capture thread"""
        Thread(target=self.update, daemon=True, name=f"{self.config.name}-capture").start()
        if self.yolo_model:
            Thread(target=self._inference_loop, daemon=True, name=f"{self.config.name}-yolo").start()
        return self
    
    def _run_yolo_inference(self, frame):
        """Run YOLO inference on frame"""
        try:
            if self.yolo_model is None:
                return
            
            results = self.yolo_model(frame, conf=self.config.confidence_threshold, verbose=False)
            detections = []
            
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and result.boxes is not None:
                    for box in result.boxes:
                        try:
                            # Handle tensor/array conversion
                            class_id = int(box.cls[0]) if hasattr(box.cls, '__len__') else int(box.cls)
                            class_name = result.names.get(class_id, f"class_{class_id}")
                            conf = float(box.conf[0]) if hasattr(box.conf, '__len__') else float(box.conf)
                            bbox = box.xyxy[0].tolist() if len(box.xyxy) > 0 else box.xyxy.tolist()
                            
                            detections.append({
                                "class": class_name,
                                "confidence": conf,
                                "bbox": bbox  # [x1, y1, x2, y2]
                            })
                        except Exception as box_err:
                            logger.debug(f"{self.config.name}: Error processing box - {box_err}")
                            continue
            
            with self.detection_lock:
                self.detections = detections
                
            if detections:
                logger.info(f"{self.config.name}: Detected {len(detections)} objects")
                
        except Exception as e:
            logger.error(f"{self.config.name}: YOLO inference error - {e}", exc_info=True)
    
    def _inference_loop(self):
        """Background thread for YOLO inference"""
        while self.running:
            frame_copy = None
            with self.frame_lock:
                if self.frame is not None and self.frame.size > 0:
                    frame_copy = self.frame.copy()
            
            if frame_copy is not None:
                self._run_yolo_inference(frame_copy)
            
            time.sleep(0.1)  # Inference throttling
        
    def update(self):
        """Capture frames in background thread"""
        consecutive_failures = 0
        max_consecutive_failures = 5
        
        while self.running:
            if self.connected and self.stream:
                try:
                    self.ret, frame = self.stream.read()
                    if self.ret and frame is not None:
                        with self.frame_lock:
                            self.frame = frame
                        self.frame_count += 1
                        consecutive_failures = 0
                        
                        # Update FPS counter
                        self.fps_counter["frames"] += 1
                        elapsed = time.time() - self.fps_counter["start_time"]
                        if elapsed >= 1.0:
                            fps = self.fps_counter["frames"] / elapsed
                            logger.debug(f"{self.config.name}: {fps:.1f} FPS")
                            self.fps_counter = {"frames": 0, "start_time": time.time()}
                    else:
                        consecutive_failures += 1
                        if consecutive_failures >= max_consecutive_failures:
                            logger.warning(f"{self.config.name}: Stream lost, attempting reconnect...")
                            self.connected = False
                            self._initialize_stream()
                            consecutive_failures = 0
                except Exception as e:
                    logger.error(f"{self.config.name}: Capture error - {e}")
                    self.connected = False
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        self._initialize_stream()
                        consecutive_failures = 0
            else:
                # Attempt reconnection
                self._initialize_stream()
                if not self.connected:
                    time.sleep(self.config.retry_delay)
            
    def read(self) -> Optional[np.ndarray]:
        """Read current frame"""
        with self.frame_lock:
            return self.frame.copy() if self.connected and self.frame is not None else None
    
    def get_detections(self) -> List[Dict]:
        """Get latest YOLO detections"""
        with self.detection_lock:
            return self.detections.copy() if self.detections else []
    
    def draw_detections(self, frame: np.ndarray) -> np.ndarray:
        """Draw YOLO detections on frame"""
        if not self.detections:
            return frame
        
        for det in self.detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            # Clamp coordinates to frame
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{det['class']} {det['confidence']:.2f}"
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame
    
    def stop(self):
        """Stop capture thread"""
        self.running = False
        if self.stream:
            self.stream.release()
        logger.info(f"{self.config.name}: Stream closed")

class StreamGridDisplay:
    """Multi-feed grid display with dynamic layout"""
    
    def __init__(self, window_title: str = "Phone Feeds", max_width: int = 1920, max_height: int = 1080):
        self.window_title = window_title
        self.max_width = max_width
        self.max_height = max_height
    
    def _create_grid_layout(self, num_feeds: int) -> Tuple[int, int]:
        """Calculate optimal grid layout (rows, cols)"""
        if num_feeds <= 1:
            return 1, 1
        # Calculate columns first (square root rounded up)
        cols = int(np.ceil(np.sqrt(num_feeds)))
        # Calculate rows needed
        rows = int(np.ceil(num_feeds / cols))
        return rows, cols
    
    def create_grid(self, frames: List[np.ndarray]) -> np.ndarray:
        """Stitch frames into grid display"""
        if not frames:
            empty = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(empty, "No streams available", (50, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return empty
        
        if len(frames) == 1:
            return frames[0]
        
        n = len(frames)
        rows, cols = self._create_grid_layout(n)
        h, w = frames[0].shape[:2]
        
        # Pad with black frames if needed
        pad_count = rows * cols - n
        for _ in range(pad_count):
            frames.append(np.zeros((h, w, 3), dtype=np.uint8))
        
        # Create grid by stacking rows
        rows_list = []
        for r in range(rows):
            row_frames = frames[r * cols:(r + 1) * cols]
            rows_list.append(np.hstack(row_frames))
        grid = np.vstack(rows_list)
        
        # Resize if necessary to fit screen
        grid_h, grid_w = grid.shape[:2]
        if grid_h > self.max_height or grid_w > self.max_width:
            scale = min(self.max_width / grid_w, self.max_height / grid_h)
            new_w, new_h = int(grid_w * scale), int(grid_h * scale)
            grid = cv2.resize(grid, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        return grid
    
    def display(self, grid: np.ndarray):
        """Display grid in window"""
        cv2.imshow(self.window_title, grid)


class PhoneStreamViewer:
    """Lightweight IP Webcam viewer with multi-feed support, YOLO inference, and mesh networking"""
    
    def __init__(self, configs: List[StreamConfig], enable_yolo: bool = False, enable_mesh: bool = True,
                 mesh_port: int = 9999, mesh_location: str = "streaming_hub"):
        self.configs = configs
        self.streams: List[WebcamStream] = []
        self.display = StreamGridDisplay()
        self.running = True
        self.global_yolo = None
        
        # Mesh network integration
        self.mesh_node = None
        self.enable_mesh = enable_mesh and MESH_AVAILABLE
        self.mesh_port = mesh_port
        self.mesh_location = mesh_location
        self.frame_id = 0
        
        # Initialize global YOLO if requested
        if enable_yolo and YOLO_AVAILABLE:
            self._initialize_global_yolo()
        
        # Initialize mesh node if requested
        if self.enable_mesh:
            self._initialize_mesh_node()
        
        for config in configs:
            config.enable_yolo = enable_yolo and (self.global_yolo is not None)
            stream = WebcamStream(config).start()
            self.streams.append(stream)
            time.sleep(0.5)  # Stagger stream initialization
    
    def _initialize_global_yolo(self):
        """Initialize global YOLO model shared across streams"""
        try:
            self.global_yolo = YOLO("yolov8n.pt")
            logger.info("Global YOLO model loaded (nano)")
        except Exception as e:
            logger.error(f"Global YOLO initialization failed: {e}")
            self.global_yolo = None
    
    def _initialize_mesh_node(self):
        """Initialize mesh network node"""
        try:
            logger.info("Initializing mesh network node...")
            self.mesh_node = IntegratedMeshNode(
                node_id=f"streaming_{int(time.time() % 10000)}",
                port=self.mesh_port,
                location_signature=self.mesh_location,
                camera_role="surveillance"  # Streaming hub is surveillance point
            )
            self.mesh_node.start()
            logger.info(f"Mesh node started: {self.mesh_node.node_id} on port {self.mesh_port}")
            
            # Register event listeners for mesh updates
            self._setup_mesh_listeners()
        except Exception as e:
            logger.error(f"Failed to initialize mesh node: {e}")
            self.mesh_node = None
            self.enable_mesh = False
    
    def _setup_mesh_listeners(self):
        """Setup mesh event listeners"""
        if not self.mesh_node:
            return
        
        def on_mismatch(event):
            logger.warning(f"MESH ALERT: Mismatch at {event.location_signature} - Person {event.person_ids[0]}")
        
        def on_transfer(event):
            logger.info(f"MESH: Transfer detected - Bag {event.bag_ids[0]}")
        
        self.mesh_node.register_event_handler(EventType.MISMATCH_ALERT, on_mismatch)
        self.mesh_node.register_event_handler(EventType.BAG_TRANSFER, on_transfer)
    
    def _prepare_frame(self, stream: WebcamStream, frame: np.ndarray) -> np.ndarray:
        """Prepare frame with metadata and optional detections"""
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        
        # Draw stream name
        cv2.putText(display_frame, stream.config.name, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw connection status
        status = "CONNECTED" if stream.connected else "RECONNECTING"
        color = (0, 255, 0) if stream.connected else (0, 0, 255)
        cv2.putText(display_frame, status, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw frame count
        cv2.putText(display_frame, f"Frames: {stream.frame_count}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Draw YOLO detections if available
        if stream.config.enable_yolo and stream.get_detections():
            display_frame = stream.draw_detections(display_frame)
            det_count = len(stream.get_detections())
            cv2.putText(display_frame, f"Detections: {det_count}", (w - 200, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
            # Broadcast detections to mesh network
            if self.enable_mesh and self.mesh_node:
                self._broadcast_detections_to_mesh(stream, det_count)
        
        # Draw mesh status
        if self.enable_mesh and self.mesh_node:
            mesh_status = "MESH OK" if self.mesh_node.running else "MESH OFF"
            mesh_color = (0, 255, 0) if self.mesh_node.running else (0, 0, 255)
            cv2.putText(display_frame, mesh_status, (w - 150, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, mesh_color, 1)
        
        return display_frame
    
    def _broadcast_detections_to_mesh(self, stream: WebcamStream, detection_count: int):
        """Broadcast YOLO detections to mesh network"""
        try:
            if not self.mesh_node or not stream.get_detections():
                return
            
            detections = stream.get_detections()
            persons = [d['id'] for d in detections if d.get('class_name') == 'person']
            bags = [d['id'] for d in detections if d.get('class_name') in ['bag', 'backpack', 'suitcase']]
            
            if persons or bags:
                self.mesh_node.process_vision_frame(
                    detected_persons=persons,
                    detected_bags=bags,
                    person_bag_links={},
                    frame_metadata={
                        'frame_id': self.frame_id,
                        'stream': stream.config.name,
                        'detection_count': detection_count,
                        'confidence': stream.config.confidence_threshold
                    }
                )
                self.frame_id += 1
        except Exception as e:
            logger.debug(f"Error broadcasting detections: {e}")
    
    def _create_placeholder_frame(self, stream: WebcamStream) -> np.ndarray:
        """Create placeholder for disconnected stream"""
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, f"{stream.config.name}", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
        cv2.putText(blank, "NO SIGNAL", (100, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        cv2.putText(blank, "Retrying...", (120, 320),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        return blank
    
    def run(self, display_interval: float = 0.033):  # ~30 FPS
        """Main event loop"""
        logger.info("PhoneStreamViewer started. Press 'q' to quit, 'r' to reset, 's' to save screenshot")
        
        try:
            while self.running:
                frames = []
                for stream in self.streams:
                    frame = stream.read()
                    if frame is not None:
                        # Resize to consistent dimensions
                        frame = cv2.resize(frame, (640, 480))
                        display_frame = self._prepare_frame(stream, frame)
                        frames.append(display_frame)
                    else:
                        frames.append(self._create_placeholder_frame(stream))
                
                if frames:
                    grid = self.display.create_grid(frames)
                    self.display.display(grid)
                
                # Handle keyboard input
                key = cv2.waitKey(int(display_interval * 1000)) & 0xFF
                if key == ord('q'):
                    logger.info("Quit signal received")
                    break
                elif key == ord('r'):
                    logger.info("Reset signal received, reconnecting all streams...")
                    for stream in self.streams:
                        stream._initialize_stream()
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"screenshot_{timestamp}.png"
                    cv2.imwrite(filename, grid)
                    logger.info(f"Screenshot saved: {filename}")
        
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down PhoneStreamViewer...")
        self.running = False
        
        # Shutdown mesh network
        if self.mesh_node and self.enable_mesh:
            try:
                logger.info("Shutting down mesh network...")
                self.mesh_node.stop()
            except Exception as e:
                logger.error(f"Error stopping mesh node: {e}")
        
        for stream in self.streams:
            stream.stop()
        cv2.destroyAllWindows()
        logger.info("Shutdown complete")


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Lightweight IP Webcam reader for Android phone camera feeds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View single phone stream
  python phone_stream_viewer.py --url http://192.168.1.100:8080/video --name "Phone1"
  
  # View multiple phone streams with YOLO inference
  python phone_stream_viewer.py --urls config.json --yolo
  
  # View multiple phones from command line
  python phone_stream_viewer.py \\
    --url http://192.168.1.100:8080/video --name "Phone1" \\
    --url http://192.168.1.101:8080/video --name "Phone2" \\
    --url http://192.168.1.102:8080/video --name "Phone3"

Controls:
  q - Quit
  r - Reset/reconnect all streams
  s - Save screenshot
        """
    )
    
    parser.add_argument('--url', action='append', dest='urls', 
                       help='URL of IP Webcam stream (can be repeated)')
    parser.add_argument('--name', action='append', dest='names',
                       help='Stream name (can be repeated, paired with --url)')
    parser.add_argument('--config', type=str,
                       help='JSON file with stream configurations')
    parser.add_argument('--yolo', action='store_true', default=False,
                       help='Enable YOLO inference for all streams')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='YOLO confidence threshold (default: 0.5)')
    parser.add_argument('--retries', type=int, default=3,
                       help='Max connection retries per stream (default: 3)')
    parser.add_argument('--retry-delay', type=int, default=2,
                       help='Delay between retries in seconds (default: 2)')
    
    # Mesh network options
    parser.add_argument('--mesh', action='store_true', default=True,
                       help='Enable mesh network broadcasting (default: enabled)')
    parser.add_argument('--no-mesh', action='store_false', dest='mesh',
                       help='Disable mesh network broadcasting')
    parser.add_argument('--mesh-port', type=int, default=9999,
                       help='UDP port for mesh network (default: 9999)')
    parser.add_argument('--mesh-location', type=str, default='streaming_hub',
                       help='Location signature for mesh network (default: streaming_hub)')
    
    return parser.parse_args()


def load_config_from_json(filepath: str) -> List[StreamConfig]:
    """Load stream configurations from JSON file"""
    import json
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        configs = []
        for item in data.get('streams', []):
            config = StreamConfig(
                url=item['url'],
                name=item.get('name', 'Unnamed'),
                max_retries=item.get('max_retries', 3),
                retry_delay=item.get('retry_delay', 2),
                enable_yolo=item.get('enable_yolo', False),
                confidence_threshold=item.get('confidence_threshold', 0.5)
            )
            configs.append(config)
        
        logger.info(f"Loaded {len(configs)} stream configurations from {filepath}")
        return configs
    except Exception as e:
        logger.error(f"Failed to load config from {filepath}: {e}")
        return []


def main():
    """Main entry point"""
    args = parse_args()
    
    # Load configurations
    configs = []
    
    # Try loading from JSON config file first
    if args.config:
        configs = load_config_from_json(args.config)
    
    # Parse command-line URL/name pairs
    if args.urls:
        names = args.names if args.names else [f"Stream {i+1}" for i in range(len(args.urls))]
        for url, name in zip(args.urls, names):
            config = StreamConfig(
                url=url,
                name=name,
                max_retries=args.retries,
                retry_delay=args.retry_delay,
                enable_yolo=args.yolo,
                confidence_threshold=args.confidence
            )
            configs.append(config)
    
    # Default demo streams if no config provided
    if not configs:
        logger.info("No streams configured. Using default demo URLs.")
        configs = [
            StreamConfig(
                url="http://10.197.139.199:8080/video",
                name="Phone 1",
                max_retries=args.retries,
                retry_delay=args.retry_delay,
                enable_yolo=args.yolo,
                confidence_threshold=args.confidence
            ),
            StreamConfig(
                url="http://10.197.139.108:8080/video",
                name="Phone 2",
                max_retries=args.retries,
                retry_delay=args.retry_delay,
                enable_yolo=args.yolo,
                confidence_threshold=args.confidence
            ),
            StreamConfig(
                url="http://10.197.139.192:8080/video",
                name="Phone 3",
                max_retries=args.retries,
                retry_delay=args.retry_delay,
                enable_yolo=args.yolo,
                confidence_threshold=args.confidence
            )
        ]
    
    logger.info(f"Initializing {len(configs)} streams...")
    for config in configs:
        logger.info(f"  - {config.name}: {config.url} (YOLO: {config.enable_yolo})")
    
    if args.mesh:
        logger.info(f"Mesh network: ENABLED (port={args.mesh_port}, location={args.mesh_location})")
    else:
        logger.info("Mesh network: DISABLED")
    
    # Create viewer and run
    viewer = PhoneStreamViewer(
        configs, 
        enable_yolo=args.yolo,
        enable_mesh=args.mesh,
        mesh_port=args.mesh_port,
        mesh_location=args.mesh_location
    )
    viewer.run()


if __name__ == "__main__":
    main()
